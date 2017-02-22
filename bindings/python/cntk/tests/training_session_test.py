# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import math
import re
import numpy as np
from .. import Function
from ..ops import times, sequence, as_block, element_select
from ..ops.tests.ops_test_utils import cntk_device
from ..utils import one_hot
from ..trainer import *
from ..training_session import *
from ..learner import *
from .. import cross_entropy_with_softmax, classification_error, parameter, \
        input_variable, times, plus, reduce_sum, Axis, cntk_py
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, FULL_DATA_SWEEP, INFINITELY_REPEAT
import pytest

input_dim = 69

ctf_data = '''\
0	|S0 3:1 |# <s>	|S1 3:1 |# <s>
0	|S0 4:1 |# A	|S1 32:1 |# ~AH
0	|S0 5:1 |# B	|S1 36:1 |# ~B
0	|S0 4:1 |# A	|S1 31:1 |# ~AE
0	|S0 7:1 |# D	|S1 38:1 |# ~D
0	|S0 12:1 |# I	|S1 47:1 |# ~IY
0	|S0 1:1 |# </s>	|S1 1:1 |# </s>
2	|S0 60:1 |# <s>	|S1 3:1 |# <s>
2	|S0 61:1 |# A	|S1 32:1 |# ~AH
3	|S0 60:1 |# <s>	|S1 3:1 |# <s>
3	|S0 61:1 |# A	|S1 32:1 |# ~AH
4	|S0 60:1 |# <s>	|S1 3:1 |# <s>
4	|S0 61:1 |# A	|S1 32:1 |# ~AH
5	|S0 60:1 |# <s>	|S1 3:1 |# <s>
5	|S0 61:1 |# A	|S1 32:1 |# ~AH
6	|S0 60:1 |# <s>	|S1 3:1 |# <s>
6	|S0 61:1 |# A	|S1 32:1 |# ~AH
7	|S0 60:1 |# <s>	|S1 3:1 |# <s>
7	|S0 61:1 |# A	|S1 32:1 |# ~AH
8	|S0 60:1 |# <s>	|S1 3:1 |# <s>
8	|S0 61:1 |# A	|S1 32:1 |# ~AH
9	|S0 60:1 |# <s>	|S1 3:1 |# <s>
9	|S0 61:1 |# A	|S1 32:1 |# ~AH
10	|S0 60:1 |# <s>	|S1 3:1 |# <s>
10	|S0 61:1 |# A	|S1 32:1 |# ~AH
'''

def mb_source(tmpdir, fileprefix, epoch_size=FULL_DATA_SWEEP):
    ctf_file = str(tmpdir/(fileprefix + '2seqtest.txt'))
    with open(ctf_file, 'w') as f:
        f.write(ctf_data)

    mbs = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
        features  = StreamDef(field='S0', shape=input_dim,  is_sparse=True),
        labels    = StreamDef(field='S1', shape=input_dim,  is_sparse=True)
        )), 
        randomize=False, epoch_size=epoch_size)
    return mbs

def create_sample_model(device):
    in1 = input_variable(shape=(input_dim,))
    labels = input_variable(shape=(input_dim,))
    p = parameter(shape=(input_dim,), init=10, device=device)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    lr_per_sample = learning_rate_schedule([0.3, 0.2, 0.1, 0.0], UnitType.sample)
    learner = sgd(z.parameters, lr_per_sample)
    trainer = Trainer(z, (ce, errs), [learner])
    return (trainer, in1, labels)

class MockProgressPrinter:
    def __init__(self, trainer, expected_cv=None, epoch_summary_counter=0):
        self.epoch_summary_counter = epoch_summary_counter 
        self.trainer = trainer        
        self.expected_cv = expected_cv
        self.minibatch_info = []

    def update_with_trainer(self, trainer, with_metric):
        self.minibatch_info.append(
            (self.epoch_summary_counter,
             (trainer.previous_minibatch_loss_average,
              trainer.previous_minibatch_evaluation_average,
              trainer.previous_minibatch_sample_count,
              trainer.total_number_of_samples_seen)))

    def epoch_summary(self, with_metric):
        self.epoch_summary_counter += 1

    def log(self, msg):
        results = re.findall("Cross Validation \[(.+?)\]: Minibatch\[.+?\]: errs = (.+?)% \* (\d+)", msg)
        assert(len(results) == 1)
        validation_index = int(results[0][0]) - 1
        assert(self.expected_cv[validation_index][0] == float(results[0][1]))
        assert(self.expected_cv[validation_index][1] == int(results[0][2]))

def test_session_sanity_check(tmpdir, device_id):

    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training")

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    c = TrainingSessionConfig(mb_source=mbs, input_vars_to_streams=input_map,
                              mb_size_schedule=minibatch_size_schedule(4))

    training_session(trainer=t, config=c).train(device)

def test_session_max_samples(tmpdir, device_id):
    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    c = TrainingSessionConfig(mb_source=mbs, input_vars_to_streams=input_map,
                              mb_size_schedule=minibatch_size_schedule(4), max_samples=20)

    training_session(trainer=t, config=c).train(device)
    assert(t.total_number_of_samples_seen == 21)

def test_session_cross_validation_at_end(tmpdir, device_id):
    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)
    mbs1 = mb_source(tmpdir, "cv")

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    printer = MockProgressPrinter(t, expected_cv=[[92, 25]])
   
    c=TrainingSessionConfig(mb_source=mbs, mb_size_schedule=minibatch_size_schedule(4), 
                            input_vars_to_streams=input_map, max_samples=20) \
        .progress_printing(printer) \
        .cross_validation(source=mbs1)
    training_session(trainer=t, config=c).train(device)

    assert(t.total_number_of_samples_seen == 21)

def test_session_cross_validation_3_times(tmpdir, device_id):
    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)
    mbs1 = mb_source(tmpdir, "cv")

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    printer = MockProgressPrinter(t, expected_cv=[[92, 25], [92, 25], [92, 25]])
    c = TrainingSessionConfig(mb_source=mbs, mb_size_schedule=minibatch_size_schedule(4), 
                              input_vars_to_streams=input_map,
                              max_samples=60) \
        .cross_validation(source=mbs1, frequency=20, schedule=minibatch_size_schedule(2)) \
        .progress_printing(printer) 
    training_session(trainer=t, config=c).train(device)

    assert(t.total_number_of_samples_seen == 61)


def test_session_cross_validation_3_times_checkpoints_2_save_all(tmpdir, device_id):
    from os import listdir
    from os.path import isfile, join

    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)
    mbs1 = mb_source(tmpdir, "cv")

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    test_dir = str(tmpdir)

    printer = MockProgressPrinter(t, expected_cv=[[92, 25], [92, 25], [92, 25]])

    c = TrainingSessionConfig(mb_source=mbs, mb_size_schedule=minibatch_size_schedule(4), 
                              input_vars_to_streams = input_map, max_samples = 60) \
        .checkpointing(frequency = 35, filename=str(tmpdir/"checkpoint_save_all"),
                       preserve_all=True) \
        .cross_validation(source = mbs1, frequency = 20) \
        .progress_printing(printer)
    training_session(trainer=t, config=c).train(device) 

    candidates = [f for f in listdir(test_dir) if isfile(join(test_dir, f)) and f.startswith("checkpoint_save_all")]

    assert("checkpoint_save_all0" in candidates)
    assert("checkpoint_save_all0.ckp" in candidates)

    assert("checkpoint_save_all1" in candidates)
    assert("checkpoint_save_all1.ckp" in candidates)

    assert("checkpoint_save_all" in candidates)
    assert("checkpoint_save_all.ckp" in candidates)

def test_session_progress_print(tmpdir, device_id):
    from os import listdir
    from os.path import isfile, join

    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    test_dir = str(tmpdir)

    printer = MockProgressPrinter(t)

    c = TrainingSessionConfig(mb_source = mbs, mb_size_schedule=minibatch_size_schedule(4), 
                              input_vars_to_streams = input_map, max_samples = 60) \
        .progress_printing(printer, frequency=10)
    training_session(trainer=t, config=c).train(device)

    assert(printer.epoch_summary_counter == 6)


def test_session_restart_from_checkpoint(tmpdir, device_id):
    from os import listdir
    from shutil import copyfile
    from os.path import isfile, join

    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    test_dir = str(tmpdir)
    printer = MockProgressPrinter(t)

    c = TrainingSessionConfig(mb_source = mbs, mb_size_schedule=minibatch_size_schedule(4), 
                              input_vars_to_streams = input_map, 
                              max_samples = 60) \
        .checkpointing(frequency = 35, filename = str(tmpdir/"restart_from_checkpoint"),
                       preserve_all = True) \
        .progress_printing(printer, frequency = 35)
    training_session(trainer=t, config=c).train(device)

    candidates = [f for f in listdir(test_dir) if isfile(join(test_dir, f)) and f.startswith("restart_from_checkpoint")]

    assert("restart_from_checkpoint0" in candidates)
    assert("restart_from_checkpoint0.ckp" in candidates)

    assert("restart_from_checkpoint1" in candidates)
    assert("restart_from_checkpoint1.ckp" in candidates)

    assert("restart_from_checkpoint" in candidates)
    assert("restart_from_checkpoint" in candidates)

    # rename 0 checkpoint
    copyfile(str(tmpdir/"restart_from_checkpoint0"), str(tmpdir/"saved_restart_from_checkpoint0"))
    copyfile(str(tmpdir/"restart_from_checkpoint0.ckp"), str(tmpdir/"saved_restart_from_checkpoint0.ckp"))

    # remove everything except for 0
    for f in candidates:
        os.remove(str(tmpdir/f))

    # restoring from a particular checkpoint and again save everything from the second epoch
    printer2 = MockProgressPrinter(t, epoch_summary_counter=1)

    c = TrainingSessionConfig(mb_source=mbs, mb_size_schedule=minibatch_size_schedule(4),
                              input_vars_to_streams = input_map, 
                              max_samples=60) \
        .checkpointing(frequency=35, filename = str(tmpdir/"saved_restart_from_checkpoint0"),
                           restore=True, preserve_all=True) \
        .progress_printing(printer=printer2, frequency = 35)

    training_session(trainer=t, config=c).train(device)

    candidates = [f for f in listdir(test_dir) if isfile(join(test_dir, f)) and f.startswith("saved_restart_from_checkpoint0")]

    assert("saved_restart_from_checkpoint00" not in candidates)
    assert("saved_restart_from_checkpoint00.ckp" not in candidates)

    assert("saved_restart_from_checkpoint01" in candidates)
    assert("saved_restart_from_checkpoint01.ckp" in candidates)

    assert("saved_restart_from_checkpoint0" in candidates)
    assert("saved_restart_from_checkpoint0.ckp" in candidates)

    # remove information about 0 epoch from the mock printer
    first_run_minibatch_info = [i for i in printer.minibatch_info if i[0] != 0]
    
    assert(first_run_minibatch_info == printer2.minibatch_info)


def test_session_cv_callback_3_times(tmpdir, device_id):

    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    counter = [0]
    def cv_callback(index, average_error, num_samples, num_mb):
        assert(counter[0] == index)
        assert average_error == 0
        assert num_samples == 0
        assert num_mb == 0
        counter[0] += 1
        return True

    c = TrainingSessionConfig(mb_source=mbs, mb_size_schedule=minibatch_size_schedule(4), 
                              input_vars_to_streams=input_map, max_samples=60) \
        .cross_validation(frequency=20, callback=cv_callback)

    training_session(trainer=t, config=c).train(device)
    assert counter == [3]

def test_session_cv_callback_with_cross_validation_3_times(tmpdir, device_id):
    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)
    cv_mbs = mb_source(tmpdir, "cv")

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    def cv_callback(index, average_error, num_samples, num_mb):
        initial_position = cv_mbs.current_position
        total_error = 0
        while True:
           mb = cv_mbs.next_minibatch(2, input_map=input_map)
           if not mb: break
           mb_error = t.test_minibatch(mb, device)
           total_error += mb_error * mb[label].num_samples

        total_samples = 25 # Please see input data
        assert((total_error * 100) / total_samples == 92)
        cv_mbs.current_position = initial_position
        return True

    c = TrainingSessionConfig(mb_source=mbs, mb_size_schedule=minibatch_size_schedule(4), 
                              input_vars_to_streams=input_map, max_samples=60) \
        .cross_validation(frequency=20, callback=cv_callback)

    training_session(trainer=t, config=c).train(device)
    assert(t.total_number_of_samples_seen == 61)


def test_session_cv_callback_early_exit(tmpdir, device_id):

    device=cntk_device(device_id)
    (t, feature, label) = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", epoch_size=INFINITELY_REPEAT)

    input_map = {
        feature : mbs.streams.features,
        label : mbs.streams.labels
    }

    counter = [0]
    def cv_callback(index, average_error, num_samples, num_mb):
        assert(counter[0] == index)
        assert average_error == 0
        assert num_samples == 0
        assert num_mb == 0
        counter[0] += 1
        return counter[0] < 1 

    c = TrainingSessionConfig(mb_source=mbs, mb_size_schedule=minibatch_size_schedule(4), 
                              input_vars_to_streams=input_map, 
                              max_samples=60) \
        .cross_validation(frequency=20, callback=cv_callback)

    training_session(trainer=t, config=c).train(device)
    assert counter == [1]
