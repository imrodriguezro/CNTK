//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <boost/algorithm/string/predicate.hpp>

#include "CNTKLibrary.h"
#include "fileutil.h"
#include "PerformanceProfiler.h"

namespace CNTK
{
    using namespace std;

    const static std::wstring s_trainingMinibatchSource = L"TrainingMinibatchSource";

    inline bool isNumber(const std::wstring& s)
    {
        return !s.empty() &&
            find_if(s.begin(), s.end(), [](wchar_t c) { return !isdigit(c); }) == s.end();
    }

    TrainingSessionConfig::TrainingSessionConfig(
        const MinibatchSourcePtr& trainingSource,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
        size_t maxNumTrainingSamples) :
        m_withCheckpointing(false),
        m_withCrossValidation(false),
        m_withProgressPrinting(false),
        m_mbSource(trainingSource),
        m_mbSizeSchedule(minibatchSizeSchedule),
        m_inputVarToStream(inputVarToStream),
        m_maxNumTrainingSamples(maxNumTrainingSamples),
        m_crossValidationSchedule(m_mbSizeSchedule)
    {
        if (!m_mbSource)
            InvalidArgument("Training source must not be null.");

        if(m_maxNumTrainingSamples == 0)
            InvalidArgument("maxNumTrainingSamples must not be zero.");

        if(m_inputVarToStream.empty())
            InvalidArgument("inputVarToStream mapping must not be empty.");
    }

    TrainingSessionConfig& TrainingSessionConfig::Checkpointing(
        const std::wstring& checkPointFileName,
        size_t checkpointFrequencyInSamples,
        bool restoreFromCheckpointIfExists,
        bool preserveAllCheckpoints)
    {
        if (m_withCheckpointing)
            RuntimeError("Checkpointing configuration has already been specified.");

        m_checkPointFileName = checkPointFileName;
        if (m_checkPointFileName.empty())
        {
            if (checkpointFrequencyInSamples != 0 && checkpointFrequencyInSamples != std::numeric_limits<size_t>::max())
                InvalidArgument("Checkpoint file name is not allowed to be empty if checkpoint frequency is non zero.");
            if (preserveAllCheckpoints)
                InvalidArgument("Checkpoint file name is not allowed to be empty if 'preserve all checkpoints' is specified.");
            checkpointFrequencyInSamples = 0;
        }

        m_preserveAllCheckpoints = preserveAllCheckpoints;
        m_restoreFromCheckpointIfExists = restoreFromCheckpointIfExists;
        m_checkpointFrequencyInSamples = checkpointFrequencyInSamples;
        m_withCheckpointing = true;
        return *this;
    }

    TrainingSessionConfig& TrainingSessionConfig::CrossValidation(
        const MinibatchSourcePtr& crossValidationSource,
        const MinibatchSizeSchedule& crossValidationSchedule,
        size_t crossValidationFrequencyInSamples)
    {
        if (m_withCrossValidation)
            RuntimeError("Cross validation configuration has already been specified.");

        m_crossValidationSource = crossValidationSource;
        m_crossValidationSchedule = crossValidationSchedule;
        m_crossValidationFrequencyInSamples = crossValidationFrequencyInSamples;
        m_withCrossValidation = true;
        return *this;
    }

    TrainingSessionConfig& TrainingSessionConfig::ProgressPrinting(size_t progressFrequency)
    {
        if (m_withProgressPrinting)
            RuntimeError("Progress printing configuration has already been specified.");

        m_progressFrequencyInSamples = progressFrequency;
        m_withProgressPrinting = true;
        return *this;
    }

    TrainingSessionPtr CreateBasicTrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        size_t checkpointFrequencyinSamples,
        const std::wstring& checkPointFileName,
        const MinibatchSourcePtr& crossValidationSource,
        const MinibatchSizeSchedule& crossValidationSchedule,
        size_t crossValidationFrequencyInSamples,
        bool restoreFromCheckpointIfExists,
        bool saveAllCheckpoints,
        size_t maxNumberOfSamples,
        size_t progressFrequency)
    {
        fprintf(stderr, "WARNING:CreateBasicTrainingSession is deprecated and will be removed in the next beta (13)."
            "Instructions for updating:"
            "Please switch to CreateTrainingSession function and then call SetCheckpointing/SetCrossValidation/SetPrintingProgress as needed.");

        return MakeSharedObject<TrainingSession>(trainingSource,
            trainer,
            modelInputToMinibatchSourceStream,
            minibatchSizeSchedule,
            checkpointFrequencyinSamples,
            checkPointFileName,
            crossValidationSource,
            crossValidationSchedule,
            crossValidationFrequencyInSamples,
            restoreFromCheckpointIfExists,
            saveAllCheckpoints,
            maxNumberOfSamples,
            progressFrequency);
    }

    TrainingSessionPtr CreateTrainingSession(
        const TrainerPtr& trainer,
        const TrainingSessionConfig& config)
    {
        return MakeSharedObject<TrainingSession>(
            trainer,
            config);
    }

    TrainingSession::TrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& schedule,
        size_t checkpointFrequencyInSamples,
        const std::wstring& checkPointFileName,
        const MinibatchSourcePtr& crossValidationSource,
        const MinibatchSizeSchedule& crossValidationSchedule,
        size_t crossValidationFrequencyInSamples,
        bool restoreFromCheckpointIfExists,
        bool saveAllCheckpoints,
        size_t maxNumberOfSamples,
        size_t progressFrequencyInSamples) 
        : TrainingSession(trainer, 
            TrainingSessionConfig(trainingSource, schedule, modelInputToMinibatchSourceStream, maxNumberOfSamples)
            .Checkpointing(checkPointFileName, checkpointFrequencyInSamples, restoreFromCheckpointIfExists, saveAllCheckpoints)
            .CrossValidation(crossValidationSource, crossValidationSchedule, crossValidationFrequencyInSamples)
            .ProgressPrinting(progressFrequencyInSamples))
    {
    }

    TrainingSession::TrainingSession(
        const TrainerPtr& trainer,
        const TrainingSessionConfig& config) :
        m_trainer(trainer),
        m_config(config),
        m_parallelAfterSamples(0),
        m_workerRank(0),
        m_numberOfWorkers(1)
    {
        if (!trainer)
            InvalidArgument("Trainer is not allowed to be null.");

        // Let's calculate the warm up period the distributed learners may need.
        // We will take the maximum warm up period required.
        auto learners = trainer->ParameterLearners();
        m_parallelAfterSamples = 0;
        for (const auto& l : learners)
        {
            auto distributed = std::dynamic_pointer_cast<DistributedLearner>(l);
            if (distributed)
            {
                m_parallelAfterSamples = std::max(m_parallelAfterSamples, distributed->ParallelizationAfter());
                m_workerRank = distributed->GetCommunicator()->CurrentWorker().m_globalRank;
                m_numberOfWorkers = distributed->GetCommunicator()->Workers().size();
            }
        }

        // Fill-in required actions.
        if (m_config.m_withCheckpointing && m_config.m_checkpointFrequencyInSamples != 0)
            m_actions.push_back({ m_config.m_checkpointFrequencyInSamples, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&)
                {
                    SaveCheckpoint(currentIndex);

                    // enable profiler after the first checkpoint
                    // This has effect only if the profiler is globally enabled by StartProfiler()
                    Microsoft::MSR::CNTK::ProfilerEnable(true);
                    return true;
                } });

        if (m_config.m_withCrossValidation && m_config.m_crossValidationFrequencyInSamples != 0)
            m_actions.push_back({ m_config.m_crossValidationFrequencyInSamples, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor& d) { return CrossValidate(currentIndex, d); } });

        if (m_config.m_withProgressPrinting && m_config.m_progressFrequencyInSamples != 0)
            m_actions.push_back({ m_config.m_progressFrequencyInSamples, 0, 0,
                [this](size_t currentIndex, const DeviceDescriptor&) { ReportProgress(currentIndex); return true; } });
    }

    void TrainingSession::Train(const DeviceDescriptor& computeDevice)
    {
        std::unordered_map<Variable, ValuePtr> minibatch;
        bool shouldTrain = m_config.m_maxNumTrainingSamples > 0;

        // Let's try to restore if required.
        size_t restoredNumberOfSamples = 0;
        if (m_config.m_restoreFromCheckpointIfExists && !m_config.m_checkPointFileName.empty())
        {
            RestoreFromCheckpoint();
            restoredNumberOfSamples = m_trainer->TotalNumberOfSamplesSeen();
        }

        // Main train loop.
        bool earlyExit = false;
        while (shouldTrain)
        {
            // Get next minibatch.
            size_t samplesLeft = earlyExit || m_config.m_maxNumTrainingSamples <= m_trainer->TotalNumberOfSamplesSeen()
                ? 0
                : m_config.m_maxNumTrainingSamples - m_trainer->TotalNumberOfSamplesSeen();

            GetTrainingMinibatch(minibatch, samplesLeft, computeDevice);

            // Train on the minibatch
            OnMinibatchStart();
            shouldTrain = m_trainer->TrainMinibatch(minibatch, computeDevice);
            OnMinibatchEnd();

            auto profMisc = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainPost);

            // Peform actions if required.
            size_t totalNumberOfSamples = m_trainer->TotalNumberOfSamplesSeen();
            for (auto& action : m_actions)
            {
                size_t index = totalNumberOfSamples / action.frequency;
                if (index != action.currentIndex)
                {
                    bool shouldContinue = action.action(action.currentIndex, computeDevice);
                    if (!shouldContinue) // If any action wants to have early exit - we stop training.
                        earlyExit = true;

                    action.currentIndex = index;
                    action.sampleCountWhenLastCalled = totalNumberOfSamples;
                }
            }
        }

        if (restoredNumberOfSamples != m_trainer->TotalNumberOfSamplesSeen())
        {
            // Let's do all actions on the last probably a partial data at the end.
            for (auto& action: m_actions)
            {
                if (m_trainer->TotalNumberOfSamplesSeen() % action.frequency != 0 &&
                    m_trainer->TotalNumberOfSamplesSeen() != action.sampleCountWhenLastCalled)
                    action.action(action.currentIndex, computeDevice);
            }
        }

        // In case of incremental - save final checkpoint.
        // This is required only when we keep all existing checkpoints, otherwise 
        // The checkpoint was already saved with the proper name.
        if (m_config.m_withCheckpointing &&
            m_config.m_preserveAllCheckpoints &&
            !fexists(m_config.m_checkPointFileName))
            SaveFinalCheckpoint();
    }

    // TODO: Possibly expose a limiting counter on the number of samples for validation.
    bool TrainingSession::CrossValidate(size_t currentIndex, const DeviceDescriptor& computeDevice)
    {
        if (m_config.m_crossValidationSource) // Running cross validation
        {
            std::unordered_map<Variable, ValuePtr> minibatch;
            double accumulatedError = 0;
            double error = 0;
            size_t totalNumberOfSamples = 0;
            size_t numberOfMinibatches = 0;

            auto checkpoint = m_config.m_crossValidationSource->GetCheckpointState();
            size_t sampleCount = 0;
            while (GetCrossValidationMinibatch(minibatch, m_config.m_crossValidationSchedule[sampleCount], computeDevice), !minibatch.empty())
            {
                error = m_trainer->TestMinibatch(minibatch, computeDevice, sampleCount);
                accumulatedError += error;
                totalNumberOfSamples += sampleCount;
                numberOfMinibatches++;
            }

            m_config.m_crossValidationSource->RestoreFromCheckpoint(checkpoint);
            return OnCrossValidationEnd(currentIndex, accumulatedError / totalNumberOfSamples, totalNumberOfSamples, numberOfMinibatches);
        }
        else // Only invoking the callback.
        {
            return OnCrossValidationEnd(currentIndex, 0, 0, 0);
        }
    }

    inline void TrainingSession::ReportProgress(size_t currentIndex)
    {
        this->OnProgress(currentIndex);
    }

    void TrainingSession::GetTrainingMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, const DeviceDescriptor& computeDevice)
    {
        size_t workerRank = m_workerRank, numberOfWorkers = m_numberOfWorkers;

        // Check if we are operating in distributed mode.
        if (m_parallelAfterSamples > m_trainer->TotalNumberOfSamplesSeen())
        {
            numberOfWorkers = 1;
            workerRank = 0;
        }

        size_t mbSize = GetMinibatchSize();
        mbSize = std::min(mbSize, maxMbSize);
        GetNextMinibatch(m_config.m_mbSource, minibatch, mbSize, workerRank, numberOfWorkers, computeDevice);
    }

    void TrainingSession::GetCrossValidationMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, const DeviceDescriptor& computeDevice)
    {
        // TODO: Support distributed cross-validation, when TestMinibatch supports it.
        GetNextMinibatch(m_config.m_crossValidationSource, minibatch, maxMbSize, 0, 1, computeDevice);
    }

    void TrainingSession::GetNextMinibatch(const MinibatchSourcePtr& source, std::unordered_map<Variable, ValuePtr>& minibatch, size_t mbSize, size_t workerRank, size_t numberOfWorkers, const DeviceDescriptor& computeDevice)
    {
        minibatch.clear();

        if (mbSize == 0)
            return;

        auto minibatchData = source->GetNextMinibatch(0 /*numberOfSequences*/, mbSize, numberOfWorkers, workerRank, computeDevice);
        if (minibatchData.empty())
            return;

        for (auto v : m_config.m_inputVarToStream)
            minibatch.insert({ v.first, minibatchData[v.second].data });
    }

    void TrainingSession::RestoreFromCheckpoint(const std::wstring& checkpointFileName)
    {
        Dictionary externalState = m_trainer->RestoreFromCheckpoint(checkpointFileName);
        m_config.m_mbSource->RestoreFromCheckpoint(externalState[s_trainingMinibatchSource].Value<Dictionary>());
    }

    void TrainingSession::SaveCheckpoint(size_t currentIndex)
    {
        OnCheckpointStart(currentIndex);
        Dictionary externalState;
        externalState[s_trainingMinibatchSource] = m_config.m_mbSource->GetCheckpointState();

        wstring checkpointFile = m_config.m_checkPointFileName;
        if (m_config.m_preserveAllCheckpoints)
            checkpointFile += std::to_wstring(currentIndex);
        m_trainer->SaveCheckpoint(checkpointFile, externalState);
        OnCheckpointEnd(currentIndex);
    }

    void TrainingSession::SaveFinalCheckpoint()
    {
        Dictionary externalState;
        externalState[s_trainingMinibatchSource] = m_config.m_mbSource->GetCheckpointState();
        m_trainer->SaveCheckpoint(m_config.m_checkPointFileName, externalState);
    }

    // Restores from a m_checkPointFileName file.
    // If the file path exists - simply restores from the corresponding file.
    // If the file path does not exist - looks into directory where the file is
    // located and picks up the file with the largest N among <m_checkPointFileName>N files,
    // Where N is some positive integer.
    void TrainingSession::RestoreFromCheckpoint()
    {
        assert(!m_config.m_checkPointFileName.empty());
        auto checkpoint = m_config.m_checkPointFileName;

        // Make sure the intermediate directories exist, so no need for further checks.
        msra::files::make_intermediate_dirs(checkpoint);

        size_t pos = checkpoint.find_last_of(L"\\/");
        wstring parent;
        wstring fileName;
        if (pos == wstring::npos)
        {
            parent = L"..";
            fileName = checkpoint;
        }
        else
        {
            parent = checkpoint.substr(0, pos);
            fileName = checkpoint.substr(pos);
        }

        std::wstring restoreFile;
        if (fexists(checkpoint))
        {
            restoreFile = checkpoint;
        }
        else
        {
            // let's check whether there are other possible candidates to restore from.
            int maxValue = -1;
            std::vector<std::wstring> files = msra::files::get_all_files_from_directory(parent);

            for (auto f : files)
            {
                if (!boost::starts_with(f, fileName))
                {
                    continue;
                }

                auto suffix = f.substr(fileName.size());
                if (!isNumber(suffix) || !fexists(parent + L"/" + f + L".ckp"))
                {
                    continue;
                }

                auto expectedNumber = msra::strfun::utf8(suffix);
                char* tmp = nullptr;
                int value = strtol(expectedNumber.c_str(), &tmp, 10);
                if (tmp != expectedNumber.c_str() + expectedNumber.size())
                    continue;

                if (value > maxValue)
                {
                    // Found a better candidate.
                    maxValue = value;
                    restoreFile = parent + L"/" + f;
                }
            }
        }

        if (restoreFile.empty()) // Nothing to restore.
            return;

        // TODO: Should have proper loggin instead.
        fprintf(stderr, "Restoring training session from the checkpoint '%ls'\n", restoreFile.c_str());

        this->RestoreFromCheckpoint(restoreFile);

        // Recalculate actions indicies.
        size_t totalNumberOfSamples = m_trainer->TotalNumberOfSamplesSeen();
        for (auto& action : m_actions)
        {
            action.currentIndex = totalNumberOfSamples / action.frequency;
            action.sampleCountWhenLastCalled = totalNumberOfSamples - totalNumberOfSamples % action.frequency;
        }
    }
}
