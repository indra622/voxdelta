# VoxDelta Design Specification

- Status: Approved concept, pending written-spec review
- Date: 2026-08-18
- Codename: VoxDelta
- Subtitle: Korean Call Emotion Shift Analysis

## 1. Overview

VoxDelta is a local-first web application for analyzing Korean customer-service call recordings. It separates the speakers, identifies the customer and agent, transcribes the call, estimates the customer's emotion for each utterance, and produces an evidence-linked report of how the customer's emotional state changed over the conversation.

The distinguishing feature is not emotion classification alone. VoxDelta identifies customer-agent-customer turn triplets in which the customer's negative-emotion score changes materially after an agent response. The product calls these **emotion recovery segments** and **emotion worsening segments**. It reports an association between an agent response and the immediately following change; it does not claim that the response caused the change.

The first release is a personal portfolio project. It runs locally, accepts recorded files rather than live streams, and compares one local baseline model with one frontier API model on the same benchmark data.

## 2. Goals and Success Criteria

### Goals

1. Analyze a recorded Korean two-speaker customer-service call end to end.
2. Make every model result traceable to an audio interval and transcript excerpt.
3. Show both categorical emotion and a continuous negative-emotion trajectory.
4. Identify and explain recovery and worsening segments using adjacent conversation turns.
5. Compare a local model and a frontier API model by quality, latency, cost, and confidence.
6. Keep orchestration, artifacts, configuration, and reports local while allowing explicitly authorized API processing.

### MVP success criteria

- WAV, MP3, and M4A files between 1 and 60 minutes are accepted; the primary demo range is 5–15 minutes.
- A two-speaker call can proceed from upload to a completed dashboard without manual file conversion.
- The user can correct the customer/agent assignment before emotion analysis continues.
- Every customer utterance has audio bounds, text, seven-emotion probabilities, an operational state, negative-emotion intensity, and confidence.
- Every reported recovery or worsening segment links the preceding customer turn, the intervening agent response, and the following customer turn.
- A failed stage can be retried without repeating completed stages.
- The same report data can render in the dashboard and export as HTML or PDF.
- A benchmark run can compare the selected local and API emotion providers with reproducible inputs and recorded latency/cost metadata.
- No raw datasets, call recordings, API keys, or generated private reports are committed to Git.

## 3. Non-goals

- Real-time streaming analysis or live agent coaching
- Public web deployment, user accounts, or organization-level multi-tenant access
- Production contact-center integrations
- More than two primary speakers in the MVP
- Autonomous customer-service decisions
- Causal claims about agent behavior and customer emotion
- Building a foundation speech model from scratch
- Long-term cloud storage of audio or reports

## 4. User Experience

### 4.1 Upload and configuration

The user selects a call recording and chooses a provider profile. Before processing starts, the app shows which stages are local, which stages use an external API, whether raw audio or derived text will leave the machine, and the configured provider names.

The app validates file type, duration, decodability, channel count, and estimated processing requirements. Stereo files are inspected for channel-separated speakers. When the two channels already correspond to separate call parties, channel separation takes precedence over diarization. Mono or mixed-channel audio proceeds through VAD and diarization.

### 4.2 Processing status

The app displays a stage-based job view:

1. Validate and normalize audio
2. Detect speech and separate speakers
3. Transcribe and align timestamps
4. Assign customer and agent roles
5. Analyze customer emotion
6. Classify agent response strategy
7. Calculate customer emotion changes
8. Generate report artifacts

Each stage has `pending`, `running`, `completed`, `failed`, or `skipped` status. Completed stage artifacts are checkpointed locally. A retry starts at the failed or user-selected stage and invalidates only dependent downstream artifacts.

### 4.3 Role confirmation

Diarization produces anonymous speaker identities such as `SPEAKER_00` and `SPEAKER_01`; it does not identify business roles. VoxDelta proposes the role assignment using channel metadata, opening greetings, and transcript cues, then requires a visible confirmation step. The user can swap the roles. The confirmed mapping is stored with the job and used by all downstream stages.

### 4.4 Analysis dashboard

The completed analysis view contains:

- An audio player synchronized with the transcript and emotion timeline
- A speaker-colored transcript with utterance timestamps
- Customer emotion probabilities, operational state, confidence, and negative-emotion intensity
- A call-level summary of starting state, ending state, peak negative-emotion interval, and overall direction
- Recovery and worsening segments represented as customer-agent-customer turn triplets
- The agent response strategy for each highlighted triplet
- The selected model/provider and processing metadata
- HTML/PDF export actions

### 4.5 Model comparison

A comparison view runs or loads the same evaluation items for one local emotion provider and one frontier API emotion provider. It presents quality metrics, calibration, mean and percentile latency, observed API cost, failure count, and model/version metadata. The comparison is an evaluation feature rather than a requirement to run every production analysis twice.

## 5. Emotion Representation

### 5.1 Seven-emotion model output

Every emotion provider returns probabilities for the AI Hub-aligned labels:

- happiness
- anger
- disgust
- fear
- neutral
- sadness
- surprise

The response also contains a confidence score, provider/model identifier, and evidence metadata indicating whether audio, text, or both were used.

### 5.2 Operational call-center state

The dashboard derives four business-facing scores from the seven-emotion output:

- **satisfied:** happiness
- **stable:** neutral
- **dissatisfied:** sadness, disgust, and fear
- **escalated:** anger

Surprise is retained as a separate overlay because it can be positive or negative. If surprise is the dominant emotion and surrounding text does not resolve its valence, the operational state is shown as `uncertain` rather than forced into one of the four states. Low-confidence predictions are also shown as `uncertain`.

The local and API adapters must expose the same seven-label schema. Provider-specific labels are mapped and calibrated in the adapter rather than in the UI.

### 5.3 Negative-emotion intensity

Each provider returns a calibrated `negative_intensity` value from 0.0 to 1.0. For the local baseline, calibration is learned from the manually annotated call-center validation set using the seven-emotion probabilities as inputs. For an API provider, the adapter requests the same rubric and then applies calibration against the same validation set.

The per-utterance series is smoothed with a three-customer-turn median window. The unsmoothed value remains available for inspection.

### 5.4 Emotion change and highlighted segments

For adjacent customer-agent-customer turns, VoxDelta calculates:

`delta = next_customer_smoothed_negative_intensity - previous_customer_smoothed_negative_intensity`

- `delta <= -0.20`: recovery segment
- `delta >= +0.20`: worsening segment
- otherwise: stable segment

The initial threshold is fixed at 0.20 for reproducibility. Benchmark reports additionally show threshold-sensitivity results so a later release can revise it without silently changing historical results.

## 6. Agent Response Strategy

The intervening agent utterance in each customer-agent-customer triplet is classified into one primary strategy and zero or more secondary strategies:

- apology
- empathy or acknowledgement
- clarification question
- information or explanation
- concrete solution or next action
- policy statement or refusal
- greeting or closing
- other

The classifier uses transcript context and may use prosodic features when the configured provider supports them. The label describes the response form; it is not a quality score. Reports use phrasing such as “recovery followed an apology and solution response,” never “the apology caused recovery.”

## 7. System Architecture

### 7.1 Application shape

- **Frontend:** React and TypeScript local web interface
- **Backend:** Python FastAPI service
- **Job runner:** stage-based local worker managed by the backend
- **Metadata:** SQLite for jobs, stage status, configurations, and artifact indexes
- **Artifacts:** local filesystem, one directory per job
- **Reports:** shared report JSON rendered by both the dashboard and HTML/PDF exporter

The frontend never calls model providers directly. It calls the local backend, which enforces provider configuration, records provenance, and writes artifacts.

### 7.2 Provider interfaces

The backend isolates model implementations behind five interfaces:

- `DiarizationProvider`: audio to timed anonymous speaker segments
- `TranscriptionProvider`: audio to timestamped Korean transcript tokens or segments
- `EmotionProvider`: aligned utterance audio/text to the normalized emotion schema
- `ResponseStrategyProvider`: conversation context to agent strategy labels
- `ReportSummaryProvider`: structured analysis to a concise narrative summary

Each provider declares:

- local or remote execution
- accepted inputs
- whether raw audio is transmitted
- model and version identifier
- timeout and retry policy
- cost metadata support
- retention-policy URL or note for remote providers

The initial implementation supports one local baseline and one API implementation where comparison is required. Additional providers are outside the MVP but can be added without changing the core pipeline or UI data contracts.

### 7.3 Canonical data contracts

The pipeline uses stable internal objects:

- `AudioAsset`: source path, normalized path, duration, channels, checksum
- `SpeakerSegment`: start, end, anonymous speaker ID, overlap flag, confidence
- `Utterance`: start, end, speaker ID, confirmed role, transcript, audio slice reference
- `EmotionResult`: seven probabilities, operational state scores, intensity, confidence, provider metadata
- `ResponseStrategyResult`: primary and secondary strategies, confidence, provider metadata
- `EmotionTransition`: previous customer utterance, agent utterance, next customer utterance, delta, classification
- `AnalysisReport`: call summary, timeline, transitions, provenance, warnings, evaluation metadata

All stage outputs are versioned JSON artifacts. A stage cache key contains the input artifact hashes, stage configuration, provider, and model version.

## 8. Local Storage and Privacy

Each analysis has an opaque job ID and a local directory containing the normalized audio, stage JSON, logs, report JSON, and exports. SQLite stores only indexes and structured metadata needed to list and resume jobs.

Remote API use is permitted for raw audio, text, or both. Before a job starts, the UI displays the exact remote stages and transmitted content. The choice is recorded in the job manifest. API credentials are loaded from environment variables or the operating-system credential store and are never written to job artifacts or Git-tracked configuration.

The user can delete a job from the dashboard. Deletion removes the local audio, intermediate artifacts, exports, and database index. Provider-side retention follows the selected provider's disclosed policy; the app must not imply that local deletion removes provider-side data.

Logs exclude credentials and avoid storing full transcripts or raw provider payloads by default. Diagnostic payload capture is an explicit per-job option.

## 9. Error Handling

- Unsupported or corrupt audio is rejected before a job is created.
- Files with more than two prominent speakers are flagged as unsupported for the MVP; the user may inspect the diarization output but cannot generate a final report without selecting two roles.
- Overlapping speech is retained with an overlap flag and lowered confidence.
- A missing role confirmation pauses the pipeline rather than guessing silently.
- API rate limits, timeouts, and provider errors fail only the current stage and preserve prior artifacts.
- Provider retries use bounded exponential backoff and never exceed three automatic attempts.
- A local fallback can be selected manually after a remote-stage failure when the interface has a compatible local provider.
- Missing or low-confidence emotion results produce visible gaps or uncertainty markers; they are not replaced with neutral values.
- Report generation refuses to make a call-level claim when too few customer turns have valid emotion results.

## 10. Data Strategy

The project uses two AI Hub sources with different roles:

- The Korean consultation-speech dataset supplies call-center-domain audio and transcripts for domain inspection, ASR evaluation, and representative demonstrations.
- The Korean conversational emotion-classification dataset supplies seven-emotion supervision for the local baseline.

Dataset files remain outside Git. The repository contains download/setup instructions, checksums or manifests where licensing permits, preprocessing code, and derived metadata that does not redistribute protected content.

Because the emotion dataset is not specifically a call-center dataset, VoxDelta also creates a small gold evaluation set from permitted call-center samples. The minimum set is 200 customer utterances from at least 20 calls plus 60 customer-agent-customer transition triplets. Two annotators independently label the operational state, negative intensity on a five-point rubric, and transition class. Disagreements are adjudicated and inter-annotator agreement is reported.

## 11. Evaluation

### Component metrics

- Diarization: diarization error rate and speaker-confusion error
- ASR: Korean character error rate and word error rate
- Seven-emotion classification: macro-F1, per-class recall, and expected calibration error
- Operational state: macro-F1 including the uncertain outcome
- Negative intensity: mean absolute error and Spearman correlation against the five-point gold rubric
- Recovery/worsening detection: macro-F1 over recovery, stable, and worsening
- Agent response strategy: macro-F1 and per-class support

### System metrics

- End-to-end success rate
- Stage failure and retry count
- Wall-clock latency per stage and total
- Peak local memory usage for local providers
- API cost per audio minute and per completed call
- Percentage of results marked uncertain

Benchmark outputs record dataset split hashes, provider/model versions, configuration, and run timestamp. Calls from the same source conversation or speaker must not be split across training and evaluation sets.

## 12. Testing Strategy

- Unit tests validate label mapping, intensity calibration, delta calculation, cache keys, and report phrasing constraints.
- Contract tests run every provider adapter against canonical fixtures and verify normalized outputs.
- Pipeline tests use short synthetic fixtures to exercise checkpoints, retries, invalidation, and role swapping.
- Integration tests analyze licensed or generated two-speaker Korean samples end to end.
- Failure tests cover corrupt audio, missing credentials, API timeout, partial transcripts, overlapping speech, and unsupported speaker count.
- Snapshot tests verify that dashboard and exported reports render the same structured results.
- Evaluation tests prevent accidental train/evaluation speaker leakage.

## 13. Repository and Version Control

- Local path: `/Users/hosungmini/codes/voxdelta`
- GitHub owner: `indra622`
- Repository visibility: Private
- Default branch: `main`
- Remote name: `origin`

The design specification is the first committed project artifact. Implementation begins only after the written specification is reviewed and an implementation plan is approved.

