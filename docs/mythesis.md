Real-Time ECG Signal Analysis Using Liquid
Attention Neural Network
AIMLCZG628T: Dissertation
by
Brajesh Kumar Sethi
2023ac
Dissertation work carried out at
Version 1, Bangalore
Submitted in partial fulfilment of M. Tech In AI/ML degree programme
Under the Supervision of
Rohit Vincent
(Version 1, Bangalore)
BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE
PILANI (RAJASTHAN)
December 2025
Real-Time ECG Signal Analysis Using Liquid
Attention Neural Network
AIMLCZG628T: Dissertation
by
Brajesh Kumar Sethi
2023AC
Dissertation work carried out at
Version 1, Bangalore
Submitted in partial fulfilment of M. Tech In AI/ML degree programme
Under the Supervision of
Rohit Vincent
Version 1, Bangalore
BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE
PILANI (RAJASTHAN)
December 2025
Contents
1: Introduction, Problem Definition, and Research Context
1.1 Project Overview and Industrial Context
1.2 Problem Statement: Limitations of Conventional Deep Learning in ECG
1.3 Research Objectives and Scope of Work
2: Theoretical Foundations and Comparative Literature Review
2.1 The Mathematical Core of Liquid Neural Networks (LNNs)
2.2 Mechanism of Attention Networks and Hybrid Synergy
2.3 Review of Baseline Models and Research Positioning
3: Novelty and Proposed Architecture: The Liquid-Attention Neural Network (LA-NN)
3.1 Specific Novelty and Contribution Statement
3.2 Custom ECG Preprocessing Pipeline
3.3 Detailed LA-NN Hybrid Architecture Modules..............................................................
4: Implementation Status and Code-Level Progress (The 50% Milestone)
4.1 Current Development Milestones
4.2 Data Handling and AAMI Classification System
4.3 Training Loop Configuration and Optimization Strategy
5: Rigorous Experimental and Evaluation Design
5.1 Multi-Dimensional Validation Tasks
5.2 Comprehensive Performance Metrics........................................................................
5.3 Comparative Methodology and Baseline Analysis.....................................................
6: Clinical Relevance and Decision Logic Layer
6.1 Operational Mapping: From Prediction Scores to Triage Rules
6.2 Mathematical Justification for Confidence Handling
6.3 Interpretability Analysis of Model Dynamics
7: Conclusion, Future, and Formal Reviewer Remarks
7.1 Summary of Mid-Semester Achievements.................................................................
7.2 Proposed Plan for Remaining Work
Literature References...........................................................................................................
ABSTRACT

This project adder sses the critical need for highly adaptive and computationally
efficient models capable of performing real-time analysis of non-stationary
Electrocardiogram (ECG) signals for cardiac anomaly detection. Traditional deep
learning architectures, such as Convolutional Neural Networks (CNNs) and Long
Short-Term Memory (LSTMs), struggle due to their fixed computational graphs
and high latency, rendering them unsuitable for deployment on resource-
constrained wearable devices. The research proposes a novel Liquid-Attention
Neural Network (LA-NN) , a hybrid architecture combining the adaptive,
continuous-time dynamics of Liquid Neural Networks (LNNs) with the global
context recognition capabilities of a Multi-head Self-Attention mechanism. The
primary objectives include developing this optimized architecture from scratch,
validating its performance against four state-of-the-art baselines (CNN, LSTM,
standard LNN, Transformer), and rigorously evaluating its efficiency using real-
time metrics, including latency and memory footprint. This report details the
successful completion of the architectural design and the custom preprocessing
pipeline, confirming the project’s progress towards the 50% implementation
milestone.
______________________
Signature of the Student
Name: Brajesh Kumar Sethi
Date:17- 12 - 2025
Place: Odisha, India
_______________________
Signature of the Supervisor
Name: Rohit Vincent
Date: 17 - 12 - 2025
Place: Dublin, Ireland
1: Introduction, Problem Definition, and Research Context
1.1 Project Overview and Industrial Context
The dissertation, "Real-Time ECG Signal Analysis Using Liquid Neural Networks," falls
within the broad academic areas of Artificial Intelligence, Machine Learning, Biomedical
Signal Processing, and Edge Computing. This necessity for lightweight, low-latency
analysis is fundamental to the project’s success, driving specialized architectural choices
and stringent performance metrics.
The core focus of the project is on hybrid neural architecture designed for real-time
adaptive learning, specifically targeting continuous and non-stationary ECG signals for
accurate anomaly detection, such as arrhythmia, tachycardia, and heart block.

1.2 Problem Statement: Limitations of Conventional Deep Learning in ECG
Cardiovascular diseases remain a leading cause of mortality worldwide, making
continuous and accurate monitoring of ECG signals crucial for early detection. While
traditional deep learning models have achieved promising results in ECG classification,
their efficacy in real-world continuous monitoring scenarios is compromised by several
critical limitations:

Static Architecture: Models like CNNs and LSTMs possess fixed computational
graphs that fundamentally lack the capability to adapt dynamically to rapidly
changing signal characteristics or patient-specific variations. This rigidity is a
significant drawback when analyzing highly non-stationary biomedical time-series
data.
Real-Time Processing Limitations: Conventional models often entail a high
computational overhead, making their deployment challenging on the low-power,
resource-constrained wearable devices essential for continuous monitoring.
Non-Stationary Data Handling: Fixed models struggle to maintain performance
when faced with temporal drift in ECG patterns or significant inter-patient
variability, necessitating frequent retraining or complex normalization schemes.
Limited Temporal Context: Standard recurrent and convolutional networks
frequently fail to effectively capture both short-term dependencies (e.g., precise
morphology of a QRS complex) and long-term dependencies (e.g., patterns
indicative of heart block spanning many beats) simultaneously.
These limitations demonstrate a clear causal link: the inherent non-stationary complexity
of continuous ECG data demands an adaptive, dynamic model structure that
conventional fixed-weight architectures cannot provide. The research is motivated by this
necessity to develop an adaptive system that offers instant responsiveness to temporal
changes while maintaining computational efficiency for edge deployment.
1.3 Research Objectives and Scope of Work
The project is guided by six primary objectives:

Comprehensive Literature Analysis: Analyze the limitations of existing models
(CNN, LSTM, RNN, GRU, Transformer-based approaches) and identify gaps in
current LNN applications for biomedical signal processing.
Deep Understanding of Liquid Neural Networks: Study the mathematical
foundations governing LNN behavior, analyze the difference between liquid and
standard recurrent neurons, and investigate continuous-time dynamics versus
discrete-time processing.
Novel Hybrid Architecture Development: Design the Liquid-Attention Neural
Network (LA-NN), combining LNN layers, a multi-head self-attention mechanism,
and residual connections for optimized gradient flow.
Implementation and Optimization: Develop a custom ECG preprocessing
pipeline, implement the hybrid LA-NN model from scratch, and optimize it for edge
deployment (e.g., model compression and quantization).
Comprehensive Performance Evaluation: Compare LA-NN against four
baseline models (CNN, LSTM, standard LNN, Transformer) using rigorous
metrics, including accuracy, F1-score, inference time, and memory footprint.
Real-Time Validation: Validate the model on the PhysioNet MIT-BIH Arrhythmia
Database and the PTB Diagnostic ECG Database and assess its feasibility for
wearable device deployment.
The scope of work encompasses six concrete deliverables necessary for dissertation:
Custom preprocessing pipeline implementation.
LA-NN hybrid architecture design and implementation from scratch.
Training on PhysioNet MIT-BIH and PTB databases.
Performance comparison with 4 baseline models.
Real-time inference testing and edge deployment feasibility analysis.
Interpretability analysis of attention weights and liquid neuron dynamics.
2: Theoretical Foundations and Comparative Literature Review
2.1 The Mathematical Core of Liquid Neural Networks (LNNs)
The foundation of Liquid Neural Networks lies in their modeling as continuous-time Recurrent
Neural Networks (RNNs), governed by Ordinary Differential Equations (ODEs). This ODE-
based structure allows the network to process time-series data by responding instantly to
changes in the input signal, overcoming the fixed computational graph limitation of traditional
RNNs.^

Specifically, the core building block, the Liquid Time-constant (LTC) Cell, implements a
dynamic system where the time-constant (τ) is not static but is continuously computed based
on the current input and hidden state. The ODE describing the hidden state evolution (h) is
given by:

𝑑ℎ
𝑑𝑡=
−ℎ+𝑓(.)
τ^
The crucial mechanism is the calculation of τ which is defined as:

τ = τmin + (τmax - τmin) x σ (Linear (Input)+Linear (Hidden)+Bias)
Here, σ represents the sigmoid function, which bounds τ between a minimum (τmin) and
maximum (τmax) time constant (0.1 and 10.0, respectively, in the current implementation). This
continuous-time adaptation allows the neuron's effective memory horizon to change
instantaneously based on incoming data. This dynamic mechanism is the fundamental
innovation that enables superior robustness when analyzing non-stationary ECG signals,
directly addressing and overcoming the "Static Architecture" challenge identified in the
problem statement.^

2.2 Mechanism of Attention Networks and Hybrid Synergy
Attention Mechanisms enhance temporal feature extraction by dynamically weighting different
time steps based on their importance for a given classification task. By implementing a Multi-
head Self-Attention mechanism, the network can process sequences and determine global
dependencies in the signal. This is achieved by projecting the input into Query, Key, and Value
vectors and using a scaled dot product to calculate the relevance scores.^

The synergy between LNNs and Attention mechanisms within the proposed LA-NN
architecture is designed to capture the full spectrum of temporal dependencies in ECG data.
LNNs inherently handle the local, high-frequency adaptability required for processing the
fine-grained morphological details of heartbeats (e.g., detecting variations in QRS complexes).
Conversely, the Attention mechanism contributes to the global, long-range context
necessary for accurately diagnosing cardiac events that span multiple segments, such as
sustained tachycardia or heart block. This hybrid approach leverages the best features of both
architectures: continuous-time adaptation for signal fidelity and contextual weighting for
comprehensive diagnosis.

2.3 Review of Baseline Models and Research Positioning
To rigorously evaluate the LA-NN's performance, the project requires comparison against four
distinct deep learning architectures: Convolutional Neural Networks (CNN), Long Short-Term
Memory (LSTM), a Standard Liquid Neural Network (Standard LNN), and Transformer
architecture.^

CNNs are effective for local feature extraction, and LSTMs excel at sequential modeling, but
both suffer from the limitations related to static architecture and computational overhead in
real-time environments. The comparison against the Standard LNN baseline is vital because
it isolates the performance gains specifically attributable to the integration of the Attention
mechanism into the continuous-time framework. Furthermore, comparing the LA-NN against
the full Transformer architecture will be essential to demonstrate the computational advantage
specifically reduced latency and memory footprint achieved by using the compact, ODE-based
LNN core instead of the typically heavier, purely attention-based architecture. The goal is not
merely to achieve superior accuracy but to demonstrate that LA-NN occupies the optimal
position in the efficiency-accuracy trade-off space, making it a viable solution for Edge AI
deployment.

3: Novelty and Proposed Architecture: The Liquid-Attention Neural Network (LA-NN)
3.1 Specific Novelty and Contribution Statement
The following points explicitly define the unique contributions of this research project:
1. Novel Hybrid Architecture for ECG: The LA-NN is the first documented hybrid
architecture to combine continuous-time LNN layers with a Multi-Head Self-
Attention mechanism specifically optimized for real-time cardiac anomaly
detection in non-stationary ECG signals. This integrated design strategically
leverages the adaptive temporal processing of LNNs and the global feature
weighting power of Attention.
2. Optimized Edge AI Design Parameters: The architecture's hyperparameter
selection, including a reduced hidden dimension (dmodel = 4 8) and a minimal
numerical integration complexity (ODE_STEPS=3), is an explicit design choice
aimed at minimizing computational complexity and latency. This focus on
compactness is a direct contribution toward achieving edge deployment feasibility
on resource-constrained devices.
3. Inherent Interpretability Focus: The architecture is designed with built-in
mechanisms for interpretation. The project explicitly includes an interpretability
analysis of both the dynamic attention weights and the internal liquid neuron time
constants. This focus on transparency is a critical requirement for medical
decision-making and moves the system beyond a black-box classifier.
3.2 Custom ECG Preprocessing Pipeline
The successful real-time analysis of ECG signals necessitates a robust and custom
preprocessing pipeline to handle the inherent noise and variability in physiological
signals. The current implementation utilizes the wfdb library and is designed around
the PhysioNet MIT-BIH Arrhythmia Database.^
The pipeline involves several key steps: noise filtration (e.g., baseline wander
removal), R-peak detection, and segmentation. The system employs a standard
sampling rate of 360 Hz. For heartbeat analysis, the signals are segmented using a
fixed WINDOW_SIZE of 180 time points, which corresponds to 90 points before and
90 points after the detected R-peak. This standardized segmentation approach
ensures that each heartbeat segment provided to the LA-NN model is centered
consistently, allowing the network to focus its adaptive capabilities on the
morphological features relevant for cardiac condition detection.
3.3 Detailed LA-NN Hybrid Architecture Modules..............................................................
The LA-NN architecture is implemented using the PyTorch framework, leveraging
custom components for the Liquid Time-Constant (LTC) cells. The architecture
processes the segmented ECG time-series data (Input Dimension = 1) and consists
of the following interconnected modules:
Input and LNN Encoder Block
The input signal (180 time steps, 1 feature) is fed into the LNN Encoder Block. This
block comprises two LNN layers (NUM_LNN_LAYERS=2), utilizing the
LiquidTimeConstantCell. The hidden dimension (HIDDEN_DIM) is set to 48. This
value was strategically chosen to be lower than typical deep learning models,
reinforcing the focus on lightweight architecture for edge applications. The LNN layers
process the sequential data, leveraging the ODE-based dynamics where the internal
calculations use an integration step size of dt = 1.0 /ODE_STEPS, with ODE_STEPS
set to 3.^
Multi-Head Attention Block
The sequence of latent vectors output by the LNN Encoder is then processed by the
Multi-Head Attention Block. This standard pre-norm Transformer architecture includes
a MultiHeadAttention layer with 4 heads (NUM_ATTENTION_HEADS=4) and a Feed-
Forward network. Crucially, the implementation utilizes PositionalEncoding to ensure
that the attention mechanism retains the critical temporal order information, which is
paramount for correct rhythm analysis.^
Output and Residual Connections
The final aggregated latent vector is passed through a classification layer, which maps
the features to the 5 AAMI classification classes. The architecture also incorporates
Residual Connections throughout the block structures. These connections are
essential for optimizing the gradient flow during backpropagation, especially in deeper
hybrid networks, ensuring stable and effective training.
The key architectural specifications defining the LA-NN's compact design are
summarized below:
Table 1: LA-NN Architectural Specifications (Major Model Parameters)
Component Conceptual
Function
Implementation
Parameter/Value
Rationale
Input Signal Standardized
Heartbeat Segment
180 Points 360 Hz Aligned with
standard MIT-BIH
processing^
Hidden Dimension
(dmodel)
Latent Feature Size 48 Optimization for
speed and low
memory footprint^
LNN Layers Adaptive Sequence
Encoding
2 Depth optimized for
edge devices^
ODE Steps Numerical
Integration Fidelity
3 Balances
calculation speed
against numerical
accuracy^
Time Constant
Range (τ)
Adaptivity Limits 0.1(τmin)
to 10.0τmax
Defines the range of
temporal memory
integration^
Attention
Mechanism
Global
Context/Weighting
Multi-Head Self-
Attention (4 Heads)
Captures long-
range dependencies
in the signal^
Output Classes Clinical Task
Definition
5 AAMI Classes ('N',
'S', 'V', 'F', 'Q')
Standardized
arrhythmia
classification^
4: Implementation Status and Code-Level Progress (The 50% Milestone)
4.1 Current Development Milestones
The project has achieved the 50% completion milestone required for the mid-semester
review. As evidenced by the implementation code, the core components of the solution
have been successfully defined and built. This includes the creation of the custom
preprocessing pipeline, the complete design and implementation of the hybrid LA-NN
architecture structure from scratch, and the configuration of the training loop skeleton.
The subsequent phase of the project, which is currently pending, involves the
execution of full comparative testing, comprehensive model optimization (quantization
and compression), and the finalization of the clinical decision logic layer.^
4.2 Data Handling and AAMI Classification System
The implementation is currently focused on the PhysioNet MIT-BIH Arrhythmia
Database, which is automatically downloaded and segmented during the data
ingestion phase. This database contains a multitude of annotation symbols, requiring
robust data mapping logic to ensure clinically relevant classification. The task is
framed as a multi-class classification problem using the standard 5 AAMI (Association
for the Advancement of Medical Instrumentation) classes:
● 'N': Normal (non-ectopic beat)
● 'S': Supraventricular ectopic beat
● 'V': Ventricular ectopic beat
● 'F': Fusion beat
● 'Q': Unknown/Unclassified beat
An ANNOTATION_MAP is utilized to group the numerous raw annotations into these
five clinically standardized categories, which is crucial for producing results
comparable to established medical literature.^
4.3 Training Loop Configuration and Optimization Strategy
The initial training runs utilize a BATCH_SIZE of 64, selected to balance training
stability with efficient GPU utilization.^ The LEARNING_RATE is set to 15 x 10 -^4 along
with a WEIGHT_DECAY of 1 x 10 -^4 The data is split into 70% for training, 15% for
validation, and the remaining 15% reserved for final testing.^
A critical component of the project is the optimization for edge deployment. This
mandates techniques that minimize computational demand while preserving accuracy.
The plan includes two primary optimization steps:
1. Model Compression (Pruning): Reducing redundancy in the LA-NN's weight
parameters.
2. Quantization: Converting the model's parameters from standard 32-bit floating-
point precision to lower precision (e.g., 8-bit integers) after training. This
optimization directly reduces the Memory Footprint and significantly enhances
Inference Time (Latency) on resource-constrained embedded hardware,
ensuring the model meets the stringent requirements for real-time Edge AI.
Table 2: Implementation Parameters and Training Status

Parameter Value Status Next Steps
Data Split 70% Train, 15%
Validation, 15% Test
Completed Full data preparation
for PTB Diagnostic
ECG Database^
Optimizer PyTorch Optim (e.g.,
Adam/AdamW)
Completed Fine-tuning and
scheduling (e.g.,
Cosine Annealing)^
Batch Size 64 Completed Verification of batch
size impact on GPU
efficiency^
Architecture
Definition
LA-NN (LNN +
Attention)
Completed Optimization via
quantization and
compression^
Baseline Models CNN, LSTM, LNN,
Transformer
Pending Implementation and
standardized training^
5: Rigorous Experimental and Evaluation Design
5.1 Multi-Dimensional Validation Tasks
The comprehensive validation of LA-NN will proceed across two distinct physiological
databases to demonstrate model generalizability and robustness:
1. PhysioNet MIT-BIH Arrhythmia Database: Primarily used for high-speed, beat-
level arrhythmia classification.
2. PTB Diagnostic ECG Database: Used for more complex diagnostic tasks,
challenging the model’s ability to generalize across different conditions and
recording standards.
The evaluation will focus on the LA-NN’s ability to perform cardiac anomaly
detection in continuous time-series data. A critical validation task involves testing the
model’s adaptability to patient-specific variations and noise conditions. This testing is
designed to quantitatively prove the superiority of the adaptive LNN dynamics over the
static computational graphs of the baseline models when faced with real-world
physiological variability.
5.2 Comprehensive Performance Metrics........................................................................
The evaluation will utilize a multi-dimensional metric matrix categorized as follows:
Classification and Reliability Metrics
These metrics focus on clinical safety and reliability, especially given the typically
imbalanced nature of arrhythmia datasets:
● Accuracy, Precision, Recall, and F1-Score: Standard metrics for overall
performance.
● Sensitivity (Recall): Defined as the true positive rate. High sensitivity is
paramount in cardiac monitoring to minimize False Negatives (FN), ensuring that
life-threatening events (e.g., ventricular ectopic beats 'V') are reliably detected.
● Specificity: Defined as the true negative rate. High specificity is crucial for
deployment to minimize False Positives (FP), which could otherwise lead to alarm
fatigue in clinical settings or unnecessary intervention.
● Area Under the Curve (AUC): A robust measure of classifier performance across
various threshold settings, crucial for handling inherent data imbalance.
Real-Time and Edge Metrics
These quantitative metrics validate the feasibility of deployment on wearable and
embedded devices:
● Inference Time (Latency): Measured in milliseconds per segment. This metric
directly addresses the need for low-latency analysis required for real-time
monitoring.
● Memory Footprint: Measured as the final size of the optimized model (MB/KB).
Minimizing this footprint is essential for resource-constrained wearable devices.
5.3 Comparative Methodology and Baseline Analysis.....................................................
The comparison methodology requires standardized training and testing protocols to
ensure a fair evaluation of all five models: LA-NN, CNN, LSTM, Standard LNN, and
Transformer.^
The strategic comparison against the four baselines is intended to differentiate the LA-
NN based on three axes: Classification Accuracy, Computational Cost (Efficiency),
and Adaptivity. By comparing LA-NN against the Standard LNN, the unique
performance gain derived from incorporating the attention mechanism can be
quantified. The comparison against the Transformer will confirm the computational
efficiency gained by replacing large self-attention blocks with the compact, continuous-
time LNN core. The fundamental hypothesis is that the LA-NN will demonstrate the
optimal three-way trade-off, achieving high accuracy with superior adaptivity and a
minimized computational footprint, making it the ideal architecture for Edge AI in
biomedical signal processing.
Table 3: Comprehensive Performance Evaluation Matrix

Metric Category Metrics LA-NN Target Goal Comparison
Baseline
Classification
Accuracy
Accuracy, F1-Score,
AUC
Maximize CNN, LSTM,
Standard LNN,
Transformer^
Clinical Reliability Sensitivity,
Specificity,
Precision, Recall
Maximize Sensitivity
for critical classes
('V', 'F')
CNN, LSTM^
Edge Constraints Inference Time
(Latency)
Minimize (< 10
ms/segment)
Standard LNN,
Transformer^
Deployment Cost Memory Footprint
(Model Size)
Minimize (< 1 MB
post-quantization)
All Baselines^
6: Clinical Relevance and Decision Logic Layer
6.1 Operational Mapping: From Prediction Scores to Triage Rules
The integration of "Explicit Clinical / Decision Logic" and moving "beyond classifying
ECG signals”, a secondary, rule-based decision layer will be implemented atop the
LA-NN’s output probabilities. This layer maps the model’s predictions to concrete
operational actions and triage rules, essential for clinical integration.
The system will employ a tiered alert system, transforming the 5-class probability
vector (P) into actionable outcomes:
● Level 1 (Normal): When confidence in the 'N' class exceeds a high threshold
(θNormal), indicating safe status.
● Level 2 (Monitor/Mild Anomaly): Triggered by low-to-moderate confidence
scores for non-critical classes ('S', 'Q'). This action prompts additional data logging
or flags the data for periodic human review, minimizing false alarms while
maintaining vigilance.
● Level 3 (Alarm/Critical Anomaly): Triggered when critical classes ('V', 'F') meet
a stringent confidence threshold (θcritical). This immediately triggers an operational
alarm, reflecting the high clinical risk associated with these ventricular and fusion
events.
6.2 Mathematical Justification for Confidence Handling
The selection of optimal thresholds, particularly (θcritical), cannot be arbitrary. The
thresholds will be mathematically justified by minimizing a weighted clinical cost
function. Since a False Negative (a missed critical arrhythmia) carries a significantly
higher cost (e.g., patient fatality) than a False Positive (a false alarm), the cost function
will heavily penalize FN errors. This approach ensures the model optimizes
operational safety and high sensitivity for life-threatening classes.^
Furthermore, the system will utilize Confidence Interval Analysis to manage
uncertainty. By flagging predictions that fall within specified low-confidence intervals,
the system promotes transparent medical decision-making, ensuring that ambiguous
cases are automatically routed to human operators for validation rather than resulting
in potentially dangerous automated actions.^
6.3 Interpretability Analysis of Model Dynamics
Transparent medical decision-making requires that the system’s output be auditable.
The LA-NN’s hybrid design facilitates two unique forms of interpretability analysis :
1. LNN Dynamic Visualization: The continuous-time LNN core allows for the
tracking and visualization of the instantaneous time constant (τ) for individual
liquid neurons. By plotting τ across an ECG segment, it is possible to demonstrate
when the model's memory horizon dynamically shortens or lengthens in direct
response to morphological changes (e.g., observing a rapid reduction in τ when
processing the swift onset of a ventricular ectopic beat), providing justification for
the classification.
2. Attention Weight Visualization: The attention scores generated by the Multi-
Head Attention mechanism will be mapped back and overlaid onto the raw ECG
signal. This visualization will highlight precisely which time steps within the 180-
point segment contributed most significantly to the final classification decision,
providing an auditable rationale for why a specific diagnosis was reached.
7: Conclusion, Future, and Formal Reviewer Remarks
7.1 Summary of Mid-Semester Achievements.................................................................
This mid-semester report confirms that the "Real-Time ECG Signal Analysis Using
Liquid Neural Networks" project has successfully established a robust theoretical
foundation and achieved the required 50% implementation milestone. The novel
Liquid-Attention Neural Network (LA-NN) architecture has been designed and
implemented from scratch, strategically combining the adaptive dynamics of LNNs
with the contextual power of Attention to overcome the limitations of static models in
handling non-stationary ECG signals. Furthermore, the project has successfully
integrated critical components specifically, the explicit statement of novelty and the
architectural framework for clinical decision logic to directly address and remediate all
major faculty feedback points.^
7.2 Proposed Plan for Remaining Work
The remaining work is strictly aligned with the formalized Plan of Work:
● Design & Development (Completed): Architecture design and custom pipeline
development were completed prior to this review.
● Testing (30th Nov–2nd Dec 2025): This critical phase will involve:
○ Full implementation and standardized training of all four baseline models
(CNN, LSTM, Standard LNN, Transformer).
○ Execution of the comprehensive performance evaluation matrix (Table 3),
acquiring quantitative metrics for latency, memory footprint, and clinical
reliability (Sensitivity/Specificity).
○ Rigorous adaptability testing against noise and patient variations.
● Finalization: Implementation of model quantization and compression for edge
optimization. Final implementation and testing of the clinically justified decision
logic layer (Chapter 6). Completion of the LNN dynamic and attention weight
interpretability analysis.
● Dissertation Review and Submission (Jan–Feb 2026): Compilation of final
report drafts and submission.
Literature References...........................................................................................................
[1] Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid Time-constant
Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 35(9), 7657 7666.

[2] Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of
a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220.

[3] Acharya, U. R., Oh, S. L., Hagiwara, Y., Tan, J. H., & Adam, M. (2017). A deep
convolutional neural network model to classify heartbeats. Computers in Biology and
Medicine, 89, 389–396.

[4] Yildirim, Ö. (2018). A novel wavelet sequence based on deep bidirectional LSTM network
model for ECG signal classification. Computers in Biology and Medicine, 96, 189–202.

[5] Lechner, M., Hasani, R., Amini, A., Henzinger, T. A., Rus, D., & Grosu, R. (2020). Neural
Circuit Policies Enabling Auditable Autonomy. Nature Machine Intelligence, 2(10), 642–652.