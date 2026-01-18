# Project Evaluation Summary: ProcessTransformer
**Repository:** https://github.com/Zaharah/processtransformer
**Domain:** Predictive Business Process Monitoring (PBPM) / Deep Learning
**Architecture:** Transformer-based (Self-Attention) for sequential event prediction.

## 1. Core Capability Analysis
The project implements a "Next State Oracle" using a Transformer architecture tailored for event logs. It replaces traditional LSTM/RNN methods to capture long-range dependencies in business processes.

* **Primary Tasks:** 1.  `NEXT_ACTIVITY`: Predicts the next discrete event (classification).
    2.  `NEXT_TIME`: Predicts timestamp of next event (regression).
    3.  `REMAINING_TIME`: Predicts time until case completion (regression).
* **Relevance to Simulation:** High. The `NEXT_ACTIVITY` model outputs a Softmax distribution, enabling stochastic sampling for Monte Carlo simulations rather than deterministic execution.

## 2. Data Ingestion & Schema (Critical for BPIC17)
The system expects a flattened, sequential event log. BPIC17 is multi-dimensional (Application, Offer, Workflow attributes); bridging this gap is the primary integration challenge.

* **Input Format:** CSV (Comma Separated).
* **Required Columns (`data_processing.py`):**
    * `Case ID`: Unique case identifier.
    * `Activity`: Categorical label (mapped to `int` via `LogsDataProcessor`).
    * `Complete Timestamp`: `datetime` format.
* **Preprocessing Logic:**
    * Converts timestamps to relative time/duration features.
    * Tokenizes `Activity` column into an integer vocabulary.
    * Generates prefixes: A case of length $L$ yields $L$ training examples (Prefix $P_t \rightarrow$ Target $E_{t+1}$).

## 3. Codebase Structure & Integration Points

### A. Data Pipeline (`processtransformer/data/processor.py`)
* **Class:** `LogsDataProcessor`
* **Action:** Reads CSV, encodes columns, splits into Train/Test.
* **Agent Task:** Modify this file to map BPIC17's specific columns (e.g., `concept:name` + `lifecycle:transition`) into the simplistic `Activity` column expected by the model.

### B. Model Definition (`processtransformer/models/transformer.py`)
* **Architecture:** Keras Functional API.
* **Key Layer:** Multi-head self-attention mechanisms tailored for sequence inputs.
* **Instantiation:**
    ```python
    model = transformer.get_next_activity_model(
        max_case_length=N, 
        vocab_size=V, 
        output_dim=V
    )
    ```

### C. Inference Logic (`next_activity.py`)
* **Current State:** Uses `np.argmax` for deterministic accuracy evaluation.
    ```python
    # Line 98 in next_activity.py
    y_pred = np.argmax(transformer_model.predict(test_token_x), axis=1)
    ```
* **Required Change for Simulation:** The Agent must override this to access raw logits/probabilities for weighted random sampling.
    ```python
    # Target Logic for Simulation Engine
    probs = transformer_model.predict(current_prefix_tokenized)
    next_activity = np.random.choice(vocab, p=probs[0])
    ```

## 4. Implementation Constraints & Requirements
* **Framework:** TensorFlow >= 2.4.
* **State Management:** The model is stateless; the "state" is the full history (prefix) of the current simulated trace. The simulation engine must maintain the growing trace buffer for every active case.
* **Vocabulary Persistence:** The mapping `x_word_dict` (Activity String $\to$ Int) created in `loader.py` must be serialized. The simulation engine must use this exact mapping to encode simulated steps before feeding them back into the model for the next step prediction.

## 5. Decision Recommendation
**FIT: 8/10**
* **Pros:** Modern architecture, supports variable length history, naturally probabilistic output suitable for simulation.
* **Cons:** Data loader is rigid (requires schema mapping for BPIC17), no native support for multi-attribute state (e.g., data payloads in BPIC17 like 'Amount' or 'Resource' are ignored unless concatenated into the Activity name).
