# 🧪 Data Preprocessing Pipeline Walkthrough

**File Path:** `src/data/preprocessing.py`

This module is responsible for the "ETL" (Extract, Transform, Load) process. it converts raw medical signal files (WFDB format) into normalized tensors for the LA-NN model.

---

## 🗺️ 1. AAMI Mapping (Lines 12–19)
The MIT-BIH dataset uses dozens of specific diagnostic codes (e.g., 'L' for Left Bundle Branch Block). To make the model useful for general screening, we map these into the **5 AAMI Standard Classes**:
*   **0 (N)**: Normal and variants.
*   **1 (S)**: Supraventricular ectopic beats.
*   **2 (V)**: Ventricular ectopic beats (High Risk).
*   **3 (F)**: Fusion beats.
*   **4 (Q)**: Paced/Unknown.

---

## 🛠️ 2. Core Utilities

### `normalize_signal` (Lines 37–43)
Applies **Z-score Normalization** ($x' = \frac{x - \mu}{\sigma}$).
*   It ensures the ECG signal has a mean of 0 and a standard deviation of 1.
*   **Proofing**: Line 41 checks if `std == 0` to prevent division-by-zero errors in flat signals.

### `process_record` (Lines 60–95)
The main "Slicing" logic for a single patient record.
1.  **Read (Line 65-66)**: Loads the MLII lead data using `wfdb`.
2.  **Locate (Line 69-71)**: Finds the annotated R-peak locations (where the heartbeat is strongest).
3.  **Boundary Check (Line 83)**: Ensures we don't try to cut a window that goes off the edge of the recording.
4.  **Segment (Line 87)**: Cuts a 180-sample window centered on the R-peak (90 samples before, 90 samples after).
5.  **Clean (Line 90)**: Normalizes the individual heartbeat segment.

---

## 🏗️ 3. The Orchestrator (`load_data`)

This function (Lines 97–164) manages the entire project dataset.

### Execution Flow:
1.  **Download (Lines 107–110)**: If the `mitdb` folder is missing, it automatically pulls the data from the PhysioNet servers.
2.  **Batch Processing (Lines 122–126)**: Iterates through all 48 patient records.
    *   It uses `tqdm` (Line 122) to show a progress bar to the user.
    *   It uses a `try/except` block (Line 123) so that one corrupted file doesn't stop the whole pipeline.
3.  **Reshaping (Line 137)**: `X = X[..., np.newaxis]`
    *   Adds the "Channel" dimension. The model expects `(Batch, Seq_Len, 1)`, not just `(Batch, Seq_Len)`.
4.  **Stratified Splitting (Lines 144–151)**:
    *   Divides the data into **Train (70%)**, **Validation (15%)**, and **Test (15%)**.
    *   **Crucial Step**: It uses `stratify=y`, which ensures the percentage of rare arrhythmias (like 'F' beats) is the same in all three sets.
5.  **Dataloader Creation (Lines 156–162)**: 
    *   Wraps the raw arrays into PyTorch `DataLoader` objects.
    *   Enables `shuffle=True` for training to ensure the model doesn't learn the order of patients.

---

## 📦 4. `ECGDataset` Class (Lines 21–35)
A standard PyTorch wrapper that converts NumPy arrays into Tensors and makes them accessible by index during the training loop.

---
**Summary Flow**: 
Download → Extract R-peaks → Segment 180ms Windows → Z-score Normalize → Stratified Split → Batch DataLoaders.
