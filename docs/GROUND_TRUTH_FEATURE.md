# Ground Truth Feature Documentation

## Overview
This feature adds comprehensive ground truth comparison functionality to the LA-NN ECG monitoring system. It allows users to see actual annotated labels alongside model predictions in both the real-time UI and generated reports.

## Feature Components

### 1. Stream Engine Enhancement (`src/api/stream_engine.py`)

#### AAMI Mapping
Added a complete mapping from MIT-BIH annotation symbols to AAMI classes:
```python
AAMI_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # Normal
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',            # Supraventricular
    'V': 'V', 'E': 'V',                                # Ventricular
    'F': 'F',                                          # Fusion
    '/': 'Q', 'f': 'Q', 'Q': 'Q'                       # Unknown/Paced
}
```

#### Ground Truth Storage
- `ground_truth` dictionary: Maps sample indices to their annotated AAMI class labels
- Automatically populated when loading WFDB records with `.atr` annotation files
- Returns `None` for custom uploaded signals without annotations

#### Enhanced Inference Results
The `run_inference()` method now returns additional fields:
- `ground_truth`: The actual annotated label (or None if unavailable)
- `is_correct`: Boolean indicating if prediction matches ground truth (or None if unavailable)

### 2. UI Enhancements (`streamlit_app.py`)

#### New Metrics Display
Added a 4th metric card showing:
- **Ground Truth Label**: Current beat's actual label
- **Real-time Accuracy**: Running accuracy percentage
- **Color Coding**: 
  - Green for correct predictions
  - Red for mismatches
  - Gray when ground truth unavailable

#### Enhanced Diagnosis Log
Each log entry now includes:
- Checkmark (✓) for correct predictions
- Cross (✗) for incorrect predictions
- Ground truth label in brackets: `[GT: V]`
- Example: `[10:45:23] ✓ Pred: V [GT: V] (Conf: 94.2%)`

#### Anomaly Grid Updates
Critical event cards now display:
- Match indicator (✓/✗) in the title
- Ground truth badge in the header
- Color-coded headers:
  - Green for correct detections
  - Red for incorrect detections
  - Default for unknown ground truth

#### Accuracy Tracking
New session state variable `accuracy_stats`:
```python
{
    'correct': 0,  # Number of correct predictions
    'total': 0     # Total predictions with ground truth
}
```

### 3. Report Enhancements

#### Performance Summary Box
Added a prominent section at the top of HTML reports showing:
- **Overall Accuracy Percentage**: Large, centered metric
- **Correct vs Total Predictions**: Detailed breakdown
- **Total Anomalies Detected**: Count of all abnormal beats

#### Critical Point Analysis Grid
Each anomaly card in the report includes:
- **Ground Truth Badge**: Shows actual label if available
- **Color-Coded Headers**:
  - Green background for correct predictions
  - Red background for incorrect predictions
  - Standard red for cases without ground truth
- **Match Indicator**: Visual ✓ or ✗ symbol

#### Styling
Added professional gradient styling for the performance box:
```css
.performance-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
```

## Usage

### For WFDB Records
1. Select any standard MIT-BIH record (e.g., 100, 106, 200)
2. Ground truth will be automatically loaded from `.atr` files
3. UI will show real-time accuracy and ground truth labels
4. Generated report includes complete accuracy metrics

### For Custom Uploads
1. Upload CSV files without annotations
2. Ground truth will show as "N/A"
3. Accuracy tracking disabled
4. System focuses on pure prediction display

## Visual Indicators

### UI Color Scheme
- **Green (✓)**: Correct prediction matching ground truth
- **Red (✗)**: Incorrect prediction (mismatch)
- **Cyan**: Normal heartbeat
- **Yellow**: Monitor-level alert
- **Red trace**: Critical abnormality

### Report Color Scheme
- **Green header**: Correctly identified critical event
- **Red header**: Misclassified event
- **Purple gradient**: Performance summary box
- **Standard ECG red grid**: Clinical paper styling

## Technical Details

### Data Flow
1. **Load Phase**: WFDB annotations parsed into `ground_truth` dict
2. **Inference Phase**: Each R-peak inference retrieves corresponding ground truth
3. **Comparison Phase**: Prediction compared with ground truth label
4. **Display Phase**: Results color-coded and displayed with accuracy stats
5. **Report Phase**: Full session statistics compiled into HTML report

### Performance Impact
- Minimal overhead: Single dictionary lookup per inference
- No additional model computations
- Negligible memory footprint (~1KB per 1000 beats)

### Compatibility
- **Backward Compatible**: Works with all existing records
- **Graceful Degradation**: Shows "N/A" when ground truth unavailable
- **No Breaking Changes**: All existing functionality preserved

## Benefits

1. **Model Validation**: Real-time verification of prediction accuracy
2. **Clinical Trust**: Healthcare providers can see model performance
3. **Research Utility**: Ideal for demonstrations and thesis presentations
4. **Debugging**: Easy identification of systematic errors
5. **Reporting**: Comprehensive accuracy metrics in downloadable reports

## Future Enhancements

Potential additions:
- Confusion matrix in report
- Per-class accuracy breakdown
- Time-series accuracy trends
- Confidence calibration plots
- ROC curves for threshold tuning

## Example Output

### UI Display
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  Heart Rate     │  Prediction     │  Alert Status   │  Ground Truth   │
│     72 BPM      │   VENTRICULAR   │  LEVEL 3: CRIT  │       V         │
│                 │                 │                 │  Accuracy: 96.5% │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Report Section
```
╔══════════════════════════════════════════════════════════════╗
║          MODEL ACCURACY (Ground Truth Comparison)            ║
║                          96.5%                               ║
║        Correct: 193 / 200 predictions | Anomalies: 48       ║
╚══════════════════════════════════════════════════════════════╝
```

## Testing

To test this feature:
1. Run the application: `streamlit run streamlit_app.py`
2. Select a record with known arrhythmias (e.g., 106 or 200)
3. Start monitoring
4. Observe ground truth labels and accuracy in real-time
5. Stop monitoring and download the clinical report
6. Verify accuracy metrics and color-coded event cards

## Conclusion

This feature significantly enhances the clinical utility and research value of the LA-NN monitoring system by providing transparent, real-time comparison with expert-annotated ground truth labels.
