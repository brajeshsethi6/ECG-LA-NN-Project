"""
Quick test script to verify ground truth feature implementation
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.api.stream_engine import ECGStreamEngine
from src.models.la_nn import BioLANN
from src.config import Config

def test_ground_truth_feature():
    print("=" * 60)
    print("Ground Truth Feature Test")
    print("=" * 60)
    
    # 1. Initialize model
    print("\n1. Loading model...")
    model = BioLANN(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        ode_steps=Config.ODE_STEPS,
        num_heads=Config.NUM_ATTENTION_HEADS,
        dropout=Config.DROPOUT
    )
    
    model_path = os.path.join(Config.MODELS_DIR, 'la_nn_best.pth')
    if os.path.exists(model_path):
        import torch
        try:
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE), strict=False)
            print("   ✓ Model loaded successfully")
        except Exception as e:
            print(f"   ! Model load warning (using random weights): {str(e)[:50]}...")
    else:
        print("   ! Model file not found (using random weights)")
    
    model = model.to(Config.DEVICE).eval()
    
    # 2. Test with WFDB record (ground truth available)
    print("\n2. Testing with MIT-BIH record 100...")
    try:
        engine = ECGStreamEngine(model, record_name='100')
        print(f"   ✓ Engine initialized")
        print(f"   - Signal length: {len(engine.signal)} samples")
        print(f"   - R-peaks found: {len(engine.ann_samples)}")
        print(f"   - Ground truth entries: {len(engine.ground_truth)}")
        
        # Test inference on a few peaks
        test_count = 0
        correct_count = 0
        for sample_idx in list(engine.ann_samples)[:10]:
            result = engine.run_inference(sample_idx)
            if result:
                test_count += 1
                has_gt = result['ground_truth'] is not None
                is_correct = result['is_correct']
                
                print(f"\n   Beat #{test_count}:")
                print(f"     Prediction: {result['prediction']}")
                print(f"     Confidence: {result['confidence']:.2%}")
                print(f"     Ground Truth: {result['ground_truth'] or 'N/A'}")
                print(f"     Correct: {is_correct if has_gt else 'N/A'}")
                
                if is_correct:
                    correct_count += 1
        
        if test_count > 0:
            accuracy = (correct_count / test_count) * 100
            print(f"\n   Overall test accuracy: {accuracy:.1f}% ({correct_count}/{test_count})")
            
    except FileNotFoundError:
        print("   ! MIT-BIH database not found (run main.py first to download)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Test with custom signal (no ground truth)
    print("\n3. Testing with custom signal (no annotations)...")
    try:
        custom_signal = np.random.randn(3600).astype(np.float32)
        engine_custom = ECGStreamEngine(model, custom_signal=custom_signal)
        print(f"   ✓ Custom engine initialized")
        print(f"   - Signal length: {len(engine_custom.signal)} samples")
        print(f"   - Detected peaks: {len(engine_custom.ann_samples)}")
        print(f"   - Ground truth entries: {len(engine_custom.ground_truth)} (expected 0)")
        
        # Test one inference
        if len(engine_custom.ann_samples) > 0:
            sample_idx = list(engine_custom.ann_samples)[0]
            result = engine_custom.run_inference(sample_idx)
            if result:
                print(f"\n   Test inference:")
                print(f"     Prediction: {result['prediction']}")
                print(f"     Ground Truth: {result['ground_truth']} (expected None)")
                print(f"     Is Correct: {result['is_correct']} (expected None)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_ground_truth_feature()
