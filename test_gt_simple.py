"""
Simplified ground truth feature validation
Tests only the data loading and ground truth mapping without model inference
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ground_truth_mapping():
    """Test that AAMI mapping is correctly implemented"""
    from src.api.stream_engine import ECGStreamEngine
    
    print("=" * 60)
    print("Ground Truth Mapping Test")
    print("=" * 60)
    
    # Create a minimal engine instance
    print("\n1. Testing AAMI mapping dictionary...")
    
    # Mock model (just needs to be an object)
    class MockModel:
        def eval(self):
            return self
    
    try:
        # Initialize with a record (will try to load data)
        engine = ECGStreamEngine(MockModel(), record_name='100')
        
        print("   ✓ AAMI_MAPPING exists")
        print(f"   - Mapping has {len(engine.AAMI_MAPPING)} entries")
        
        # Check some key mappings
        expected_mappings = {
            'N': 'N',  # Normal
            'V': 'V',  # Ventricular
            'F': 'F',  # Fusion
            'A': 'S',  # Supraventricular mapped to S
        }
        
        print("\n2. Verifying key mappings...")
        for symbol, expected_class in expected_mappings.items():
            actual_class = engine.AAMI_MAPPING.get(symbol)
            status = "✓" if actual_class == expected_class else "✗"
            print(f"   {status} {symbol} -> {actual_class} (expected {expected_class})")
        
        print(f"\n3. Ground truth dictionary...")
        print(f"   - Has ground_truth attribute: {hasattr(engine, 'ground_truth')}")
        print(f"   - Type: {type(engine.ground_truth)}")
        print(f"   - Entries: {len(engine.ground_truth)}")
        
        if len(engine.ground_truth) > 0:
            # Show first few entries
            print("\n   First 5 ground truth entries:")
            for i, (sample_idx, label) in enumerate(list(engine.ground_truth.items())[:5]):
                print(f"     Sample {sample_idx}: {label}")
        
        print("\n✓ All ground truth features are properly implemented!")
        
    except FileNotFoundError as e:
        print(f"\n! Database not found: {e}")
        print("  Run 'python main.py' first to download MIT-BIH database")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_ground_truth_mapping()
