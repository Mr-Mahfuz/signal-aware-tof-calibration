import pickle
import m2cgen as m2c
import os

def create_output_dir():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def export_model_to_c():
    out_dir = create_output_dir()
    rf_path = os.path.join(out_dir, 'rf_model.pkl')
    
    if not os.path.exists(rf_path):
        print(f"Error: Could not find model at {rf_path}")
        print("Please run 02_train.py first.")
        return
        
    print(f"Loading Random Forest model from {rf_path}...")
    with open(rf_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Transpiling model to C code (this may take a minute for large forests)...")
    try:
        c_code = m2c.export_to_c(model)
        
        c_file_path = os.path.join(out_dir, 'model_inference.c')
        with open(c_file_path, 'w') as f:
            f.write(c_code)
            
        print(f"Success! Exported C code to {c_file_path}")
        print("\nYou can now copy the score() function from this file into your ESP32 project.")
        print("Function signature: double score(double * input)")
        print("Input array mapping:")
        print("input[0] = raw_distance_mm")
        print("input[1] = pan_calibrated_deg")
        print("input[2] = tilt_calibrated_deg")
        print("input[3] = signal_rate_kcps")
        print("input[4] = ambient_rate_kcps")
        print("input[5] = sigma_mm")
        print("\nPrediction = score(input)")
        print("Corrected Distance = raw_distance_mm + Prediction")
        
    except Exception as e:
        print(f"Error transpiling model: {e}")

if __name__ == "__main__":
    export_model_to_c()
