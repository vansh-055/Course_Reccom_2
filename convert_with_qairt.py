import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import pickle
import subprocess
import sys

# --- Model Definitions (Required for loading the model) ---
class FactorizationMachine(nn.Module):
    def __init__(self, num_users, num_courses, embedding_dim):
        super(FactorizationMachine, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.course_embedding = nn.Embedding(num_courses, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user, course):
        user_emb = self.user_embedding(user.long())
        course_emb = self.course_embedding(course.long())
        interaction = torch.cat([user_emb, course_emb], dim=1)
        output = self.linear(interaction)
        return output

class ConversionReadyFactorizationMachine(nn.Module):
    def __init__(self, num_users, num_courses, embedding_dim):
        super(ConversionReadyFactorizationMachine, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.course_embedding = nn.Embedding(num_courses, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_id, course_id):
        user_emb = self.user_embedding(user_id.long())
        course_emb = self.course_embedding(course_id.long())
        combined = torch.cat([user_emb, course_emb], dim=1)
        output = self.linear(combined)
        return output

def run_command(setup_script, python_exe, tool_script, args):
    """
    Helper function to run an SDK tool by first sourcing the envsetup.ps1 script
    in a PowerShell session.
    """
    full_command_str = f". '{setup_script}'; & '{python_exe}' '{tool_script}' {' '.join(args)}"
    
    print(f"🚀 Executing PowerShell command...")
    print(f"   {full_command_str}")
    
    try:
        result = subprocess.run(
            ["powershell.exe", "-Command", full_command_str],
            check=True, text=True, capture_output=True
        )
        print("✔️ Command successful.")
        if result.stdout:
            print(f"--- STDOUT ---\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Command failed with exit code {e.returncode}")
        print(f"--- STDOUT ---\n{e.stdout}")
        print(f"--- STDERR ---\n{e.stderr}")
        sys.exit(1)

def main():
    print("🚀 Starting modern QAIRT conversion and quantization process...")

    # --- Configuration ---
    model_dir = Path("models2").resolve()
    output_dir = Path("qairt_output").resolve()
    data_dir = output_dir / "quantization_data"
    
    # --- SNPE/QAIRT SDK Configuration ---
    sdk_root = Path(r"C:\QualcommSNPE\qairt\2.36.0.250627")
    
    sdk_python_exe = sdk_root / "bin" / "venv" / "Scripts" / "python.exe"
    env_setup_script = sdk_root / "bin" / "envsetup.ps1"
    
    qairt_converter_script = sdk_root / "bin" / "x86_64-windows-msvc" / "qairt-converter"
    qairt_quantizer_script = sdk_root / "bin" / "x86_64-windows-msvc" / "qairt-quantizer"

    # --- Create Directories ---
    output_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    print(f"✔️ Output directory '{output_dir}' is ready.")

    # --- Step 1: Export PyTorch model to ONNX ---
    print("\n--- [1/4] Loading model and exporting to ONNX ---")
    onnx_path = output_dir / 'model.onnx'
    
    with open(model_dir / 'student_encoder.pkl', 'rb') as f: student_encoder = pickle.load(f)
    with open(model_dir / 'course_encoder.pkl', 'rb') as f: course_encoder = pickle.load(f)
    quantized_model_torch = torch.load(model_dir / 'quantized_model_full.pth', map_location='cpu', weights_only=False)

    num_users, num_courses = len(student_encoder.classes_), len(course_encoder.classes_)
    embedding_dim = quantized_model_torch.user_embedding.weight.shape[1]
    
    clean_model = ConversionReadyFactorizationMachine(num_users, num_courses, embedding_dim)
    clean_model.load_state_dict(quantized_model_torch.state_dict(), strict=False)
    clean_model.eval()

    torch.onnx.export(
        clean_model, (torch.tensor([0]), torch.tensor([0])), str(onnx_path),
        input_names=['user_id', 'course_id'], output_names=['prediction'],
        opset_version=11, dynamic_axes={'user_id': {0: 'batch_size'}, 'course_id': {0: 'batch_size'}}
    )
    print(f"✔️ Successfully created ONNX model: {onnx_path}")
    
    # --- Step 2: Convert ONNX to Float DLC ---
    print("\n--- [2/4] Converting ONNX to Float DLC ---")
    float_dlc_path = output_dir / "model_float.dlc"

    # FIX: Added --input_dim arguments for both dynamic inputs.
    converter_args = [
        "--input_network", f"'{onnx_path}'",
        "--output_path", f"'{float_dlc_path}'",
        "--input_dim", "user_id", "'1,1'",
        "--input_dim", "course_id", "'1,1'"
    ]
    run_command(env_setup_script, sdk_python_exe, qairt_converter_script, converter_args)
    
    # --- Step 3: Create representative data files for quantization ---
    print("\n--- [3/4] Creating quantization input files ---")
    input_list_path = output_dir / "input_list.txt"
    user_raw_path = data_dir / "user_id_0.raw"
    course_raw_path = data_dir / "course_id_0.raw"
    
    dummy_input = np.array([0], dtype=np.float32)
    dummy_input.tofile(user_raw_path)
    dummy_input.tofile(course_raw_path)

    content = f"user_id:='{user_raw_path.resolve()}'\ncourse_id:='{course_raw_path.resolve()}'"
    with open(input_list_path, 'wb') as f:
        f.write(content.encode('ascii'))
    print(f"✔️ Successfully created input_list.txt for quantization.")

    # --- Step 4: Quantize the DLC ---
    print("\n--- [4/4] Quantizing DLC to INT8 ---")
    quantized_dlc_path = output_dir / "model_quantized.dlc"
    quantizer_args = [
        "--input_dlc", f"'{float_dlc_path}'",
        "--input_list", f"'{input_list_path}'",
        "--output_dlc", f"'{quantized_dlc_path}'"
    ]
    run_command(env_setup_script, sdk_python_exe, qairt_quantizer_script, quantizer_args)
    
    print("\n🎉 --- Process Complete! --- 🎉")
    print(f"Final quantized model is ready at: {quantized_dlc_path}")


if __name__ == "__main__":
    try:
        from sklearn.exceptions import InconsistentVersionWarning
        import warnings
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except ImportError:
        pass
    main()