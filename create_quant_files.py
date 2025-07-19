import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress the InconsistentVersionWarning from scikit-learn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- Model Definitions (Required for loading the model) ---
# These classes must match the definitions in your train_model.py
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

def generate_files():
    print("Generating SNPE conversion files and execution script...")

    # --- Configuration ---
    # **IMPORTANT**: Changed model_dir to 'models2'
    model_dir = Path("models2").resolve()
    output_dir = Path("snpe_conversion_output").resolve()
    data_dir = output_dir / "quantization_data"
    input_list_path = output_dir / "input_list.txt"
    user_raw_path = data_dir / "user_id_0.raw"
    course_raw_path = data_dir / "course_id_0.raw"
    onnx_path = output_dir / 'model.onnx'

    # SNPE SDK Configuration
    # This path is based on your last successful identification: C:\QualcommSNPE\qairt\2.36.0.250627
    snpe_sdk_root = Path(r"C:\QualcommSNPE\qairt\2.36.0.250627")
    snpe_python_exe = snpe_sdk_root / "bin" / "venv" / "Scripts" / "python.exe"
    snpe_dlc_converter_script = snpe_sdk_root / "bin" / "x86_64-windows-msvc" / "snpe-onnx-to-dlc"
    snpe_quantizer_exe = snpe_sdk_root / "bin" / "x86_64-windows-msvc" / "snpe-dlc-quant"
    snpe_setup_script = snpe_sdk_root / "bin" / "envsetup.ps1"

    # --- Create Directories ---
    output_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    # --- Load Model and Export to ONNX ---
    # Load encoders
    try:
        with open(model_dir / 'student_encoder.pkl', 'rb') as f:
            student_encoder = pickle.load(f)
        with open(model_dir / 'course_encoder.pkl', 'rb') as f:
            course_encoder = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Encoder files not found in {model_dir}. Please ensure 'train_model.py' ran successfully and saved files to 'models2'.")
        return

    num_users = len(student_encoder.classes_)
    num_courses = len(course_encoder.classes_)

    # Load quantized model
    try:
        quantized_model = torch.load(model_dir / 'quantized_model_full.pth', map_location='cpu', weights_only=False)
        quantized_model.eval()
    except FileNotFoundError:
        print(f"Error: Quantized model 'quantized_model_full.pth' not found in {model_dir}. Ensure 'train_model.py' ran successfully.")
        return
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        return

    embedding_dim = quantized_model.user_embedding.weight.shape[1]
    clean_model = ConversionReadyFactorizationMachine(num_users, num_courses, embedding_dim)

    # Load state dict, carefully handling potential quantization artifacts if direct load fails
    try:
        clean_model.load_state_dict(quantized_model.state_dict(), strict=False)
    except RuntimeError as e:
        print(f"Warning: Strict state_dict loading failed: {e}. Attempting to load non-strictly, this might indicate slight mismatches but could be acceptable for quantized models.")
        # Attempt to load only parameters that match
        model_dict = clean_model.state_dict()
        pretrained_dict = {k: v for k, v in quantized_model.state_dict().items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        clean_model.load_state_dict(model_dict)

    clean_model.eval()

    # Export to ONNX
    try:
        torch.onnx.export(
            clean_model, (torch.tensor([0]), torch.tensor([0])), str(onnx_path),
            input_names=['user_id', 'course_id'], output_names=['prediction'],
            opset_version=11, dynamic_axes={'user_id': {0: 'batch_size'}, 'course_id': {0: 'batch_size'}, 'prediction': {0: 'batch_size'}}
        )
        print(f"Successfully created: {onnx_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        return

    # --- Create Quantization Files with Explicit Windows Line Endings ---
    dummy_input_val = np.array([0], dtype=np.float32)
    dummy_input_val.tofile(user_raw_path)
    dummy_input_val.tofile(course_raw_path)
    print(f"Created: {user_raw_path}")
    print(f"Created: {course_raw_path}")

    # Use os.linesep to ensure correct line endings for Windows
    content = f"user_id:={user_raw_path.resolve()}{os.linesep}course_id:={course_raw_path.resolve()}"
    with open(input_list_path, 'w') as f:
        f.write(content)
    print(f"Successfully created: {input_list_path} with correct Windows line endings.")

    # --- Generate the Powershell Execution Script ---
    ps_script_path = Path("run_conversion.ps1").resolve()
    float_dlc_path = output_dir / "model_float.dlc"
    quantized_dlc_path = output_dir / "model_quantized.dlc"

    ps_script_content = f"""
Write-Host "--- Starting SNPE Conversion ---" -ForegroundColor Green

# Step 1: Set up the SNPE environment
Write-Host "Setting up SNPE environment..."
# Execute envsetup.ps1 using dot-sourcing and explicit architecture
. "{snpe_setup_script}" -arch X86_64

# Check if the last command (envsetup.ps1) resulted in an error
# Note: envsetup.ps1 can sometimes exit without setting $LASTEXITCODE, but PowerShell might still capture errors.
if ($LASTEXITCODE -ne 0) {{
    Write-Host "ERROR: envsetup.ps1 exited with code $LASTEXITCODE. This indicates an internal problem within the SNPE SDK setup script." -ForegroundColor Red;
    exit 1
}}
if ($Error.Count -gt 0) {{
    Write-Host "ERROR: PowerShell detected an error during envsetup.ps1 execution, even if \$LASTEXITCODE was 0." -ForegroundColor Red;
    Write-Host "Last PowerShell Error: $($Error[0].Exception.Message)" -ForegroundColor Red;
    # Clear the error to avoid it affecting subsequent commands unnecessarily
    $Error.Clear() # Clear errors from $Error collection for subsequent commands
    exit 1
}}

Write-Host "[INFO] QAIRT_SDK_ROOT: {snpe_sdk_root}"
Write-Host "[INFO] QAIRT SDK environment setup complete"
Write-Host "SNPE Environment check passed." -ForegroundColor Green


# Step 2: Run ONNX to DLC conversion
Write-Host "`nConverting ONNX to float DLC..."
# Note: User and course IDs are expected to be single values for this model, hence -d user_id 1 -d course_id 1
& "{snpe_python_exe}" "{snpe_dlc_converter_script}" -d user_id 1 -d course_id 1 -i "{onnx_path}" -o "{float_dlc_path}"
if ($LASTEXITCODE -ne 0) {{ Write-Host "ERROR: ONNX to DLC conversion failed." -ForegroundColor Red; exit 1 }}

# Step 3: Run DLC Quantization
Write-Host "`nQuantizing DLC..."
& "{snpe_quantizer_exe}" --input_dlc "{float_dlc_path}" --input_list "{input_list_path}" --output_dlc "{quantized_dlc_path}"
if ($LASTEXITCODE -ne 0) {{ Write-Host "ERROR: DLC quantization failed." -ForegroundColor Red; exit 1 }}

Write-Host "`n--- CONVERSION COMPLETE ---" -ForegroundColor Green
Write-Host "Final model is ready at: {quantized_dlc_path}"
"""
    with open(ps_script_path, 'w') as f:
        f.write(ps_script_content)

    print(f"\nSuccessfully created Powershell execution script: {ps_script_path}")
    print("You can now proceed to Step 3: Run '.\\run_conversion.ps1' in PowerShell.")

if __name__ == "__main__":
    generate_files()