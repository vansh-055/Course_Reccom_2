# PyTorch to Quantized DLC Conversion Workflow

This document provides a complete, step-by-step guide for training a course recommendation model with PyTorch and converting it into a quantized SNPE DLC file using the modern QAIRT toolchain. The workflow is automated into two main Python scripts.

## Prerequisites

- **Python**: A working Python environment (e.g., Python 3.10+)
- **Qualcomm Neural Processing SDK (SNPE/QAIRT)**: The SDK must be installed on your system. This guide assumes the path is `C:\QualcommSNPE\qairt\2.36.0.250627`
- **PyTorch and other ML libraries**: Required for model training

## Setup Instructions

Follow these steps to set up the environment before running the scripts.

### 1. Set up the SDK Virtual Environment

First, create a dedicated Python virtual environment for the Qualcomm SDK tools. This ensures that the tools run with the correct dependencies.

Open a PowerShell or Command Prompt terminal and run:

```powershell
# Navigate to the SDK's bin directory
cd C:\QualcommSNPE\qairt\2.36.0.250627\bin

# Create the virtual environment
python -m venv .\venv
```

### 2. Install Required Packages

Next, install all necessary Python packages into both your main Python environment (for training) and the SDK's virtual environment (for conversion).

**A. Install packages for your main environment (for training):**

```bash
pip install torch pandas scikit-learn
```

**B. Install packages for the SDK's venv (for conversion):**

Run this single command to install all packages required by the `qairt-converter` and `qairt-quantizer` tools.

```bash
C:\QualcommSNPE\qairt\2.36.0.250627\bin\venv\Scripts\pip.exe install numpy pandas scikit-learn onnx PyYAML packaging
```

## Workflow Execution

After setup, the entire process is just two steps. Run these commands from your project's root directory (e.g., `C:\dev\recommendation-backend`).

### Step 1: Train the PyTorch Model

This script trains the recommendation model using `data.json` and saves the trained model, along with encoder files, into a `models2` directory.

Save this code as `train_model.py`:

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import json
from torch.quantization import quantize_dynamic

class FactorizationMachine(nn.Module):
    def __init__(self, num_users, num_courses, embedding_dim):
        super(FactorizationMachine, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.course_embedding = nn.Embedding(num_courses, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user, course):
        user_emb = self.user_embedding(user)
        course_emb = self.course_embedding(course)
        interaction = torch.cat([user_emb, course_emb], dim=1)
        output = self.linear(interaction)
        return output

class CourseRecommendationSystem:
    def __init__(self, data_path='data.json', embedding_dim=16, learning_rate=0.001):
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.quantized_model = None
        self.le_student = LabelEncoder()
        self.le_course = LabelEncoder()

    def load_and_preprocess_data(self):
        with open(self.data_path, 'r') as f:
            json_data = json.load(f)
        records = []
        for student_semester_key, courses_list in json_data.items():
            student_id = '-'.join(student_semester_key.split('-')[:-1])
            for course_item in courses_list:
                records.append({
                    'student_id': student_id,
                    'course_id': course_item['course_id'],
                    'course_grade': course_item['course_grade']
                })
        df = pd.DataFrame(records)
        df.dropna(subset=['student_id', 'course_id', 'course_grade'], inplace=True)
        df['student_id'] = self.le_student.fit_transform(df['student_id'])
        df['course_id'] = self.le_course.fit_transform(df['course_id'])
        df['course_grade'] = df['course_grade'] / 10.0
        return df

    def prepare_data_loaders(self, df, batch_size=256):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_user = torch.LongTensor(train_df['student_id'].values)
        train_course = torch.LongTensor(train_df['course_id'].values)
        train_grade = torch.FloatTensor(train_df['course_grade'].values)
        train_dataset = TensorDataset(train_user, train_course, train_grade)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    def train_model(self, train_loader, num_epochs=10):
        num_users = len(self.le_student.classes_)
        num_courses = len(self.le_course.classes_)
        self.model = FactorizationMachine(num_users, num_courses, self.embedding_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print("\n--- Starting Model Training ---")
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for user, course, grade in train_loader:
                optimizer.zero_grad()
                output = self.model(user, course).squeeze()
                loss = criterion(output, grade)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}')
        print("Training completed.")

    def apply_dynamic_quantization(self):
        self.model.eval()
        self.quantized_model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        print("Dynamic quantization applied.")

    def save_for_production(self, save_dir='models2'):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.quantized_model, os.path.join(save_dir, 'quantized_model_full.pth'))
        with open(os.path.join(save_dir, 'student_encoder.pkl'), 'wb') as f:
            pickle.dump(self.le_student, f)
        with open(os.path.join(save_dir, 'course_encoder.pkl'), 'wb') as f:
            pickle.dump(self.le_course, f)
        print(f"Production files saved to '{save_dir}/'")

def main():
    system = CourseRecommendationSystem()
    df = system.load_and_preprocess_data()
    train_loader = system.prepare_data_loaders(df)
    system.train_model(train_loader)
    system.apply_dynamic_quantization()
    system.save_for_production()

if __name__ == "__main__":
    main()
```

Save this sample data as `data.json`:

```json
{
    "student1-2023": [
        {"course_id": "CS101", "course_grade": 85},
        {"course_id": "MA202", "course_grade": 92}
    ],
    "student2-2023": [
        {"course_id": "CS101", "course_grade": 95},
        {"course_id": "PY301", "course_grade": 88}
    ],
    "student3-2024": [
        {"course_id": "MA202", "course_grade": 78},
        {"course_id": "PY301", "course_grade": 81}
    ],
    "student4-2024": [
        {"course_id": "CS101", "course_grade": 90},
        {"course_id": "EE401", "course_grade": 85}
    ]
}
```

Run the training script:

```bash
python train_model.py
```

### Step 2: Convert to Quantized DLC

This script automates the entire conversion process. It loads the trained PyTorch model, exports it to ONNX, converts it to a float DLC, and finally quantizes it to an 8-bit fixed-point DLC.

Save this code as `convert_with_qairt.py`:

```python
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
```

Run the conversion script:

```bash
python convert_with_qairt.py
```

## Output

After the process completes successfully, your final quantized model will be located at:

```
qairt_output/model_quantized.dlc
```

This `.dlc` file is ready to be deployed using the Qualcomm Neural Processing SDK runtime.

## Summary

This workflow provides a complete automation for converting PyTorch models to optimized DLC format:

1. **Training**: Creates a factorization machine model for course recommendations
2. **Quantization**: Applies dynamic quantization to reduce model size
3. **ONNX Export**: Converts the model to ONNX format for cross-platform compatibility
4. **DLC Conversion**: Uses QAIRT tools to create float and quantized DLC files
5. **Deployment Ready**: Final quantized model optimized for Qualcomm hardware

The entire process is streamlined into just two Python scripts that handle all the complex conversion steps automatically.
