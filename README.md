# PyTorch to SNPE DLC Conversion Guide

This document provides a complete, step-by-step guide for training a course recommendation model with PyTorch and converting it into a quantized SNPE DLC file. This workflow has been redesigned to be more robust and handle common environment issues on Windows.

## Workflow Overview

The process is divided into three clear stages:

1. **Model Training**: Train the PyTorch model to create the models directory
2. **Generate Conversion Scripts**: Run a Python script that prepares all necessary files, including a dedicated PowerShell script to perform the conversion  
3. **Execute Conversion**: Run the single, auto-generated PowerShell script to get your final .dlc file

---

## Step 1: Train the PyTorch Model

This script trains the recommendation model using `data.json`. Its output is the models directory containing `quantized_model_full.pth` and the encoder files.

**Save this code as `train_model.py`:**

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

    def save_for_production(self, save_dir='models'):
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

---

## Step 2: Generate Conversion Scripts and Files

This script is the core of the new workflow. It generates the `.onnx` model, the correctly formatted `input_list.txt`, and, most importantly, a `run_conversion.ps1` PowerShell script that you will execute in the next step.

**Save this code as `create_conversion_files.py`:**

```python
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import pickle

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

def generate_files():
    print("Generating SNPE conversion files and execution script...")

    # --- Configuration ---
    model_dir = Path("models").resolve()
    output_dir = Path("snpe_conversion_output").resolve()
    data_dir = output_dir / "quantization_data"
    input_list_path = output_dir / "input_list.txt"
    user_raw_path = data_dir / "user_id_0.raw"
    course_raw_path = data_dir / "course_id_0.raw"
    onnx_path = output_dir / 'model.onnx'
    
    # SNPE SDK Configuration
    snpe_sdk_root = Path(r"C:\Users\vikas\OneDrive\Desktop\Bits teaching material\PS1-Swecha\Aneesh Sir\Qualcomm\v2.36.0.250627\qairt\2.36.0.250627")
    snpe_python_exe = snpe_sdk_root / "bin" / "venv" / "Scripts" / "python.exe"
    snpe_dlc_converter_script = snpe_sdk_root / "bin" / "x86_64-windows-msvc" / "snpe-onnx-to-dlc"
    snpe_quantizer_exe = snpe_sdk_root / "bin" / "x86_64-windows-msvc" / "snpe-dlc-quant"
    snpe_setup_script = snpe_sdk_root / "bin" / "envsetup.ps1"

    # --- Create Directories ---
    output_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    # --- Load Model and Export to ONNX ---
    with open(model_dir / 'student_encoder.pkl', 'rb') as f: 
        student_encoder = pickle.load(f)
    with open(model_dir / 'course_encoder.pkl', 'rb') as f: 
        course_encoder = pickle.load(f)
    
    num_users, num_courses = len(student_encoder.classes_), len(course_encoder.classes_)
    
    quantized_model = torch.load(model_dir / 'quantized_model_full.pth', map_location='cpu', weights_only=False)
    quantized_model.eval()
    
    embedding_dim = quantized_model.user_embedding.weight.shape[1]
    clean_model = ConversionReadyFactorizationMachine(num_users, num_courses, embedding_dim)
    
    with torch.no_grad():
        clean_model.load_state_dict(quantized_model.state_dict(), strict=False)
    clean_model.eval()
    
    torch.onnx.export(
        clean_model, (torch.tensor([0]), torch.tensor([0])), str(onnx_path),
        input_names=['user_id', 'course_id'], output_names=['prediction'],
        opset_version=11, 
        dynamic_axes={
            'user_id': {0: 'batch_size'}, 
            'course_id': {0: 'batch_size'}, 
            'prediction': {0: 'batch_size'}
        }
    )
    print(f"Successfully created: {onnx_path}")

    # --- Create Quantization Files with Forced Linux-style Line Endings ---
    dummy_input = np.array([0], dtype=np.float32)
    dummy_input.tofile(user_raw_path)
    dummy_input.tofile(course_raw_path)
    content = f"user_id:={user_raw_path.resolve()}\ncourse_id:={course_raw_path.resolve()}"
    with open(input_list_path, 'wb') as f:
        f.write(content.encode('ascii'))
    print(f"Successfully created: {input_list_path}")

    # --- Generate the PowerShell Execution Script ---
    ps_script_path = Path("run_conversion.ps1").resolve()
    float_dlc_path = output_dir / "model_float.dlc"
    quantized_dlc_path = output_dir / "model_quantized.dlc"

    ps_script_content = f'''
Write-Host "--- Starting SNPE Conversion ---" -ForegroundColor Green

# Step 1: Set up the SNPE environment
Write-Host "Setting up SNPE environment..."
& "{snpe_setup_script}"

# Step 2: Run ONNX to DLC conversion
Write-Host "`nConverting ONNX to float DLC..."
& "{snpe_python_exe}" "{snpe_dlc_converter_script}" -d user_id 1 -d course_id 1 -i "{onnx_path}" -o "{float_dlc_path}"
if ($LASTEXITCODE -ne 0) {{ Write-Host "ERROR: ONNX to DLC conversion failed." -ForegroundColor Red; exit 1 }}

# Step 3: Run DLC Quantization
Write-Host "`nQuantizing DLC..."
& "{snpe_quantizer_exe}" --input_dlc "{float_dlc_path}" --input_list "{input_list_path}" --output_dlc "{quantized_dlc_path}"
if ($LASTEXITCODE -ne 0) {{ Write-Host "ERROR: DLC quantization failed." -ForegroundColor Red; exit 1 }}

Write-Host "`n--- CONVERSION COMPLETE ---" -ForegroundColor Green
Write-Host "Final model is ready at: {quantized_dlc_path}"
'''
    with open(ps_script_path, 'w') as f:
        f.write(ps_script_content)
    
    print(f"\nSuccessfully created PowerShell execution script: {ps_script_path}")
    print("You can now proceed to Step 3.")

if __name__ == "__main__":
    from sklearn.exceptions import InconsistentVersionWarning
    import warnings
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    generate_files()
```

---

## Step 3: Execute the Conversion

This is the final, simplified step. You will run the PowerShell script that was automatically generated by the Python script in Step 2.

### In a PowerShell terminal, run these commands:

1. **Navigate to your project directory:**
   ```powershell
   cd C:\dev\recommendation-backend
   ```

2. **(Optional) Set Execution Policy:** If you haven't run PowerShell scripts before, you may need to run this command once per session to allow it:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

3. **Run the auto-generated conversion script:**
   ```powershell
   .\run_conversion.ps1
   ```

The script will now handle the environment setup and execute the conversion and quantization steps in sequence. After it finishes, your `model_quantized.dlc` file will be ready in the `snpe_conversion_output` folder.

---

## Requirements

### Python Dependencies
- pandas
- torch
- scikit-learn
- numpy
- pathlib

### System Requirements
- Windows OS
- SNPE SDK installed and configured
- PowerShell execution policy allowing script execution

### File Structure
```
project/
├── data.json
├── train_model.py
├── create_conversion_files.py
├── run_conversion.ps1 (auto-generated)
├── models/
│   ├── quantized_model_full.pth
│   ├── student_encoder.pkl
│   └── course_encoder.pkl
└── snpe_conversion_output/
    ├── model.onnx
    ├── input_list.txt
    ├── model_float.dlc
    ├── model_quantized.dlc
    └── quantization_data/
        ├── user_id_0.raw
        └── course_id_0.raw
```

---

## Usage

1. Run the training script: `python train_model.py`
2. Generate conversion files: `python create_conversion_files.py`
3. Execute the conversion: `.\run_conversion.ps1`

Your final quantized SNPE model will be available as `model_quantized.dlc` in the `snpe_conversion_output` directory.
