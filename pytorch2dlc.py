import torch
import torch.nn as nn
import pickle
import numpy as np
import os
from pathlib import Path

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

def generate_conversion_files(model_dir='models', output_dir='snpe_conversion_output'):
    """
    Loads the trained model and generates the .onnx and input_list.txt files required for conversion.
    """
    print("🚀 Starting File Generation Process 🚀")
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n[Step 1/2] Loading model and exporting to ONNX...")
    with open(model_dir / 'student_encoder.pkl', 'rb') as f: student_encoder = pickle.load(f)
    with open(model_dir / 'course_encoder.pkl', 'rb') as f: course_encoder = pickle.load(f)
    
    num_users, num_courses = len(student_encoder.classes_), len(course_encoder.classes_)
    
    # Load the model, explicitly allowing it because we trust the source
    quantized_model = torch.load(model_dir / 'quantized_model_full.pth', map_location='cpu', weights_only=False)
    quantized_model.eval()
    
    embedding_dim = quantized_model.user_embedding.weight.shape[1]
    clean_model = ConversionReadyFactorizationMachine(num_users, num_courses, embedding_dim)
    
    with torch.no_grad():
        clean_model.load_state_dict(quantized_model.state_dict(), strict=False)
    clean_model.eval()
    
    onnx_path = output_dir / 'model.onnx'
    torch.onnx.export(
        clean_model, (torch.tensor([0]), torch.tensor([0])), str(onnx_path),
        input_names=['user_id', 'course_id'], output_names=['prediction'],
        opset_version=11, dynamic_axes={'user_id': {0: 'batch_size'}, 'course_id': {0: 'batch_size'}, 'prediction': {0: 'batch_size'}}
    )
    print(f"✅ ONNX model saved to: {onnx_path}")

    print("\n[Step 2/2] Creating input data for quantization...")
    input_data_dir = output_dir / 'quantization_data'
    input_data_dir.mkdir(exist_ok=True)
    sample_file = input_data_dir / "sample.raw"
    np.array([0], dtype=np.float32).tofile(sample_file)

    input_list_path = output_dir / "input_list.txt"
    with open(input_list_path, 'w') as f:
        f.write(f"user_id:={sample_file.resolve()}\n")
        f.write(f"course_id:={sample_file.resolve()}\n")
    print(f"✅ Quantization input list saved to: {input_list_path}")
    print("\n🎉🎉🎉 FILE GENERATION COMPLETE! 🎉🎉🎉")
    print("You can now run the 'run_conversion.ps1' script in Powershell.")


if __name__ == "__main__":
    from sklearn.exceptions import InconsistentVersionWarning
    import warnings
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    
    generate_conversion_files()