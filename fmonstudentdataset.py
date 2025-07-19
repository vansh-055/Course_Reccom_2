import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import csv
import os
import numpy as np
from torch.quantization import quantize_dynamic, QConfig, default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import torch.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import get_default_qconfig

class FactorizationMachine(nn.Module):
    def __init__(self, num_users, num_courses, embedding_dim):
        super(FactorizationMachine, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.course_embedding = nn.Embedding(num_courses, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 2, 1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, user, course):
        # Quantization stub
        user = self.quant(user.float())
        course = self.quant(course.float())
        
        user_emb = self.user_embedding(user.long())
        course_emb = self.course_embedding(course.long())
        
        # Concatenate user and course embeddings
        interaction = torch.cat([user_emb, course_emb], dim=1)
        output = self.linear(interaction)
        
        # Dequantization stub
        output = self.dequant(output)
        return output

class CourseRecommendationSystem:
    def __init__(self, data_path='data.json', embedding_dim=10, learning_rate=0.001):
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.quantized_model = None
        self.le_student = LabelEncoder()
        self.le_course = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data from JSON file"""
        # Load JSON data
        with open(self.data_path, 'r') as json_file:
            json_data = json.load(json_file)
        
        # Extract courses from JSON
        courses = []
        for sublist in json_data.values():
            if isinstance(sublist, list):
                for course in sublist:
                    if isinstance(course, dict):
                        courses.append(course)
        
        # Create DataFrame
        df = pd.DataFrame(courses)
        
        # Encode categorical variables
        df['student_id'] = self.le_student.fit_transform(df['student_id'])
        df['course_id'] = self.le_course.fit_transform(df['course_id'])
        
        # Normalize grades to [0, 1]
        df['course_grade'] = df['course_grade'] / 10.0
        
        return df
    
    def prepare_data_loaders(self, df, batch_size=64, test_size=0.2):
        """Prepare train and test data loaders"""
        # Train-test split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Convert to tensors
        train_user = torch.LongTensor(train_df['student_id'].values)
        train_course = torch.LongTensor(train_df['course_id'].values)
        train_grade = torch.FloatTensor(train_df['course_grade'].values)
        
        test_user = torch.LongTensor(test_df['student_id'].values)
        test_course = torch.LongTensor(test_df['course_id'].values)
        test_grade = torch.FloatTensor(test_df['course_grade'].values)
        
        # Create data loaders
        train_dataset = TensorDataset(train_user, train_course, train_grade)
        test_dataset = TensorDataset(test_user, test_course, test_grade)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, test_df
    
    def train_model(self, train_loader, num_epochs=10):
        """Train the Factorization Machine model"""
        num_users = len(self.le_student.classes_)
        num_courses = len(self.le_course.classes_)
        
        # Initialize model
        self.model = FactorizationMachine(num_users, num_courses, self.embedding_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("Training model...")
        for epoch in range(num_epochs):
            total_loss = 0.0
            self.model.train()
            
            for batch_idx, (user, course, grade) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(user, course).squeeze()
                loss = criterion(output, grade)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        print("Training completed!")
        return self.model
    
    def evaluate_model(self, test_loader):
        """Evaluate the model on test data"""
        self.model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for user, course, grade in test_loader:
                output = self.model(user, course).squeeze()
                loss = criterion(output, grade)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        print(f'Test MSE: {avg_loss:.4f}')
        return avg_loss
    
    def prepare_for_quantization(self):
        """Prepare model for quantization"""
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare for quantization
        self.model.qconfig = get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        
        return self.model
    
    def calibrate_model(self, calibration_loader):
        """Calibrate the model for quantization"""
        print("Calibrating model for quantization...")
        self.model.eval()
        
        with torch.no_grad():
            for i, (user, course, grade) in enumerate(calibration_loader):
                self.model(user, course)
                if i >= 100:  # Use limited samples for calibration
                    break
        
        print("Calibration completed!")
    
    def quantize_model(self, calibration_loader):
        """Apply quantization to the model"""
        print("Starting quantization process...")
        
        # Step 1: Prepare model for quantization
        self.prepare_for_quantization()
        
        # Step 2: Calibrate model
        self.calibrate_model(calibration_loader)
        
        # Step 3: Convert to quantized model
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        print("Quantization completed!")
        return self.quantized_model
    
    def dynamic_quantization(self):
        """Apply dynamic quantization (simpler approach)"""
        print("Applying dynamic quantization...")
        
        self.quantized_model = quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Embedding}, 
            dtype=torch.qint8
        )
        
        print("Dynamic quantization completed!")
        return self.quantized_model
    
    def compare_models(self, test_loader):
        """Compare original and quantized models"""
        print("\n=== Model Comparison ===")
        
        # Original model evaluation
        original_loss = self.evaluate_model(test_loader)
        
        # Quantized model evaluation
        self.quantized_model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for user, course, grade in test_loader:
                output = self.quantized_model(user, course).squeeze()
                loss = criterion(output, grade)
                total_loss += loss.item()
        
        quantized_loss = total_loss / len(test_loader)
        
        print(f"Original Model MSE: {original_loss:.4f}")
        print(f"Quantized Model MSE: {quantized_loss:.4f}")
        print(f"Performance difference: {abs(original_loss - quantized_loss):.4f}")
        
        # Model size comparison
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in self.quantized_model.parameters() if hasattr(p, 'element_size'))
        
        print(f"Original model size: {original_size / 1024:.2f} KB")
        print(f"Quantized model size: {quantized_size / 1024:.2f} KB (estimated)")
        
        return original_loss, quantized_loss
    
    def save_models(self, save_dir='models'):
        """Save both original and quantized models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save original model
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'original_model.pth'))
        torch.save(self.model, os.path.join(save_dir, 'original_model_full.pth'))
        
        # Save quantized model
        torch.save(self.quantized_model.state_dict(), os.path.join(save_dir, 'quantized_model.pth'))
        torch.save(self.quantized_model, os.path.join(save_dir, 'quantized_model_full.pth'))
        
        # Save encoders
        import pickle
        with open(os.path.join(save_dir, 'student_encoder.pkl'), 'wb') as f:
            pickle.dump(self.le_student, f)
        with open(os.path.join(save_dir, 'course_encoder.pkl'), 'wb') as f:
            pickle.dump(self.le_course, f)
        
        print(f"Models saved to {save_dir}/")
    
    def recommend_courses_quantized(self, student_id, input_courses, top_k=5):
        """Generate recommendations using quantized model"""
        self.quantized_model.eval()
        
        # Convert input_courses to proper format
        user_tensor = torch.LongTensor([student_id])
        all_course_ids = torch.arange(len(self.le_course.classes_))
        user_ids = torch.full_like(all_course_ids, fill_value=student_id)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.quantized_model(user_ids, all_course_ids).squeeze()
        
        # Exclude already taken courses
        for course_name, _ in input_courses.items():
            try:
                course_idx = self.le_course.transform([course_name])[0]
                if course_idx < len(predictions):
                    predictions[course_idx] = float('-inf')
            except ValueError:
                continue  # Course not in training data
        
        # Get top recommendations
        top_indices = torch.topk(predictions, min(top_k, len(predictions))).indices
        top_courses = self.le_course.inverse_transform(top_indices.numpy())
        
        # Filter out already taken courses
        recommendations = [course for course in top_courses if course not in input_courses]
        
        return recommendations[:top_k]

def main():
    # Initialize the system
    system = CourseRecommendationSystem(
        data_path='data.json',
        embedding_dim=10,
        learning_rate=0.001
    )
    
    try:
        # Load and preprocess data
        df = system.load_and_preprocess_data()
        print(f"Loaded {len(df)} records")
        
        # Prepare data loaders
        train_loader, test_loader, test_df = system.prepare_data_loaders(df, batch_size=64)
        
        # Train the model
        system.train_model(train_loader, num_epochs=10)
        
        # Evaluate original model
        print("\n=== Original Model Evaluation ===")
        system.evaluate_model(test_loader)
        
        # Apply dynamic quantization (recommended approach)
        system.dynamic_quantization()
        
        # Compare models
        system.compare_models(test_loader)
        
        # Save models
        system.save_models()
        
        # Test recommendation with quantized model
        print("\n=== Testing Quantized Model Recommendations ===")
        sample_courses = {
            'CHEMISTRY LABORATORY': 8,
            'GENERAL CHEMISTRY': 7,
            'ELECTRICAL SCIENCES': 3,
            'ADDITIVE MANUFACTURING': 1,
            'PRACTICE SCHOOL I': 10,
            'PHYSICS LABORATORY': 5
        }
        
        recommendations = system.recommend_courses_quantized(
            student_id=123, 
            input_courses=sample_courses, 
            top_k=5
        )
        
        print(f"Top 5 recommendations: {recommendations}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure 'data.json' exists in the current directory")

if __name__ == "__main__":
    main()