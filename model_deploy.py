import torch
import torch.nn as nn
import pickle
import numpy as np
import os
from typing import Dict, List, Tuple
import json

class QuantizedFactorizationMachine(nn.Module):
    """Quantized version of the Factorization Machine for deployment"""
    
    def __init__(self, num_users, num_courses, embedding_dim):
        super(QuantizedFactorizationMachine, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.course_embedding = nn.Embedding(num_courses, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 2, 1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, user, course):
        user = self.quant(user.float())
        course = self.quant(course.float())
        
        user_emb = self.user_embedding(user.long())
        course_emb = self.course_embedding(course.long())
        
        interaction = torch.cat([user_emb, course_emb], dim=1)
        output = self.linear(interaction)
        
        output = self.dequant(output)
        return output

class ProductionRecommendationSystem:
    """Production-ready recommendation system using quantized model"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.quantized_model = None
        self.student_encoder = None
        self.course_encoder = None
        self.model_info = {}
        
    def load_quantized_model(self):
        """Load the quantized model and encoders"""
        try:
            # Load quantized model
            model_path = os.path.join(self.model_dir, 'quantized_model_full.pth')
            self.quantized_model = torch.load(model_path, map_location='cpu')
            self.quantized_model.eval()
            
            # Load encoders
            with open(os.path.join(self.model_dir, 'student_encoder.pkl'), 'rb') as f:
                self.student_encoder = pickle.load(f)
            with open(os.path.join(self.model_dir, 'course_encoder.pkl'), 'rb') as f:
                self.course_encoder = pickle.load(f)
            
            self.model_info = {
                'num_users': len(self.student_encoder.classes_),
                'num_courses': len(self.course_encoder.classes_),
                'courses_list': list(self.course_encoder.classes_)
            }
            
            print(" Quantized model loaded successfully!")
            print(f"Model supports {self.model_info['num_users']} users and {self.model_info['num_courses']} courses")
            
        except Exception as e:
            print(f" Error loading model: {e}")
            raise
    
    def predict_grade(self, student_id: int, course_name: str) -> float:
        """Predict grade for a specific student-course pair"""
        try:
            # Convert course name to ID
            course_id = self.course_encoder.transform([course_name])[0]
            
            # Create tensors
            user_tensor = torch.LongTensor([student_id])
            course_tensor = torch.LongTensor([course_id])
            
            # Get prediction
            with torch.no_grad():
                prediction = self.quantized_model(user_tensor, course_tensor).item()
            
            # Convert back to original scale (0-10)
            return prediction * 10.0
            
        except ValueError:
            print(f" Course '{course_name}' not found in training data")
            return 0.0
    
    def get_recommendations(self, student_id: int, taken_courses: List[str], 
                          exclude_courses: List[str] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k course recommendations for a student"""
        if exclude_courses is None:
            exclude_courses = []
        
        # Get all available courses
        all_courses = self.model_info['courses_list']
        
        # Filter out taken and excluded courses
        available_courses = [
            course for course in all_courses 
            if course not in taken_courses and course not in exclude_courses
        ]
        
        # Get predictions for all available courses
        recommendations = []
        for course in available_courses:
            predicted_grade = self.predict_grade(student_id, course)
            recommendations.append((course, predicted_grade))
        
        # Sort by predicted grade and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def batch_recommendations(self, student_id: int, candidate_courses: List[str]) -> List[Tuple[str, float]]:
        """Get predictions for a batch of courses (more efficient)"""
        try:
            # Convert course names to IDs
            course_ids = []
            valid_courses = []
            
            for course in candidate_courses:
                try:
                    course_id = self.course_encoder.transform([course])[0]
                    course_ids.append(course_id)
                    valid_courses.append(course)
                except ValueError:
                    continue
            
            if not course_ids:
                return []
            
            # Create tensors
            user_ids = torch.LongTensor([student_id] * len(course_ids))
            course_tensor = torch.LongTensor(course_ids)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.quantized_model(user_ids, course_tensor).squeeze()
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
            
            # Convert to original scale and pair with course names
            results = []
            for course, pred in zip(valid_courses, predictions):
                results.append((course, pred.item() * 10.0))
            
            return sorted(results, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f" Error in batch recommendations: {e}")
            return []
    
    def filter_by_branch(self, recommendations: List[Tuple[str, float]], 
                        branch: str, cdc_courses: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        """Filter recommendations to exclude CDC courses for a specific branch"""
        if branch not in cdc_courses:
            return recommendations
        
        branch_cdc = set(cdc_courses[branch])
        filtered = [
            (course, score) for course, score in recommendations 
            if course not in branch_cdc
        ]
        return filtered
    
    def get_model_stats(self) -> Dict:
        """Get model statistics and metadata"""
        model_size = sum(p.numel() * p.element_size() for p in self.quantized_model.parameters() if hasattr(p, 'element_size'))
        
        return {
            'model_type': 'Quantized Factorization Machine',
            'num_users': self.model_info['num_users'],
            'num_courses': self.model_info['num_courses'],
            'model_size_kb': model_size / 1024,
            'quantization': 'INT8 Dynamic',
            'supported_courses': len(self.model_info['courses_list'])
        }
    
    def save_deployment_config(self, config_path: str = 'deployment_config.json'):
        """Save deployment configuration"""
        config = {
            'model_info': self.model_info,
            'model_stats': self.get_model_stats(),
            'deployment_settings': {
                'batch_size': 32,
                'max_recommendations': 10,
                'grade_threshold': 5.0
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f" Deployment config saved to {config_path}")

class RecommendationAPI:
    """Simple API wrapper for the recommendation system"""
    
    def __init__(self, model_dir: str = 'models'):
        self.system = ProductionRecommendationSystem(model_dir)
        self.system.load_quantized_model()
        
        # Load CDC courses for filtering
        self.cdc_courses = self._load_cdc_courses()
    
    def _load_cdc_courses(self) -> Dict[str, List[str]]:
        """Load CDC courses for different branches"""
        # This would typically load from your CDCs.csv file
        # For now, return empty dict
        return {}
    
    def recommend(self, student_id: int, taken_courses: List[str], 
                 branch: str = None, top_k: int = 5) -> Dict:
        """Main recommendation endpoint"""
        try:
            # Get recommendations
            recommendations = self.system.get_recommendations(
                student_id=student_id,
                taken_courses=taken_courses,
                top_k=top_k * 2  # Get more to allow for filtering
            )
            
            # Filter by branch if specified
            if branch and self.cdc_courses:
                recommendations = self.system.filter_by_branch(
                    recommendations, branch, self.cdc_courses
                )
            
            # Limit to top_k
            recommendations = recommendations[:top_k]
            
            return {
                'status': 'success',
                'student_id': student_id,
                'branch': branch,
                'recommendations': [
                    {
                        'course': course,
                        'predicted_grade': round(score, 2),
                        'confidence': min(score / 10.0, 1.0)
                    }
                    for course, score in recommendations
                ],
                'total_recommendations': len(recommendations)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_performance(self, student_id: int, course_name: str) -> Dict:
        """Predict performance for a specific course"""
        try:
            predicted_grade = self.system.predict_grade(student_id, course_name)
            
            return {
                'status': 'success',
                'student_id': student_id,
                'course': course_name,
                'predicted_grade': round(predicted_grade, 2),
                'confidence': min(predicted_grade / 10.0, 1.0),
                'recommendation': 'Recommended' if predicted_grade >= 6.0 else 'Not Recommended'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

def main():
    """Demo the quantized model deployment"""
    print(" Quantized Course Recommendation System Demo")
    print("=" * 50)
    
    try:
        # Initialize the API
        api = RecommendationAPI()
        
        # Print model stats
        stats = api.system.get_model_stats()
        print(" Model Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 50)
        
        # Demo 1: Get recommendations
        print(" Demo 1: Course Recommendations")
        taken_courses = [
            'CHEMISTRY LABORATORY',
            'GENERAL CHEMISTRY',
            'ELECTRICAL SCIENCES',
            'ADDITIVE MANUFACTURING',
            'PRACTICE SCHOOL I',
            'PHYSICS LABORATORY'
        ]
        
        result = api.recommend(
            student_id=123,
            taken_courses=taken_courses,
            branch='B.E Electrical & Electronic',
            top_k=5
        )
        
        if result['status'] == 'success':
            print(f"Top {len(result['recommendations'])} recommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"  {i}. {rec['course']} (Grade: {rec['predicted_grade']}, Confidence: {rec['confidence']:.2f})")
        else:
            print(f" Error: {result['message']}")
        
        print("\n" + "=" * 50)
        
        # Demo 2: Performance prediction
        print(" Demo 2: Performance Prediction")
        test_course = "MACHINE LEARNING"
        
        result = api.predict_performance(
            student_id=123,
            course_name=test_course
        )
        
        if result['status'] == 'success':
            print(f"Prediction for '{test_course}':")
            print(f"  Predicted Grade: {result['predicted_grade']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Recommendation: {result['recommendation']}")
        else:
            print(f" Error: {result['message']}")
        
        # Save deployment config
        api.system.save_deployment_config()
        
    except Exception as e:
        print(f" Initialization error: {e}")
        print("Make sure you have run the training script first to generate the quantized model.")

if __name__ == "__main__":
    main()