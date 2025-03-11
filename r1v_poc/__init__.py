"""
R1-V Proof of Concept Integration Package

This package implements enhanced video analysis capabilities using the R1-V and R1-Omni models:
- Visual analysis with R1-V
- Emotion recognition with R1-Omni
- Recommendation system based on analyses
- Integration with Movie Maker application
"""

from .r1v_analyzer import R1VAnalyzer
from .emotion_detector import EmotionDetector
from .recommender import Recommender
from .model_utils import get_r1v_model, get_r1omni_model
from .r1v_routes import register_r1v_routes

__all__ = [
    'R1VAnalyzer',
    'EmotionDetector',
    'Recommender',
    'get_r1v_model',
    'get_r1omni_model',
    'register_r1v_routes'
] 