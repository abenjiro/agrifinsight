"""Machine Learning Models Package"""

from app.ml_models.disease_detector import get_disease_detector, DiseaseDetector
from app.ml_models.class_mappings import CLASS_NAMES, TREATMENT_RECOMMENDATIONS

__all__ = [
    'get_disease_detector',
    'DiseaseDetector',
    'CLASS_NAMES',
    'TREATMENT_RECOMMENDATIONS'
]
