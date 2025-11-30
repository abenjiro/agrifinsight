"""
Disease class mappings and treatment recommendations
Generated from PlantVillage dataset with 38 classes
"""

# Class names in the exact order they were trained
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper_bell___Bacterial_spot',
    'Pepper_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Treatment recommendations for each disease
TREATMENT_RECOMMENDATIONS = {
    'Apple___Apple_scab': {
        'disease_name': 'Apple Scab',
        'severity': 'moderate',
        'description': 'A fungal disease that causes dark, scabby lesions on leaves and fruit.',
        'treatments': [
            'Apply fungicides containing captan or myclobutanil during early spring',
            'Remove and destroy infected leaves and fruit',
            'Prune trees to improve air circulation',
            'Choose resistant apple varieties for future planting'
        ],
        'prevention': [
            'Rake and remove fallen leaves in autumn',
            'Apply dormant spray in early spring',
            'Maintain proper spacing between trees'
        ]
    },
    'Apple___Black_rot': {
        'disease_name': 'Apple Black Rot',
        'severity': 'high',
        'description': 'A fungal disease causing fruit rot and leaf spot.',
        'treatments': [
            'Remove infected fruit, leaves, and branches',
            'Apply fungicides (captan, thiophanate-methyl) from pink bud stage',
            'Prune dead wood and cankers',
            'Destroy mummified fruit'
        ],
        'prevention': [
            'Maintain tree vigor through proper fertilization',
            'Ensure good drainage',
            'Remove potential infection sources'
        ]
    },
    'Apple___Cedar_apple_rust': {
        'disease_name': 'Cedar Apple Rust',
        'severity': 'moderate',
        'description': 'A fungal disease requiring both apple and cedar trees to complete its lifecycle.',
        'treatments': [
            'Apply fungicides (myclobutanil, propiconazole) from green tip through June',
            'Remove nearby cedar/juniper trees if possible',
            'Plant resistant apple varieties'
        ],
        'prevention': [
            'Avoid planting apple trees near cedar/juniper',
            'Choose resistant cultivars',
            'Monitor cedar trees for galls'
        ]
    },
    'Cherry___Powdery_mildew': {
        'disease_name': 'Cherry Powdery Mildew',
        'severity': 'moderate',
        'description': 'White powdery fungal growth on leaves and shoots.',
        'treatments': [
            'Apply sulfur or potassium bicarbonate sprays',
            'Use fungicides (myclobutanil, propiconazole)',
            'Prune infected shoots',
            'Improve air circulation'
        ],
        'prevention': [
            'Choose resistant varieties',
            'Avoid overhead irrigation',
            'Remove infected plant debris'
        ]
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
        'disease_name': 'Gray Leaf Spot',
        'severity': 'moderate',
        'description': 'Fungal disease causing rectangular gray lesions on corn leaves.',
        'treatments': [
            'Apply foliar fungicides (azoxystrobin, pyraclostrobin)',
            'Rotate to non-host crops',
            'Use resistant hybrids'
        ],
        'prevention': [
            'Practice crop rotation (2-3 years)',
            'Till under crop residue',
            'Plant resistant varieties'
        ]
    },
    'Corn___Common_rust': {
        'disease_name': 'Common Rust',
        'severity': 'moderate',
        'description': 'Small, circular to elongate brown pustules on leaves.',
        'treatments': [
            'Apply fungicides if disease appears before tasseling',
            'Use triazole or strobilurin fungicides',
            'Plant resistant hybrids'
        ],
        'prevention': [
            'Plant early-maturing varieties',
            'Choose resistant hybrids',
            'Scout fields regularly'
        ]
    },
    'Corn___Northern_Leaf_Blight': {
        'disease_name': 'Northern Leaf Blight',
        'severity': 'high',
        'description': 'Long, elliptical gray-green lesions on corn leaves.',
        'treatments': [
            'Apply fungicides (azoxystrobin, propiconazole)',
            'Remove crop residue',
            'Rotate crops',
            'Use resistant hybrids'
        ],
        'prevention': [
            'Practice 2-3 year crop rotation',
            'Till under residue',
            'Plant resistant varieties'
        ]
    },
    'Grape___Black_rot': {
        'disease_name': 'Grape Black Rot',
        'severity': 'high',
        'description': 'Fungal disease causing mummified berries and leaf lesions.',
        'treatments': [
            'Apply fungicides (mancozeb, myclobutanil) from bud break',
            'Remove mummified fruit',
            'Prune for air circulation',
            'Remove infected leaves'
        ],
        'prevention': [
            'Destroy infected fruit and leaves',
            'Maintain vine canopy management',
            'Apply preventive fungicides'
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        'disease_name': 'Grape Black Measles (Esca)',
        'severity': 'high',
        'description': 'Complex fungal disease causing wood decay and leaf symptoms.',
        'treatments': [
            'Remove severely infected vines',
            'Prune out affected wood',
            'No highly effective chemical treatment available',
            'Maintain vine health through proper nutrition'
        ],
        'prevention': [
            'Avoid wounding vines',
            'Use clean pruning tools',
            'Delay pruning until late winter'
        ]
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'disease_name': 'Grape Leaf Blight',
        'severity': 'moderate',
        'description': 'Fungal leaf spot disease.',
        'treatments': [
            'Apply copper-based fungicides',
            'Remove infected leaves',
            'Improve air circulation'
        ],
        'prevention': [
            'Proper canopy management',
            'Avoid overhead irrigation',
            'Remove plant debris'
        ]
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'disease_name': 'Citrus Greening (HLB)',
        'severity': 'critical',
        'description': 'Devastating bacterial disease spread by psyllid insects.',
        'treatments': [
            'No cure available - remove infected trees',
            'Control psyllid vectors with insecticides',
            'Use antibiotic trunk injections (experimental)',
            'Implement quarantine measures'
        ],
        'prevention': [
            'Use certified disease-free nursery stock',
            'Control Asian citrus psyllid populations',
            'Remove infected trees promptly',
            'Implement regional disease management'
        ]
    },
    'Peach___Bacterial_spot': {
        'disease_name': 'Bacterial Spot',
        'severity': 'moderate',
        'description': 'Bacterial disease causing leaf spots and fruit lesions.',
        'treatments': [
            'Apply copper sprays',
            'Use oxytetracycline during bloom',
            'Prune to improve air flow',
            'Remove infected plant material'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Avoid overhead irrigation',
            'Use drip irrigation',
            'Apply preventive copper sprays'
        ]
    },
    'Pepper_bell___Bacterial_spot': {
        'disease_name': 'Bacterial Spot on Pepper',
        'severity': 'moderate',
        'description': 'Bacterial disease causing leaf and fruit spots.',
        'treatments': [
            'Apply copper-based bactericides',
            'Remove infected plants',
            'Improve air circulation',
            'Avoid working with wet plants'
        ],
        'prevention': [
            'Use disease-free seeds and transplants',
            'Practice crop rotation (3-4 years)',
            'Avoid overhead irrigation',
            'Disinfect tools'
        ]
    },
    'Potato___Early_blight': {
        'disease_name': 'Early Blight',
        'severity': 'moderate',
        'description': 'Fungal disease causing concentric ring lesions on leaves.',
        'treatments': [
            'Apply fungicides (chlorothalonil, mancozeb)',
            'Remove infected leaves',
            'Improve air circulation',
            'Avoid overhead irrigation'
        ],
        'prevention': [
            'Rotate crops (3-4 years)',
            'Use disease-free seed potatoes',
            'Mulch to prevent soil splash',
            'Maintain plant vigor'
        ]
    },
    'Potato___Late_blight': {
        'disease_name': 'Late Blight',
        'severity': 'critical',
        'description': 'Highly destructive fungal disease that can destroy entire crops.',
        'treatments': [
            'Apply fungicides (chlorothalonil, mancozeb) preventively',
            'Destroy infected plants immediately',
            'Harvest potatoes before disease spreads',
            'Use systemic fungicides in severe cases'
        ],
        'prevention': [
            'Use certified disease-free seed',
            'Avoid planting near tomatoes',
            'Destroy volunteer potatoes',
            'Monitor weather for favorable disease conditions',
            'Apply preventive fungicide sprays'
        ]
    },
    'Squash___Powdery_mildew': {
        'disease_name': 'Powdery Mildew',
        'severity': 'moderate',
        'description': 'White powdery fungal growth on leaves.',
        'treatments': [
            'Apply sulfur or potassium bicarbonate',
            'Use fungicides (myclobutanil)',
            'Remove heavily infected leaves',
            'Improve air circulation'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Space plants properly',
            'Avoid overhead watering',
            'Apply preventive sprays'
        ]
    },
    'Strawberry___Leaf_scorch': {
        'disease_name': 'Leaf Scorch',
        'severity': 'moderate',
        'description': 'Fungal disease causing purple-bordered leaf spots.',
        'treatments': [
            'Remove and destroy infected leaves',
            'Apply fungicides during bloom and harvest',
            'Improve air circulation',
            'Renovate strawberry bed after harvest'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Avoid overhead irrigation',
            'Renovate beds regularly',
            'Remove plant debris'
        ]
    },
    'Tomato___Bacterial_spot': {
        'disease_name': 'Bacterial Spot',
        'severity': 'moderate',
        'description': 'Bacterial disease affecting leaves and fruit.',
        'treatments': [
            'Apply copper-based sprays',
            'Remove infected plants',
            'Use resistant varieties',
            'Improve air circulation'
        ],
        'prevention': [
            'Use disease-free transplants',
            'Practice crop rotation',
            'Avoid overhead irrigation',
            'Stake and prune plants'
        ]
    },
    'Tomato___Early_blight': {
        'disease_name': 'Early Blight',
        'severity': 'moderate',
        'description': 'Fungal disease with concentric ring lesions.',
        'treatments': [
            'Apply fungicides (chlorothalonil, mancozeb)',
            'Remove lower infected leaves',
            'Mulch to prevent soil splash',
            'Improve air circulation'
        ],
        'prevention': [
            'Rotate crops (3-4 years)',
            'Mulch plants',
            'Avoid overhead watering',
            'Stake and prune properly'
        ]
    },
    'Tomato___Late_blight': {
        'disease_name': 'Late Blight',
        'severity': 'critical',
        'description': 'Devastating disease that can destroy crops rapidly.',
        'treatments': [
            'Apply fungicides (chlorothalonil, mancozeb) immediately',
            'Remove and destroy infected plants',
            'Avoid overhead irrigation',
            'Use systemic fungicides if needed'
        ],
        'prevention': [
            'Use certified disease-free transplants',
            'Avoid planting near potatoes',
            'Monitor weather conditions',
            'Apply preventive fungicides'
        ]
    },
    'Tomato___Leaf_Mold': {
        'disease_name': 'Leaf Mold',
        'severity': 'moderate',
        'description': 'Fungal disease common in greenhouses.',
        'treatments': [
            'Improve ventilation',
            'Reduce humidity',
            'Apply fungicides (chlorothalonil)',
            'Remove infected leaves'
        ],
        'prevention': [
            'Ensure good air circulation',
            'Reduce humidity',
            'Space plants adequately',
            'Plant resistant varieties'
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'disease_name': 'Septoria Leaf Spot',
        'severity': 'moderate',
        'description': 'Fungal disease causing small circular leaf spots.',
        'treatments': [
            'Apply fungicides (chlorothalonil, mancozeb)',
            'Remove infected lower leaves',
            'Mulch to prevent splash',
            'Improve air flow'
        ],
        'prevention': [
            'Rotate crops',
            'Mulch plants',
            'Stake and prune',
            'Avoid overhead watering'
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'disease_name': 'Spider Mites',
        'severity': 'moderate',
        'description': 'Tiny pests causing stippling and webbing on leaves.',
        'treatments': [
            'Spray with water to dislodge mites',
            'Apply insecticidal soap or neem oil',
            'Use miticides if infestation is severe',
            'Release predatory mites'
        ],
        'prevention': [
            'Maintain adequate moisture',
            'Avoid over-fertilizing',
            'Monitor regularly',
            'Encourage beneficial insects'
        ]
    },
    'Tomato___Target_Spot': {
        'disease_name': 'Target Spot',
        'severity': 'moderate',
        'description': 'Fungal disease causing concentric ring lesions.',
        'treatments': [
            'Apply fungicides (chlorothalonil, azoxystrobin)',
            'Remove infected leaves',
            'Improve air circulation',
            'Mulch to prevent splash'
        ],
        'prevention': [
            'Rotate crops',
            'Space plants properly',
            'Remove plant debris',
            'Avoid overhead irrigation'
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'disease_name': 'Tomato Yellow Leaf Curl Virus',
        'severity': 'high',
        'description': 'Viral disease spread by whiteflies.',
        'treatments': [
            'Remove infected plants immediately',
            'Control whitefly vectors with insecticides',
            'Use reflective mulches',
            'No cure available'
        ],
        'prevention': [
            'Use virus-resistant varieties',
            'Control whiteflies',
            'Use insect screening in greenhouses',
            'Remove infected plants promptly'
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'disease_name': 'Tomato Mosaic Virus',
        'severity': 'high',
        'description': 'Viral disease causing mottled leaves and stunted growth.',
        'treatments': [
            'Remove and destroy infected plants',
            'Disinfect tools and hands',
            'No cure available',
            'Control aphid vectors'
        ],
        'prevention': [
            'Use resistant varieties',
            'Avoid handling tobacco before tomatoes',
            'Disinfect tools regularly',
            'Control aphids'
        ]
    },
    # Healthy plants
    'Apple___healthy': {
        'disease_name': 'Healthy Apple',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Maintain good cultural practices', 'Monitor regularly for early disease detection']
    },
    'Blueberry___healthy': {
        'disease_name': 'Healthy Blueberry',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Maintain proper soil pH', 'Ensure adequate drainage', 'Monitor for pests']
    },
    'Cherry___healthy': {
        'disease_name': 'Healthy Cherry',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Prune for air circulation', 'Remove fallen fruit and leaves']
    },
    'Corn___healthy': {
        'disease_name': 'Healthy Corn',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Rotate crops', 'Monitor for pests and diseases']
    },
    'Grape___healthy': {
        'disease_name': 'Healthy Grape',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Maintain good canopy management', 'Monitor regularly']
    },
    'Peach___healthy': {
        'disease_name': 'Healthy Peach',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Prune for air circulation', 'Monitor for pests']
    },
    'Pepper_bell___healthy': {
        'disease_name': 'Healthy Pepper',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Maintain good spacing', 'Monitor for pests']
    },
    'Potato___healthy': {
        'disease_name': 'Healthy Potato',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Use certified seed', 'Rotate crops', 'Monitor regularly']
    },
    'Raspberry___healthy': {
        'disease_name': 'Healthy Raspberry',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Prune properly', 'Maintain good air circulation']
    },
    'Soybean___healthy': {
        'disease_name': 'Healthy Soybean',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Rotate crops', 'Monitor for pests']
    },
    'Strawberry___healthy': {
        'disease_name': 'Healthy Strawberry',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Renovate beds regularly', 'Remove old leaves']
    },
    'Tomato___healthy': {
        'disease_name': 'Healthy Tomato',
        'severity': 'none',
        'description': 'Plant appears healthy with no disease symptoms.',
        'treatments': ['Continue regular maintenance and monitoring'],
        'prevention': ['Stake and prune properly', 'Rotate crops', 'Monitor regularly']
    }
}


def get_class_info(class_name):
    """Get formatted disease information"""
    parts = class_name.split('___')
    crop = parts[0].replace('_', ' ')
    condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'

    is_healthy = 'healthy' in class_name.lower()

    return {
        'crop': crop,
        'condition': condition,
        'is_healthy': is_healthy,
        'full_name': class_name
    }
