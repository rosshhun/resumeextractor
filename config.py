# import os
# from pathlib import Path
# import numpy as np
#
# # Project root directory
# ROOT_DIR = Path(__file__).resolve().parent
#
# # Data directory
# DATASET_DIR = ROOT_DIR / 'dataset'
#
# # File paths
# RESUME_DATASET_PATH = DATASET_DIR / 'resume_dataset.json'
# KNOWN_SKILLS_PATH = DATASET_DIR / 'known_skills.json'
# SKILL_SYNONYMS_PATH = DATASET_DIR / 'skill_synonyms.json'
#
# # Model parameters
# FASTTEXT_VECTOR_SIZE = 300
# FASTTEXT_WINDOW = 10
# FASTTEXT_MIN_COUNT = 5
# FASTTEXT_EPOCHS = 50
#
# # Skill extractor parameters
# SKILL_EXTRACTOR_THRESHOLD = 0.5
# TFIDF_MAX_FEATURES = 10000
#
# # Important single-character tokens
# IMPORTANT_SINGLE_CHAR_TOKENS = {'c', 'r', 'j'}
#
# # Cross-validation parameters
# CV_FOLDS = 3  # Increased from 3 to 5
#
# # Hyperband parameters
# HYPERBAND_MIN_ITER = 1
# HYPERBAND_MAX_ITER = 50  # Increased from 81 to 243 (3^5)
# HYPERBAND_ETA = 3
#
# # Neural network parameters
# HIDDEN_LAYER_SIZES = [128, 64, 32]
# BATCH_SIZE = 32
# MAX_EPOCHS = 50  # Increased from 100 to 300
#
# # Ensure dataset directory exists
# os.makedirs(DATASET_DIR, exist_ok=True)
#
# # Output directory for analysis results
# OUTPUT_DIR = ROOT_DIR / 'output'
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# # Expanded Hyperparameter search space
# PARAM_DISTRIBUTIONS = {
#     'skill_extractor__threshold': np.linspace(0.1, 0.9, 9).tolist(),
#     'skill_extractor__tfidf_max_features': (5000, 30000),
#     'skill_extractor__learning_rate': np.logspace(-5, -1, 5).tolist(),
#     'skill_extractor__batch_size': [16, 32, 64, 128, 256],
#     'skill_extractor__hidden_layer_sizes': [(64,), (128,), (256,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
#     'skill_extractor__dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
# }
#
# # Resource parameter for Hyperband
# RESOURCE_PARAM = 'skill_extractor__epochs'
#
#

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Data directory
DATASET_DIR = ROOT_DIR / 'dataset'

# File paths
RESUME_DATASET_PATH = DATASET_DIR / 'resume_dataset.json'
KNOWN_SKILLS_PATH = DATASET_DIR / 'known_skills.json'
SKILL_SYNONYMS_PATH = DATASET_DIR / 'skill_synonyms.json'

# Output directory
OUTPUT_DIR = ROOT_DIR / 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FastText parameters
FASTTEXT_VECTOR_SIZE = 50  # Reduced from 100
FASTTEXT_WINDOW = 3  # Reduced from 5
FASTTEXT_MIN_COUNT = 2
FASTTEXT_EPOCHS = 3  # Reduced from 10

# Skill extractor parameters
SKILL_EXTRACTOR_THRESHOLD = 0.5
TFIDF_MAX_FEATURES = 500  # Reduced from 1000

# XGBoost parameters
XGBOOST_MAX_DEPTH = 3
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_N_ESTIMATORS = 50  # Increased for early stopping
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8

# Training parameters
EPOCHS = 3
PATIENCE = 1  # For early stopping
BATCH_SIZE = 32

# Cross-validation parameters
CV_FOLDS = 2
CV_SCORING = 'f1_weighted'

# Data Augmentation parameters
AUGMENTATION_FACTOR = 0.1  # Percentage of data to augment
NUM_AUGMENTATIONS = 1  # Number of augmentations per sample

# Important single-character tokens
IMPORTANT_SINGLE_CHAR_TOKENS = {'c', 'r', 'j'}

# Resource utilization
MAX_CPU_USAGE = 0.9  # Use up to 90% of CPU cores

# GPU settings
USE_GPU = True  # Set to False to force CPU usage
PIN_MEMORY = True  # Faster data transfer to GPU