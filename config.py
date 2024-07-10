import os
from pathlib import Path
import torch

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

# Model parameters
MODEL_NAME = "prajjwal1/bert-tiny"  # Smaller model for faster training and inference
NUM_LABELS = 1  # This will be overwritten in main.py based on the actual number of skills

# Training parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32  # Increased for faster training
MAX_LENGTH = 256  # Reduced for faster processing, adjust based on your data
EPOCHS = 2  # Increased for better learning
LEARNING_RATE = 3e-5  # Slightly increased for faster convergence
WARMUP_STEPS = 0.1  # 10% of total steps for warmup
TEMPERATURE = 1.5  # Reduced for softer probabilities in distillation
ALPHA = 0.7  # More weight on true labels vs. teacher predictions
DROPOUT_RATE = 0.2  # Reduced slightly to retain more information

# SkillExtractor parameters
SKILL_EXTRACTOR_THRESHOLD = 0.4  # Lowered to catch more potential skills
TFIDF_MAX_FEATURES = 15000  # Increased for more features

# XGBoost parameters
XGBOOST_MAX_DEPTH = 2  # Increased for more complex trees
XGBOOST_LEARNING_RATE = 0.05  # Lowered for more stable learning
XGBOOST_N_ESTIMATORS = 200  # Increased number of trees
XGBOOST_SUBSAMPLE = 0.9  # Increased for more robust models
XGBOOST_COLSAMPLE_BYTREE = 0.9  # Increased for more feature utilization

# FastText parameters
FASTTEXT_VECTOR_SIZE = 200  # Reduced for faster training, still effective
FASTTEXT_WINDOW = 8  # Increased for broader context
FASTTEXT_MIN_COUNT = 2  # Increased to reduce noise from rare words
FASTTEXT_EPOCHS = 25  # Increased for better embeddings

# GPU settings
USE_GPU = True
PIN_MEMORY = True

# CPU settings
MAX_CPU_USAGE = 0.9  # Increased to utilize more CPU power

# Cross-validation parameters
CV_FOLDS = 3
CV_SPLITS = 3

# Early stopping
PATIENCE = 5  # Reduced for faster training cycles

# Data Augmentation parameters
AUGMENTATION_PROB = 0.7  # Increased for more augmentation
MAX_AUGMENTED_SAMPLES = 3  # Increased for more diversity
SYNONYM_REPLACEMENT_PROB = 0.15
RANDOM_DELETION_PROB = 0.05  # Reduced to preserve more original content
RANDOM_SWAP_PROB = 0.15
RANDOM_INSERTION_PROB = 0.15
AUGMENTATION_FACTOR = 0.3  # Increased for more augmentation
NUM_AUGMENTATIONS = 3  # Increased for more diverse samples

# Important single-character tokens
IMPORTANT_SINGLE_CHAR_TOKENS = {'c', 'r', 'j', 'C', 'R', 'J'}  # Added uppercase versions

# Hyperparameter tuning
LR_VALUES = [1e-5, 3e-5, 5e-5]  # Adjusted range
EPOCH_VALUES = [10, 20, 30]  # Increased for potentially better performance
ALPHA_VALUES = [0.5, 0.7, 0.9]  # Adjusted to favor true labels more

# ONNX export
ONNX_OPSET_VERSION = 12  # Updated to latest stable version

# Random seed for reproducibility
RANDOM_SEED = 42