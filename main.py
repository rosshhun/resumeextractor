import logging
from sklearn.model_selection import train_test_split
from data.loader import load_resume_dataset, load_known_skills
from models.skill_extractor import SkillExtractor
from utils.metrics import skill_accuracy_scorer
from utils.analysis import analyze_feature_importance, error_analysis
from config import OUTPUT_DIR
import os
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Using device: cpu")  # We're not using GPU for the main process

    resumes = load_resume_dataset()
    known_skills = load_known_skills()

    texts = [resume['text'] for resume in resumes]
    skills = [resume['skills'] for resume in resumes]

    X_train, X_test, y_train, y_test = train_test_split(texts, skills, test_size=0.2, random_state=42)

    skill_extractor = SkillExtractor()

    logger.info("Starting model training...")
    skill_extractor.fit(X_train, y_train)
    logger.info("Model training completed.")

    y_pred = skill_extractor.predict(X_test)
    test_accuracy = skill_accuracy_scorer(skill_extractor, X_test, y_test)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    try:
        feature_importance = analyze_feature_importance(skill_extractor)
        logger.info("Feature Importance:")
        for feature, importance in feature_importance.items():
            logger.info(f"{feature}: {importance:.4f}")
    except Exception as e:
        logger.error(f"An error occurred during feature importance analysis: {str(e)}")

    try:
        error_analysis_results = error_analysis(y_test, y_pred, known_skills)
        logger.info("Error Analysis:")
        logger.info(f"Confusion Matrix:\n{error_analysis_results['confusion_matrix']}")
        logger.info("Top 5 skills by F1-score:")
        for skill, metrics in list(error_analysis_results['skill_metrics'].items())[:5]:
            logger.info(
                f"{skill}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1_score']:.4f}")
    except Exception as e:
        logger.error(f"An error occurred during error analysis: {str(e)}")

    model_filename = os.path.join(OUTPUT_DIR, 'best_skill_extraction_model.pkl')
    logger.info(f"Saving model to {model_filename}")
    with open(model_filename, 'wb') as f:
        pickle.dump(skill_extractor, f)
    logger.info(f"Model saved successfully")

    # Test loading the model
    logger.info("Testing model loading...")
    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)
    logger.info("Model loaded successfully")

    # Test the loaded model
    test_prediction = loaded_model.predict(X_test[:1])
    logger.info(f"Test prediction from loaded model: {test_prediction}")

if __name__ == "__main__":
    main()