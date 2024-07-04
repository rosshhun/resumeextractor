import pickle
import logging
from sklearn.model_selection import train_test_split
from data.loader import load_resume_dataset, load_known_skills
from models.skill_extractor import SkillExtractor
from utils.metrics import skill_accuracy_scorer
from utils.analysis import analyze_feature_importance, error_analysis
from config import (CV_FOLDS, OUTPUT_DIR, HYPERBAND_MIN_ITER, HYPERBAND_MAX_ITER,
                    HYPERBAND_ETA, PARAM_DISTRIBUTIONS, RESOURCE_PARAM)
from sklearn.pipeline import Pipeline
from custom_optimizer import HyperbandSearchCV, CustomInteger
import torch
from testmodel.test_model import test_model
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    resumes = load_resume_dataset()
    known_skills = load_known_skills()

    texts = [resume['text'] for resume in resumes]
    skills = [resume['skills'] for resume in resumes]

    X_train, X_test, y_train, y_test = train_test_split(texts, skills, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('skill_extractor', SkillExtractor(device=device))
    ])

    param_distributions = PARAM_DISTRIBUTIONS.copy()
    param_distributions['skill_extractor__tfidf_max_features'] = CustomInteger(
        *PARAM_DISTRIBUTIONS['skill_extractor__tfidf_max_features'])

    n_jobs = multiprocessing.cpu_count() - 1  # Use all but one CPU core

    hyperband_search = HyperbandSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        resource_param=RESOURCE_PARAM,
        min_iter=HYPERBAND_MIN_ITER,
        max_iter=HYPERBAND_MAX_ITER,
        eta=HYPERBAND_ETA,
        scoring=skill_accuracy_scorer,
        n_jobs=n_jobs,
        cv=CV_FOLDS,
        random_state=42,
        verbose=2
    )

    logger.info("Starting Hyperband optimization...")
    try:
        hyperband_search.fit(X_train, y_train)
        logger.info("Hyperband optimization completed.")
        logger.info(f"Best parameters: {hyperband_search.best_params_}")
        logger.info(f"Best score: {hyperband_search.best_score_:.4f}")
    except Exception as e:
        logger.error(f"An error occurred during Hyperband optimization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return

    best_skill_extractor = hyperband_search.best_estimator_
    y_pred = best_skill_extractor.predict(X_test)
    test_accuracy = skill_accuracy_scorer(best_skill_extractor, X_test, y_test)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    try:
        feature_importance = analyze_feature_importance(best_skill_extractor.named_steps['skill_extractor'])
        logger.info("Feature Importance:")
        for feature, importance in feature_importance.items():
            logger.info(f"{feature}: {importance:.4f}")
    except Exception as e:
        logger.error(f"An error occurred during feature importance analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

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
        import traceback
        logger.error(traceback.format_exc())

    logger.info("Testing model on sample data:")
    test_model(best_skill_extractor, resumes)

    model_filename = OUTPUT_DIR / 'best_skill_extraction_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_skill_extractor, f)

    logger.info(f"Best model saved successfully as {model_filename}")

if __name__ == "__main__":
    main()