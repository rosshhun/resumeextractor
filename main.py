import logging
import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from data.loader import load_resume_dataset, load_known_skills
from models.skill_extractor import SkillExtractor
from models.student_skill_extractor import StudentSkillExtractor
from utils.metrics import skill_extraction_accuracy
from utils.analysis import analyze_feature_importance, error_analysis
from config import *

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info(f"Using device: {DEVICE}")

        # Load and preprocess data
        resumes = load_resume_dataset()
        known_skills = load_known_skills()

        logger.info(f"Loaded {len(resumes)} resumes and {len(known_skills)} known skills")

        texts = [resume['text'] for resume in resumes]
        skills = [[1 if skill in resume['skills'] else 0 for skill in known_skills] for resume in resumes]

        X_train, X_test, y_train, y_test = train_test_split(texts, skills, test_size=0.2, random_state=RANDOM_SEED)
        logger.info(f"Split data into {len(X_train)} training samples and {len(X_test)} test samples")

        # Train the teacher model
        teacher_model = SkillExtractor()
        logger.info("Starting teacher model training...")
        teacher_model.fit(X_train, y_train)
        logger.info("Teacher model training completed.")

        # Evaluate teacher model
        y_pred_teacher = teacher_model.transform(X_test)
        teacher_accuracy = skill_extraction_accuracy(y_test, y_pred_teacher)
        logger.info(f"Teacher Model Test Accuracy: {teacher_accuracy:.4f}")

        # Hyperparameter tuning for student model
        kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        best_score = 0
        best_params = {}

        logger.info("Starting hyperparameter tuning for student model...")
        for lr in LR_VALUES:
            for epochs in EPOCH_VALUES:
                for alpha in ALPHA_VALUES:
                    scores = []
                    for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
                        X_train_fold, X_val_fold = [X_train[i] for i in train_index], [X_train[i] for i in val_index]
                        y_train_fold, y_val_fold = [y_train[i] for i in train_index], [y_train[i] for i in val_index]

                        logger.info(f"Fold {fold}: Training on {len(X_train_fold)} samples, validating on {len(X_val_fold)} samples")

                        student_model = StudentSkillExtractor(num_labels=len(known_skills), alpha=alpha)
                        student_model.fit(X_train_fold, y_train_fold, teacher_model=teacher_model, epochs=epochs, learning_rate=lr)
                        y_pred_student = student_model.predict(X_val_fold)
                        score = skill_extraction_accuracy(y_val_fold, y_pred_student)
                        scores.append(score)
                        logger.info(f"Fold {fold} - LR: {lr}, Epochs: {epochs}, Alpha: {alpha}, Score: {score:.4f}")

                    avg_score = np.mean(scores)
                    logger.info(f"Average score for LR={lr}, Epochs={epochs}, Alpha={alpha}: {avg_score:.4f}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {'lr': lr, 'epochs': epochs, 'alpha': alpha}

        logger.info(f"Best parameters: {best_params}")

        # Train final student model with best parameters
        logger.info("Training final student model with best parameters...")
        final_student_model = StudentSkillExtractor(num_labels=len(known_skills), alpha=best_params['alpha'])
        final_student_model.fit(X_train, y_train, teacher_model=teacher_model,
                                epochs=best_params['epochs'], learning_rate=best_params['lr'])
        logger.info("Final student model training completed.")

        # Evaluate the final student model
        y_pred_student = final_student_model.predict(X_test)
        student_accuracy = skill_extraction_accuracy(y_test, y_pred_student)
        logger.info(f"Final Student Model Test Accuracy: {student_accuracy:.4f}")

        # Compare teacher and student models
        logger.info(f"Teacher Model Accuracy: {teacher_accuracy:.4f}")
        logger.info(f"Student Model Accuracy: {student_accuracy:.4f}")
        if teacher_accuracy > 0:
            improvement = (student_accuracy - teacher_accuracy) / teacher_accuracy * 100
            logger.info(f"Improvement: {improvement:.2f}%")
        else:
            logger.info("Cannot calculate improvement percentage as teacher accuracy is 0.")

        # Analyze feature importance (if applicable to your student model)
        try:
            feature_importance = analyze_feature_importance(final_student_model)
            logger.info("Feature Importance:")
            for feature, importance in feature_importance.items():
                logger.info(f"{feature}: {importance:.4f}")
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {str(e)}")

        # Perform error analysis
        error_results = error_analysis(y_test, y_pred_student, known_skills)
        logger.info("Error Analysis Results:")
        logger.info(f"Confusion Matrix:\n{error_results['confusion_matrix']}")
        logger.info("Top 5 skills by F1-score:")
        for skill, metrics in list(error_results['skill_metrics'].items())[:5]:
            logger.info(
                f"{skill}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-score={metrics['f1_score']:.4f}")

        # Save the final student model
        model_filename = os.path.join(OUTPUT_DIR, 'best_student_skill_extraction_model.pth')
        logger.info(f"Saving final student model to {model_filename}")
        torch.save(final_student_model.state_dict(), model_filename)
        logger.info(f"Final student model saved successfully")

        # Export the final student model to ONNX
        onnx_filename = os.path.join(OUTPUT_DIR, 'student_skill_extraction_model.onnx')
        logger.info(f"Exporting final student model to ONNX: {onnx_filename}")
        final_student_model.to_onnx(onnx_filename)
        logger.info(f"Final student model exported to ONNX successfully")

        # Save known skills for later use
        with open(os.path.join(OUTPUT_DIR, 'known_skills.json'), 'w') as f:
            json.dump(known_skills, f)
        logger.info("Known skills saved successfully")

        logger.info("Skill extraction model training and evaluation completed.")

    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()