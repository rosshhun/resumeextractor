import pickle
import logging
from data.loader import load_resume_dataset
from utils.metrics import skill_extraction_accuracy
from config import OUTPUT_DIR
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def test_model(model, test_data, num_samples=10):
    random.shuffle(test_data)
    test_samples = test_data[:num_samples]

    total_accuracy = 0
    for sample in test_samples:
        text = sample['text']
        true_skills = set(sample['skills'])

        predicted_skills = set(skill for skill, _ in model.predict([text])[0])

        accuracy = len(true_skills.intersection(predicted_skills)) / len(true_skills.union(predicted_skills))
        total_accuracy += accuracy

        logger.info(f"Sample text: {text[:100]}...")
        logger.info(f"True skills: {true_skills}")
        logger.info(f"Predicted skills: {predicted_skills}")
        logger.info(f"Accuracy: {accuracy:.4f}\n")

    average_accuracy = total_accuracy / num_samples
    logger.info(f"Average accuracy across {num_samples} samples: {average_accuracy:.4f}")

def main():
    model_path = OUTPUT_DIR / 'best_skill_extraction_model.pkl'
    model = load_model(model_path)

    resumes = load_resume_dataset()

    logger.info("Testing model on sample data:")
    test_model(model, resumes)


if __name__ == "__main__":
    main()