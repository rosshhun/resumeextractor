import os
import json
import torch
from models.student_skill_extractor import StudentSkillExtractor
from config import OUTPUT_DIR, MODEL_NAME, NUM_LABELS, DEVICE, DATASET_DIR
from data.preprocessor import AdvancedPreprocessor


def load_model(model_path, known_skills_path):
    # Load known skills
    with open(known_skills_path, 'r') as f:
        known_skills = json.load(f)

    # Initialize the model
    model = StudentSkillExtractor(model_name=MODEL_NAME, num_labels=len(known_skills), device=DEVICE)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    return model, known_skills


def predict_skills(model, text, known_skills, threshold=0.8):
    preprocessor = AdvancedPreprocessor()
    processed_text = preprocessor.preprocess_text(text)
    processed_text = ' '.join(processed_text)

    # Predict
    with torch.no_grad():
        predictions = model.transform([processed_text])[0]

    # Extract skills above the threshold
    skills = [(known_skills[i], float(prob)) for i, prob in enumerate(predictions) if prob > threshold]

    # Sort skills by probability (descending)
    skills.sort(key=lambda x: x[1], reverse=True)

    return skills


def main():
    model_path = os.path.join(OUTPUT_DIR, 'best_student_skill_extraction_model.pth')
    known_skills_path = os.path.join(DATASET_DIR, 'known_skills.json')

    model, known_skills = load_model(model_path, known_skills_path)
    text = """resume"""
    skills = predict_skills(model, text, known_skills)

    print("\nPredicted skills:")
    for skill, probability in skills:
        print(f"{skill}: {probability:.4f}")



if __name__ == "__main__":
    main()