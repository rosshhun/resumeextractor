import os
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from config import OUTPUT_DIR, MODEL_NAME, DATASET_DIR
from data.preprocessor import AdvancedPreprocessor


def load_model(model_path, known_skills_path):
    # Load known skills
    with open(known_skills_path, 'r') as f:
        known_skills = json.load(f)

    # Load ONNX model
    ort_session = ort.InferenceSession(model_path)

    return ort_session, known_skills


def predict_skills(ort_session, text, known_skills, threshold=0.8):
    preprocessor = AdvancedPreprocessor()
    processed_text = preprocessor.preprocess_text(text)
    processed_text = ' '.join(processed_text)

    # Tokenize the text
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer(processed_text, return_tensors="np", padding=True, truncation=True, max_length=512)

    # Run inference
    ort_inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    predictions = ort_outputs[0][0]

    # Apply sigmoid function to get probabilities
    predictions = 1 / (1 + np.exp(-predictions))

    # Extract skills above the threshold
    skills = [(known_skills[i], float(prob)) for i, prob in enumerate(predictions) if prob > threshold]

    # Sort skills by probability (descending)
    skills.sort(key=lambda x: x[1], reverse=True)

    return skills


def main():
    model_path = os.path.join(OUTPUT_DIR, 'student_skill_extraction_model.onnx')
    known_skills_path = os.path.join(DATASET_DIR, 'known_skills.json')

    ort_session, known_skills = load_model(model_path, known_skills_path)

    text = """
    Bachelors/Master's degree with a focus in Information Technology / Computer Science or related field
    Expertise in JavaScript / TypeScript, HTML, CSS, Python
    Expertise with modern JS frameworks (React, JQuery, Angular, etc.) and responsive web design.
    """

    skills = predict_skills(ort_session, text, known_skills)

    print("\nPredicted skills:")
    for skill, probability in skills:
        print(f"{skill}: {probability:.4f}")


if __name__ == "__main__":
    main()