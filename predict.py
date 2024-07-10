from flask import Flask, render_template, request, jsonify
import os
import pickle
import logging
from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    return model

def predict_skills(model, text):
    logger.info("Predicting skills")
    predictions = model.predict([text])
    skills = []

    for pred in predictions:
        if isinstance(pred, list):
            logger.info("Prediction format: list of (skill, confidence) tuples")
            for skill, confidence in pred:
                skills.append({'skill': str(skill), 'confidence': float(confidence)})
        elif isinstance(pred, tuple):
            logger.info("Prediction format: (skill, confidence) tuple")
            skill, confidence = pred
            skills.append({'skill': str(skill), 'confidence': float(confidence)})
        else:
            logger.warning(f"Unexpected prediction format: {pred}")

    return skills

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_resume', methods=['POST'])
def process_resume():
    data = request.get_json()
    resume_text = data['resume_text']

    model_path = os.path.join(OUTPUT_DIR, 'best_skill_extraction_model.pkl')
    model = load_model(model_path)
    resume_skills = predict_skills(model, resume_text)
    logger.info(f"Resume skills predicted: {resume_skills}")

    return jsonify({'resume_skills': resume_skills})

@app.route('/process_job_description', methods=['POST'])
def process_job_description():
    data = request.get_json()
    job_description = data['job_description']
    resume_skills = data['resume_skills']

    model_path = os.path.join(OUTPUT_DIR, 'best_skill_extraction_model.pkl')
    model = load_model(model_path)
    job_skills = predict_skills(model, job_description)
    logger.info(f"Job skills predicted: {job_skills}")

    return jsonify({
        'job_skills': job_skills,
        'resume_skills': resume_skills
    })

if __name__ == '__main__':
    app.run(debug=True)