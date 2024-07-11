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
    Job description
Company Description

Artech is the 10th Largest IT Staffing Company in the US, according to Staffing Industry Analysts' 2012 annual report. Artech provides technical expertise to fill gaps in clients' immediate skill-sets availability, deliver emerging technology skill-sets, refresh existing skill base, allow for flexibility in project planning and execution phases, and provide budgeting/financial flexibility by offering contingent labor as a variable cost.

Job Description

We are on a mission to build a new streaming platform for IP Detail Records.

We need to build a collection platform in order to do this and at scale.

We collect billions of records every day! This will be built on an internal openstack environment using Go, Scala, C++ or Java.

What you will do:

Build distributed, high-throughput, real-time data data collection systems

Do it in Go, Scala, C/C++, Java or others

Connect with Kafka and Storm for a data streaming service and open source databases like MongoDB

Help us open source this project and build a new open source community.

Qualifications

Skills & requirements

Who you must be:

You have significant experience with Go and its standard library

Before Go, progressive experience with Scala, Java or C/C++. Perl, Python a plus.

Must be able to work in full-stack team with operations and QA team members.

Must be super collaborative with non-developers.

You tend to obsess over code simplicity and performance

Startup like team in a large enterprise.

Your Github shows your chops

Bonus:

You wrote your own data pipelines once or twice before (and know what you did wrong).

You have battle scars with MongDB, Kafka, Storm

Additional Information

For more information, Please contact

Siva Kumar

973-507-7543

siva.kumar(at)artechinfo.com
    """

    skills = predict_skills(ort_session, text, known_skills)

    print("\nPredicted skills:")
    for skill, probability in skills:
        print(f"{skill}: {probability:.4f}")


if __name__ == "__main__":
    main()