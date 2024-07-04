import json
from config import RESUME_DATASET_PATH, KNOWN_SKILLS_PATH, SKILL_SYNONYMS_PATH

def load_json_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_resume_dataset():
    return load_json_data(RESUME_DATASET_PATH)

def load_known_skills():
    return load_json_data(KNOWN_SKILLS_PATH)

def load_skill_synonyms():
    return load_json_data(SKILL_SYNONYMS_PATH)