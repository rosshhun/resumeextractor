import json
import os
import pickle
from typing import List, Dict
import logging
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = 'output'


def load_model(model_path: str):
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    return model


def predict_skills(model, text: str) -> List[str]:
    logger.info("Predicting skills with custom model")
    predictions = model.predict([text])
    skills = [skill for pred in predictions for skill, _ in pred] if predictions else []
    return skills


def save_batch(data: List[Dict], output_file: str, batch_num: int):
    with open(f"{output_file}_batch_{batch_num}.json", 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Batch {batch_num} saved to {output_file}_batch_{batch_num}.json")


def read_job_descriptions(file_path: str) -> List[str]:
    logger.info(f"Reading job descriptions from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'job_description' in item:
                    yield item['job_description']
        else:
            logger.error(f"Expected a JSON array, but got {type(data)}")


def process_chunk(model, chunk, start_index, output_file, batch_size):
    processed_data = []
    batch_num = start_index // batch_size + 1

    for i, job_description in enumerate(chunk, start=start_index):
        logger.info(f"Processing job description {i+1}/{len(chunk)+start_index}")
        custom_skills = predict_skills(model, job_description)

        processed_data.append({
            "text": job_description[:100] + "...",
            "custom_model_skills": custom_skills
        })

        logger.info(f"Processed job description {i+1}.")
        logger.info(f"Custom model skills: {custom_skills}")

        if len(processed_data) % batch_size == 0:
            save_batch(processed_data, output_file, batch_num)
            processed_data = []
            batch_num += 1

    if processed_data:
        save_batch(processed_data, output_file, batch_num)


def process_job_descriptions(input_file: str, output_file: str, model_path: str, batch_size: int = 1000):
    model = load_model(model_path)

    job_descriptions = list(read_job_descriptions(input_file))

    if not job_descriptions:
        logger.warning("No job descriptions found. Check the input file.")
        return

    num_processes = multiprocessing.cpu_count()
    chunk_size = len(job_descriptions) // num_processes

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, [(model, job_descriptions[i:i+chunk_size], i, output_file, batch_size)
                                               for i in range(0, len(job_descriptions), chunk_size)])

    logger.info(f"All data processed and saved in batches")


if __name__ == '__main__':
    input_file = 'job_descriptions_formatted.json'
    output_file = 'bert_training_data'
    model_path = os.path.join(OUTPUT_DIR, 'best_skill_extraction_model.pkl')

    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist.")
    else:
        process_job_descriptions(input_file, output_file, model_path)