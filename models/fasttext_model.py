import os
import logging
import fasttext
import numpy as np
import psutil
from config import (FASTTEXT_VECTOR_SIZE, FASTTEXT_WINDOW, FASTTEXT_MIN_COUNT,
                    FASTTEXT_EPOCHS, OUTPUT_DIR, MAX_CPU_USAGE)

logger = logging.getLogger(__name__)


def get_optimal_threads():
    cpu_count = psutil.cpu_count()
    return max(1, int(cpu_count * MAX_CPU_USAGE))


def prepare_training_data(texts, known_skills, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(f"__label__text {text}\n")
        for skill in known_skills:
            f.write(f"__label__skill {skill}\n")
    logger.info(f"Training data prepared and saved to {output_file}")


def train_fasttext(input_file, model_path):
    logger.info("Training FastText model from scratch")
    optimal_threads = get_optimal_threads()

    model = fasttext.train_unsupervised(
        input=input_file,
        model='skipgram',
        dim=FASTTEXT_VECTOR_SIZE,
        ws=FASTTEXT_WINDOW,
        minCount=FASTTEXT_MIN_COUNT,
        epoch=FASTTEXT_EPOCHS,
        thread=optimal_threads
    )
    model.save_model(model_path)
    logger.info(f"FastText model saved to {model_path}")
    return model


def load_fasttext_model(model_path):
    return fasttext.load_model(model_path)


def get_word_vector(model, word):
    return model.get_word_vector(word)


def get_sentence_vector(model, sentence):
    """
    Get the sentence vector using FastText model.
    Remove newlines and extra spaces from the sentence.
    """
    # Remove newlines and extra spaces
    cleaned_sentence = ' '.join(sentence.split())
    return model.get_sentence_vector(cleaned_sentence)
