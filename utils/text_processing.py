import numpy as np
from fuzzywuzzy import fuzz
from nltk.util import ngrams
from data.preprocessor import AdvancedPreprocessor

preprocessor = AdvancedPreprocessor()


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
    vec1 (numpy array): First vector
    vec2 (numpy array): Second vector

    Returns:
    float: Cosine similarity score
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def fuzzy_match(text, skill):
    """
    Calculate the fuzzy match score between a text and a skill.

    Args:
    text (str): Input text
    skill (str): Skill to match against

    Returns:
    float: Fuzzy match score (0-1)
    """
    return fuzz.partial_ratio(text.lower(), skill.lower()) / 100.0


def context_score(text, skill, window_size=5):
    """
    Calculate the context score for a skill within a text.

    Args:
    text (str): Input text
    skill (str): Skill to evaluate
    window_size (int): Size of the context window

    Returns:
    float: Context score
    """
    tokens = preprocessor.preprocess_text(text)
    skill_tokens = preprocessor.preprocess_text(skill)

    score = 0
    for i, token in enumerate(tokens):
        if token in skill_tokens:
            context = tokens[max(0, i - window_size):min(len(tokens), i + window_size + 1)]
            score += sum(1 for t in context if t in skill_tokens) / len(context)

    return score / len(skill_tokens) if skill_tokens else 0


def ngram_match(text, skill, n_range=(2, 4)):
    """
    Calculate the n-gram match score between a text and a skill.

    Args:
    text (str): Input text
    skill (str): Skill to match against
    n_range (tuple): Range of n-gram sizes to consider

    Returns:
    float: N-gram match score
    """
    text = text.lower()
    skill = skill.lower()
    text_ngrams = set()
    skill_ngrams = set()

    for n in range(n_range[0], n_range[1] + 1):
        text_ngrams.update(''.join(gram) for gram in ngrams(text, n))
        skill_ngrams.update(''.join(gram) for gram in ngrams(skill, n))

    intersection = text_ngrams.intersection(skill_ngrams)
    return len(intersection) / len(skill_ngrams) if skill_ngrams else 0