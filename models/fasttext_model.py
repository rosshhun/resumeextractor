from gensim.models import FastText
from gensim.models.phrases import Phrases, Phraser
from config import FASTTEXT_VECTOR_SIZE, FASTTEXT_WINDOW, FASTTEXT_MIN_COUNT, FASTTEXT_EPOCHS
from data.preprocessor import AdvancedPreprocessor
import torch

preprocessor = AdvancedPreprocessor()

def train_fasttext(texts, known_skills):
    tokenized_texts = [preprocessor.preprocess_text(text) for text in texts]

    bigram = Phrases(tokenized_texts, min_count=5, threshold=10)
    trigram = Phrases(bigram[tokenized_texts], threshold=10)

    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    tokenized_texts = [trigram_mod[bigram_mod[doc]] for doc in tokenized_texts]

    for skill in known_skills:
        skill_tokens = preprocessor.preprocess_text(skill)
        if len(skill_tokens) > 1:
            tokenized_texts.append(skill_tokens)
        if len(skill_tokens) > 1:
            tokenized_texts.append([skill.lower().replace(' ', '_')])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastText(
        sentences=tokenized_texts,
        vector_size=FASTTEXT_VECTOR_SIZE,
        window=FASTTEXT_WINDOW,
        min_count=FASTTEXT_MIN_COUNT,
        workers=4,
        epochs=FASTTEXT_EPOCHS,
        sg=1
    )

    if device.type == 'cuda':
        model.wv.vectors = torch.tensor(model.wv.vectors, device=device)
        model.wv.vectors_norm = torch.tensor(model.wv.vectors_norm, device=device) if model.wv.vectors_norm is not None else None

    return model