import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data.preprocessor import AdvancedPreprocessor
from utils.text_processing import cosine_similarity, fuzzy_match, context_score, ngram_match
from config import (SKILL_EXTRACTOR_THRESHOLD, TFIDF_MAX_FEATURES,
                    HIDDEN_LAYER_SIZES, BATCH_SIZE, MAX_EPOCHS)
from models.fasttext_model import train_fasttext
from data.loader import load_known_skills, load_skill_synonyms
import logging
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class SkillExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=SKILL_EXTRACTOR_THRESHOLD, tfidf_max_features=TFIDF_MAX_FEATURES,
                 learning_rate=0.001, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                 hidden_layer_sizes=HIDDEN_LAYER_SIZES, dropout_rate=0.5, device=None):
        self.threshold = threshold
        self.tfidf_max_features = tfidf_max_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SkillExtractor using device: {self.device}")
        self.tfidf_vectorizer = None
        self.fasttext_model = None
        self.svd = TruncatedSVD(n_components=100)
        self.preprocessor = AdvancedPreprocessor()
        self.model = None
        self.scaler = MinMaxScaler()
        self.known_skills = load_known_skills()
        self.skill_synonyms = load_skill_synonyms()
        self._is_fitted = False
        self._current_epoch = 0

    def fit(self, X, y):
        if isinstance(X, (pd.Series, np.ndarray)):
            X = X.tolist()

        if not self._is_fitted:
            self._initial_fit(X, y)
        else:
            self._continue_fit(X, y)

        return self

    def _initial_fit(self, X, y):
        logger.info("Preparing texts for TF-IDF and FastText")
        prepared_texts = [' '.join(self.preprocessor.preprocess_text(str(text))) for text in X]
        prepared_skills = [' '.join(self.preprocessor.preprocess_text(str(skill))) for skill in self.known_skills]
        all_texts = prepared_texts + prepared_skills

        logger.info("Fitting TF-IDF vectorizer")
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 4), lowercase=True,
                                                max_features=self.tfidf_max_features)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        logger.info(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")

        logger.info("Applying SVD to TF-IDF features")
        self.svd.fit(tfidf_matrix)

        logger.info("Training FastText model")
        self.fasttext_model = train_fasttext(X, self.known_skills)
        logger.info(f"FastText vocabulary size: {len(self.fasttext_model.wv.key_to_index)}")

        self._train_model(X, y)

    def _continue_fit(self, X, y):
        self._train_model(X, y, continue_training=True)

    def _train_model(self, X, y, continue_training=False):
        logger.info("Preparing features for neural network")
        features = []
        labels = []
        for text, skills in zip(X, y):
            for skill in self.known_skills:
                features.append(self._get_similarity_features(text, skill))
                labels.append(1 if skill in skills else 0)

        X_train = self.scaler.fit_transform(features)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(labels, dtype=torch.float32).to(self.device)

        if not continue_training:
            input_size = X_train.shape[1]
            self.model = Net(input_size, self.hidden_layer_sizes, self.dropout_rate).to(self.device)
            self._current_epoch = 0

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger.info(f"{'Continuing' if continue_training else 'Starting'} neural network training")
        for epoch in range(self._current_epoch, self.epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
            self._current_epoch += 1

        self._is_fitted = True

    def _get_similarity_features(self, text, skill):
        text_features = self.extract_features(' '.join(self.preprocessor.preprocess_text(text)))
        skill_features = self.extract_features(skill)
        cosine_sim = cosine_similarity(text_features, skill_features)
        fuzzy_sim = fuzzy_match(text, skill)
        context_sim = context_score(text, skill)
        ngram_sim = ngram_match(text, skill, n_range=(2, 4))

        # Add more features
        jaccard_sim = self._jaccard_similarity(text, skill)
        levenshtein_dist = self._levenshtein_distance(text, skill)

        return [cosine_sim, fuzzy_sim, context_sim, ngram_sim, jaccard_sim, levenshtein_dist]

    def transform(self, X):
        if isinstance(X, (pd.Series, np.ndarray)):
            X = X.tolist()
        logger.info(f"Transforming {len(X)} samples")
        return [self.extract_skills(str(text)) for text in X]

    def predict(self, X):
        return self.transform(X)

    def extract_skills(self, text):
        preprocessed_text = self.preprocessor.preprocess_text(text)
        text_features = self.extract_features(' '.join(preprocessed_text))

        extracted_skills = set()
        for skill in self.known_skills:
            skill_lower = skill.lower()
            synonyms = self.skill_synonyms.get(skill_lower, [skill_lower])

            if any(synonym in preprocessed_text for synonym in synonyms):
                extracted_skills.add((skill, 1.0))
                continue

            similarity_features = self._get_similarity_features(text, skill)
            similarity_features = torch.tensor(similarity_features, dtype=torch.float32).to(self.device)
            combined_score = self.model(similarity_features).item()

            if combined_score > self.threshold:
                extracted_skills.add((skill, combined_score))

        return sorted(extracted_skills, key=lambda x: x[1], reverse=True)

    def extract_features(self, text):
        tfidf_features = self.tfidf_vectorizer.transform([text])
        svd_features = self.svd.transform(tfidf_features)
        fasttext_features = np.mean(
            [self.fasttext_model.wv[token] for token in text.split() if token in self.fasttext_model.wv], axis=0)
        return np.concatenate([svd_features[0], fasttext_features])

    def get_feature_importance(self):
        return self.model.get_feature_importance()

    def _jaccard_similarity(self, text1, text2):
        set1 = set(self.preprocessor.preprocess_text(text1))
        set2 = set(self.preprocessor.preprocess_text(text2))
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) > 0 else 0

    def _levenshtein_distance(self, text1, text2):
        from Levenshtein import distance
        return 1 - (distance(text1, text2) / max(len(text1), len(text2)))


class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(Net, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_feature_importance(self):
        # Return the weights of the first layer as feature importance
        return self.model[0].weight.data.cpu().numpy().mean(axis=0)