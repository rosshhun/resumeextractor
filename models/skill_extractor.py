import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from data.preprocessor import AdvancedPreprocessor
from utils.text_processing import cosine_similarity, fuzzy_match, context_score, ngram_match
from config import (SKILL_EXTRACTOR_THRESHOLD, TFIDF_MAX_FEATURES,
                    XGBOOST_MAX_DEPTH, XGBOOST_LEARNING_RATE, XGBOOST_N_ESTIMATORS,
                    XGBOOST_SUBSAMPLE, XGBOOST_COLSAMPLE_BYTREE, USE_GPU, PIN_MEMORY,
                    BATCH_SIZE, OUTPUT_DIR, CV_FOLDS, PATIENCE)
from models.fasttext_model import (prepare_training_data, train_fasttext,
                                   load_fasttext_model, get_sentence_vector)
from data.loader import load_known_skills, load_skill_synonyms
import logging
from sklearn.decomposition import TruncatedSVD
from Levenshtein import distance
import os
import psutil

logger = logging.getLogger(__name__)


class SkillExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=SKILL_EXTRACTOR_THRESHOLD, tfidf_max_features=TFIDF_MAX_FEATURES):
        self.threshold = threshold
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_vectorizer = None
        self.fasttext_model = None
        self.svd = TruncatedSVD(n_components=100)
        self.preprocessor = AdvancedPreprocessor()
        self.model = None
        self.scaler = MinMaxScaler()
        self.known_skills = load_known_skills()
        self.skill_synonyms = load_skill_synonyms()
        self.fasttext_model_path = os.path.join(OUTPUT_DIR, 'fasttext_model')
        self.device = 'cpu'  # We're not using GPU for now

    def fit(self, X, y):
        if isinstance(X, (pd.Series, np.ndarray)):
            X = X.tolist()

        logger.info("Performing data augmentation")
        augmented_data = []
        for text, skills in zip(X, y):
            augmented_data.extend(self.preprocessor.augment_data(text, skills))

        X_augmented, y_augmented = zip(*augmented_data)
        X.extend(X_augmented)
        y.extend(y_augmented)

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

        logger.info("Preparing FastText training data")
        training_data_path = os.path.join(OUTPUT_DIR, 'fasttext_training_data.txt')
        prepare_training_data(prepared_texts, prepared_skills, training_data_path)

        logger.info("Training FastText model")
        train_fasttext(training_data_path, self.fasttext_model_path)

        logger.info("Loading FastText model")
        self.fasttext_model = load_fasttext_model(self.fasttext_model_path)

        logger.info("Preparing features for XGBoost")
        features = []
        labels = []
        for text, skills in zip(X, y):
            for skill in self.known_skills:
                features.append(self._get_similarity_features(text, skill))
                labels.append(1 if skill in skills else 0)

        X_train = self.scaler.fit_transform(features)
        y_train = np.array(labels)

        logger.info("Training XGBoost model with cross-validation")
        self.model = self._create_xgboost_model()

        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            eval_set = [(X_fold_val, y_fold_val)]
            self.model.fit(
                X_fold_train, y_fold_train,
                eval_set=eval_set,
                verbose=False
            )

            y_fold_pred = self.model.predict(X_fold_val)
            fold_score = f1_score(y_fold_val, y_fold_pred, average='weighted')
            cv_scores.append(fold_score)
            logger.info(f"Fold {fold} F1 Score: {fold_score:.4f}")

        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

        # Final fit on all data
        eval_set = [(X_train, y_train)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        logger.info(f"Best iteration: {self.model.best_iteration}")
        logger.info(f"Best score: {self.model.best_score}")

        return self

    def _create_xgboost_model(self):
        tree_method = 'hist'  # Default to CPU
        if USE_GPU:
            try:
                # Check if GPU is available for XGBoost
                param = {'tree_method': 'gpu_hist'}
                xgb.train(param, xgb.DMatrix(np.random.randn(1, 1), label=[0]))
                tree_method = 'gpu_hist'
                logger.info("GPU is available for XGBoost. Using GPU acceleration.")
            except xgb.core.XGBoostError:
                logger.info("GPU is not available for XGBoost. Falling back to CPU.")

        return XGBClassifier(
            max_depth=XGBOOST_MAX_DEPTH,
            learning_rate=XGBOOST_LEARNING_RATE,
            n_estimators=XGBOOST_N_ESTIMATORS,
            subsample=XGBOOST_SUBSAMPLE,
            colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
            tree_method=tree_method,
            n_jobs=psutil.cpu_count(logical=False),
            enable_categorical=True,
            early_stopping_rounds=PATIENCE,
            eval_metric='logloss'
        )

    def _get_similarity_features(self, text, skill):
        text_features = self.extract_features(text)
        skill_features = self.extract_features(skill)
        cosine_sim = cosine_similarity(text_features, skill_features)
        fuzzy_sim = fuzzy_match(text, skill)
        context_sim = context_score(text, skill)
        ngram_sim = ngram_match(text, skill, n_range=(2, 4))
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
            similarity_features = self.scaler.transform([similarity_features])
            combined_score = self.model.predict_proba(similarity_features)[0][1]

            if combined_score > self.threshold:
                extracted_skills.add((skill, combined_score))

        return sorted(extracted_skills, key=lambda x: x[1], reverse=True)

    def extract_features(self, text):
        tfidf_features = self.tfidf_vectorizer.transform([text])
        svd_features = self.svd.transform(tfidf_features)
        fasttext_features = get_sentence_vector(self.fasttext_model, text)
        return np.concatenate([svd_features[0], fasttext_features])

    def get_feature_importance(self):
        return self.model.feature_importances_

    def _jaccard_similarity(self, text1, text2):
        set1 = set(self.preprocessor.preprocess_text(text1))
        set2 = set(self.preprocessor.preprocess_text(text2))
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) > 0 else 0

    def _levenshtein_distance(self, text1, text2):
        return 1 - (distance(text1, text2) / max(len(text1), len(text2)))

    def __getstate__(self):
        """Custom method for pickling the object"""
        state = self.__dict__.copy()
        # Don't pickle fasttext_model, but save the path
        del state['fasttext_model']
        state['fasttext_model_path'] = self.fasttext_model_path
        return state

    def __setstate__(self, state):
        """Custom method for unpickling the object"""
        self.__dict__.update(state)
        # Load the fasttext model
        self.fasttext_model = load_fasttext_model(self.fasttext_model_path)