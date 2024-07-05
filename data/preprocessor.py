import re
import string
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import IMPORTANT_SINGLE_CHAR_TOKENS, AUGMENTATION_FACTOR, NUM_AUGMENTATIONS
from data.loader import load_skill_synonyms

SKILL_SYNONYMS = load_skill_synonyms()

COMPOUND_WORD_PATTERN = re.compile(r'\b\w+[-_]\w+\b')
ABBREVIATION_PATTERN = re.compile(r'\b[A-Z\.]+\b')
VERSION_PATTERN = re.compile(r'\b\d+(\.\d+)+\b')

class AdvancedPreprocessor:
    def __init__(self):
        self.skill_synonyms = SKILL_SYNONYMS
        self.important_single_char_tokens = IMPORTANT_SINGLE_CHAR_TOKENS
        self.stop_words = set(stopwords.words('english')) - self.important_single_char_tokens
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        text = text.lower()
        text = self.expand_abbreviations(text)
        text = re.sub(r'(?<![a-z])[-_]|[-_](?![a-z])', ' ', text)
        text = self.handle_compound_words(text)
        text = ''.join(ch for ch in text if ch not in string.punctuation or ch in '#.+_-')
        tokens = word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            token = self.handle_special_cases(token)
            if token not in self.stop_words or token in self.important_single_char_tokens:
                lemmatized_token = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized_token)
        return processed_tokens

    def expand_abbreviations(self, text):
        def replace(match):
            abbr = match.group(0).lower().replace('.', '')
            for skill, synonyms in self.skill_synonyms.items():
                if abbr in synonyms:
                    return skill
            return match.group(0)
        return ABBREVIATION_PATTERN.sub(replace, text)

    def handle_compound_words(self, text):
        return COMPOUND_WORD_PATTERN.sub(lambda match: match.group(0).replace('-', ' ').replace('_', ' '), text)

    def handle_special_cases(self, token):
        if VERSION_PATTERN.match(token):
            return token
        if token.lower() in ['c++', 'c#', '.net', 'f#', 'r']:
            return token.lower()
        if '-' in token or '_' in token:
            return token.replace('-', ' ').replace('_', ' ')
        return token

    def augment_data(self, text, skills):
        augmented_data = []
        for _ in range(NUM_AUGMENTATIONS):
            augmented_text = self.synonym_replacement(text)
            augmented_text = self.random_insertion(augmented_text)
            augmented_text = self.random_deletion(augmented_text)
            augmented_data.append((augmented_text, skills))
        return augmented_data

    def synonym_replacement(self, text):
        words = self.preprocess_text(text)
        n = max(1, int(len(words) * AUGMENTATION_FACTOR))
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        return ' '.join(new_words)

    def random_insertion(self, text):
        words = self.preprocess_text(text)
        n = max(1, int(len(words) * AUGMENTATION_FACTOR))
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return ' '.join(new_words)

    def random_deletion(self, text):
        words = self.preprocess_text(text)
        if len(words) == 1:
            return words[0]
        n = max(1, int(len(words) * AUGMENTATION_FACTOR))
        new_words = words.copy()
        for _ in range(n):
            self.remove_word(new_words)
        return ' '.join(new_words)

    def get_synonyms(self, word):
        synonyms = set()
        for skill, syn_list in self.skill_synonyms.items():
            if word in syn_list:
                synonyms.update(syn_list)
        return synonyms

    def add_word(self, words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = random.choice(words)
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(list(synonyms))
        random_idx = random.randint(0, len(words)-1)
        words.insert(random_idx, random_synonym)

    def remove_word(self, words):
        if len(words) > 1:
            random_idx = random.randint(0, len(words)-1)
            words.pop(random_idx)