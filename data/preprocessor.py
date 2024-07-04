import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import IMPORTANT_SINGLE_CHAR_TOKENS
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
        # Convert to lowercase
        text = text.lower()

        # Expand abbreviations
        text = self.expand_abbreviations(text)

        # Replace hyphens and underscores with spaces, but keep them for compound words
        text = re.sub(r'(?<![a-z])[-_]|[-_](?![a-z])', ' ', text)
        text = self.handle_compound_words(text)

        # Remove punctuation, keeping important symbols
        text = ''.join(ch for ch in text if ch not in string.punctuation or ch in '#.+_-')

        # Tokenize
        tokens = word_tokenize(text)

        # Handle special cases, remove stop words, and lemmatize
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
        # Handle version numbers
        if VERSION_PATTERN.match(token):
            return token

        # Handle special characters in technology names
        if token.lower() in ['c++', 'c#', '.net', 'f#', 'r']:
            return token.lower()

        # Handle compound words
        if '-' in token or '_' in token:
            return token.replace('-', ' ').replace('_', ' ')

        return token