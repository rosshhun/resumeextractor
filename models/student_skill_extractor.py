import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import *
import random
import nltk
from nltk.corpus import wordnet
import logging

nltk.download('wordnet', quiet=True)

logger = logging.getLogger(__name__)

class SkillDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label),
            'idx': idx  # Add this line to keep track of the original index
        }

    def augment_text(self, text):
        words = text.split()
        augmented_words = words.copy()

        # Synonym replacement
        if random.random() < SYNONYM_REPLACEMENT_PROB:
            n = max(1, int(len(words) * SYNONYM_REPLACEMENT_PROB))
            random_word_list = list(set([word for word in words if word.isalnum()]))
            random.shuffle(random_word_list)
            num_replaced = 0
            for random_word in random_word_list:
                synonyms = self.get_synonyms(random_word)
                if len(synonyms) >= 1:
                    synonym = random.choice(list(synonyms))
                    augmented_words = [synonym if word == random_word else word for word in augmented_words]
                    num_replaced += 1
                if num_replaced >= n:
                    break

        # Random deletion
        if random.random() < RANDOM_DELETION_PROB:
            augmented_words = [word for word in augmented_words if random.random() > RANDOM_DELETION_PROB]

        # Random swap
        if random.random() < RANDOM_SWAP_PROB:
            n = max(1, int(len(augmented_words) * RANDOM_SWAP_PROB))
            for _ in range(n):
                idx1, idx2 = random.sample(range(len(augmented_words)), 2)
                augmented_words[idx1], augmented_words[idx2] = augmented_words[idx2], augmented_words[idx1]

        # Random insertion
        if random.random() < RANDOM_INSERTION_PROB:
            n = max(1, int(len(augmented_words) * RANDOM_INSERTION_PROB))
            for _ in range(n):
                self.add_word(augmented_words)

        return ' '.join(augmented_words)

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def add_word(self, words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = random.choice(words)
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(words) - 1)
        words.insert(random_idx, random_synonym)


class StudentSkillExtractor(nn.Module, BaseEstimator, TransformerMixin):
    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS, device=DEVICE,
                 temperature=TEMPERATURE, alpha=ALPHA):
        super(StudentSkillExtractor, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def fit(self, X, y, teacher_model=None, epochs=EPOCHS, batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS):
        train_dataset = SkillDataset(X, y, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        criterion = nn.BCEWithLogitsLoss()

        if teacher_model:
            logger.info(f"Predicting probabilities for {len(X)} samples")
            teacher_probs = teacher_model.predict_proba(X)
            teacher_probs = torch.tensor(teacher_probs, dtype=torch.float32).to(self.device)
            logger.info(f"Teacher probabilities shape: {teacher_probs.shape}")

        for epoch in range(epochs):
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                indices = batch['idx']  # Get the original indices

                optimizer.zero_grad()
                student_logits = self(input_ids, attention_mask)

                if teacher_model:
                    teacher_logits = teacher_probs[indices]
                    loss = self._compute_distillation_loss(student_logits, labels, teacher_logits)
                else:
                    loss = criterion(student_logits, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        return self

    def transform(self, X):
        self.eval()
        dataset = SkillDataset(X, [np.zeros(self.num_labels) for _ in X], self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        results = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                results.extend(probs)
        return results

    def _compute_distillation_loss(self, student_logits, labels, teacher_logits):
        T = self.temperature

        soft_targets = torch.sigmoid(teacher_logits / T)
        student_probs = torch.sigmoid(student_logits / T)

        kl_div = F.kl_div(
            F.logsigmoid(student_logits / T),
            soft_targets,
            reduction='batchmean'
        ) * (T ** 2)

        student_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

        return (self.alpha * student_loss) + ((1 - self.alpha) * kl_div)

    def predict(self, X):
        probs = self.transform(X)
        return [[(i, prob) for i, prob in enumerate(sample) if prob > 0.5] for sample in probs]

    def to_onnx(self, output_path):
        self.eval()
        dummy_input = self.tokenizer("dummy input", return_tensors="pt")
        input_names = ['input_ids', 'attention_mask']
        output_names = ['output']
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size'}
        }

        torch.onnx.export(self,
                          (dummy_input['input_ids'], dummy_input['attention_mask']),
                          output_path,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          opset_version=ONNX_OPSET_VERSION,
                          do_constant_folding=True)