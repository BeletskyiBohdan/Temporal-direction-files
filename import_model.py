!pip install spacy keybert networkx stanza pymorphy2 pymorphy2-dicts-uk

import torch
import json
import pymorphy2
import re
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product

# 2. Завантаження словника
with open("/kaggle/input/temporal-direction-ua/scikitlearn/500-epoches-10100-texts/6/vocab.json", "r", encoding="utf-8") as f:
    word_to_int = json.load(f)

# 3. Визначення архітектури моделі
class TemporalModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TemporalModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim * 2, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 32)
        self.fc6 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        emb1 = self.embedding(x1).mean(dim=1)
        emb2 = self.embedding(x2).mean(dim=1)
        x = torch.cat((emb1, emb2), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        return self.fc6(x).squeeze()

# 4. Ініціалізація пристрою (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Створення екземпляру моделі та завантаження ваг
embedding_dim = 100  # Повинно відповідати під час тренування
model = TemporalModel(len(word_to_int), embedding_dim)
model.to(device)

# Завантаження ваг моделі
model.load_state_dict(torch.load("/kaggle/input/temporal-direction-ua/scikitlearn/500-epoches-10100-texts/6/word-model.pth", map_location=device))
model.eval()

# 6. Ініціалізація морфологічного аналізатора
morph = pymorphy2.MorphAnalyzer(lang='uk')

# 7. Визначення функції очищення тексту
def lemmatize_text(input_text, morph):
    words = input_text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)

def clean_text(text, stopwords, morph=pymorphy2.MorphAnalyzer(lang='uk')):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"(?:ht|f)tps?:\/\/[-a-zA-Z0-9.]+\.[a-zA-Z]{2,3}(?:\/(?:[^\"<=]|=)*)?", '', text)
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)
    text = re.sub(r'(?<=\b[а-яіїєґА-ЯІЇЄҐ])(\s)(?=[а-яіїєґА-ЯІЇЄҐ]\b)', lambda match: match.group(0).replace(" ", ""), text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"[^\w\s'ʼ]", '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r"[^а-яіїєґ'ʼ\s]", '', text)

    for stop_phrase in stopwords:
        text = re.sub(r'\b' + re.escape(stop_phrase) + r'\b', '', text)

    return lemmatize_text(text, morph)

# 8. Визначення функції temporal_direction
def temporal_direction(word1, word2):
    """
    Функція обчислює темпоральний показник (скаляр) для пари слів,
    виходячи з тренованої моделі та словника word_to_int.
    Повертає float (плюсові/мінусові значення).
    """
    # Перевірка наявності слів у словнику
    if word1 not in word_to_int or word2 not in word_to_int:
        print(f"One or both words '{word1}' and '{word2}' are not in the vocabulary.")
        return 0

    # Отримання індексів слів
    index1 = word_to_int[word1]
    index2 = word_to_int[word2]

    # Підготовка тензорів
    word1_tensor = torch.tensor([index1], dtype=torch.long, device=device)
    word2_tensor = torch.tensor([index2], dtype=torch.long, device=device)

    with torch.no_grad():
        # Отримання embeddings
        embedding1 = model.embedding(word1_tensor).to(torch.float32)
        embedding2 = model.embedding(word2_tensor).to(torch.float32)

        # Конкатенація embeddings
        input_tensor = torch.cat((embedding1, embedding2), dim=1)

        # Прогон через модель
        x = model.relu(model.fc1(input_tensor))
        x = model.dropout(x)
        x = model.relu(model.fc2(x))
        x = model.dropout(x)
        x = model.relu(model.fc3(x))
        x = model.dropout(x)
        x = model.relu(model.fc4(x))
        x = model.dropout(x)
        x = model.relu(model.fc5(x))
        temporal_indicator = model.fc6(x).item()

    return temporal_indicator

# 9. Приклад використання функції temporal_direction
if __name__ == "__main__":
    # Визначення стоп-слів (приклад, замініть на ваші)
    stopwords = set(["і", "в", "на", "з", "до", "що", "є", "за", "це", "ти"])

    # Приклади слів
    word_a = "англіканський"
    word_b = "торг"

    # Обчислення темпорального показника
    score = temporal_direction(word_a, word_b)
    print(f"Темпоральний показник між словами '{word_a}' та '{word_b}': {score}")
