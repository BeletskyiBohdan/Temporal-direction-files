!pip install stanza pymorphy2 pymorphy2-dicts-uk

import gc
import itertools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pymorphy2
import random
import re
import seaborn as sns
import stanza
import statistics
import tempfile
import time
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from IPython.display import clear_output
from itertools import combinations
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV


config = {
    'json_file_path': 'news.json',
    'stopwords_file_path': 'stopwords_ua.txt',
    'text_file': 'encoded_texts.pkl',
    'date_file': 'normalized_dates.npy',
    'batch_size': 256,
    'num_samples': 400,
    'epochs': 100,
    'learning_rate': 0.00001,
    'vocab_file': 'vocab.json',
    'embedding_dim': 100,
    'load_model_file': 'word-model.pth',
    'model_file': 'tmodel5.pth',
    'results_csv': 'temporal_indicators.csv',
    'matrix_csv': 'temporal_matrix.csv',
    'max_seq_length': 128,
    'learning':False
}

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes} хв : {seconds} сек"


def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    messages = [msg for msg in data["messages"] if msg["type"] == "message"]
    dates = [datetime.strptime(msg["date"], "%Y-%m-%dT%H:%M:%S") for msg in messages]
    texts = [' '.join([part["text"] if isinstance(part, dict) else part for part in msg["text"]]) for msg in messages]
    
    return messages, dates, texts


def lemmatize_text(input_text, morph):
    words = input_text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)


def clean_text(text, stopwords, morph=pymorphy2.MorphAnalyzer(lang='uk')):
    text = text.lower()
    text = re.sub(r"(?:ht|f)tps?:\/\/[-a-zA-Z0-9.]+\.[a-zA-Z]{2,3}(?:\/(?:[^\"<=]|=)*)?", '', text)
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)
    text = re.sub(r'(?<=\b[а-яіїєґА-ЯІЇЄҐ])(\s)(?=[а-яіїєґА-ЯІЇЄҐ]\b)', lambda match: match.group(0).replace(" ", ""), text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"[^\w\s'ʼ]", '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"[^а-яіїєґ'ʼ\s]", '', text)

    for stop_phrase in stopwords:
        text = re.sub(r'\b' + re.escape(stop_phrase) + r'\b', '', text)

    return lemmatize_text(text, morph)


def preprocess_texts(texts, dates, stopwords, morph):
    def process_text_and_date(args):
        text, date = args
        cleaned = clean_text(text, stopwords, morph)
        return (cleaned, date) if cleaned else (None, None)

    cleaned_texts, cleaned_dates = [], []
    total_texts = len(texts)
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_text_and_date, zip(texts, dates))
        for idx, (cleaned, date) in enumerate(results, 1):
            if cleaned:
                cleaned_texts.append(cleaned)
                cleaned_dates.append(date)

            elapsed_time = time.time() - start_time
            average_time_per_text = elapsed_time / idx
            remaining_time = average_time_per_text * (total_texts - idx)
            elapsed_formatted = format_time(elapsed_time)
            remaining_formatted = format_time(remaining_time)
            print(f"Оброблено текстів: {idx}/{total_texts} | Пройшло: {elapsed_formatted} | Залишилось: {remaining_formatted}", end='\r')

    print()
    return cleaned_texts, cleaned_dates


def preprocess_texts_batch(texts, dates, stopwords, morph, batch_size=15000):
    cleaned_texts, cleaned_dates = [], []
    total_batches = (len(texts) + batch_size - 1) // batch_size  

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        
        batch_texts = texts[start:end]
        batch_dates = dates[start:end]
        
        
        cleaned_batch_texts, cleaned_batch_dates = preprocess_texts(batch_texts, batch_dates, stopwords, morph)
        
        cleaned_texts.extend(cleaned_batch_texts)
        cleaned_dates.extend(cleaned_batch_dates)
        
        
        gc.collect()
        
        
        print(f"Оброблено батчів: {batch_idx + 1}/{total_batches}", end='\r')
    
    print()  
    
    return cleaned_texts, cleaned_dates


def create_vocab(cleaned_texts, vocab_file_path):
    all_words = [word for text in cleaned_texts for word in text.split()]
    unique_words = list(set(all_words))
    word_to_int = {word: i for i, word in enumerate(unique_words)}
    
    with open(vocab_file_path, 'w') as f:
        json.dump(word_to_int, f)
    
    return word_to_int


def encode_and_save_texts(cleaned_texts, word_to_int, text_file_path):
    encoded_texts = [[word_to_int.get(word, -1) for word in text.split() if word in word_to_int] for text in cleaned_texts]
    with open(text_file_path, 'wb') as f:
        pickle.dump(encoded_texts, f)

def normalize_and_save_dates(dates, date_file_path):
    min_date = datetime(2019, 1, 1)
    normalized_dates = [(date - min_date).total_seconds() for date in dates]
    np.save(date_file_path, normalized_dates)

def generate_pairs(epoch, num_samples, batch_size):
    encoded_texts = np.load(config['text_file'], allow_pickle=True)
    normalized_dates = np.load(config['date_file'], allow_pickle=True)
    selected_indices = np.random.choice(len(encoded_texts), num_samples, replace=False)
    selected_texts = [encoded_texts[i] for i in selected_indices]
    selected_dates = [normalized_dates[i] for i in selected_indices]
    pairs = []
    targets = []
    max_seq_length = config['max_seq_length']
    for (msg1, date1), (msg2, date2) in combinations(zip(selected_texts, selected_dates), 2):
        indices1 = msg1[:max_seq_length] + [0] * max(0, max_seq_length - len(msg1))
        indices2 = msg2[:max_seq_length] + [0] * max(0, max_seq_length - len(msg2))
        pairs.append(indices1 + indices2)
        time_diff = sign((date2 - date1))
        targets.append(time_diff)
    pairs_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    targets_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    np.save(pairs_file.name, pairs)
    np.save(targets_file.name, targets)
    return pairs_file.name, targets_file.name

def train_model_optimized(model, num_epochs, num_samples, batch_size, device, max_memory_gb=20):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    train_losses = []
    max_gpu_batch_size = estimate_max_batch_size(model, device, max_memory_gb)
    batch_size = min(batch_size, max_gpu_batch_size)
    
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        pairs_file, targets_file = generate_pairs(epoch, num_samples, batch_size)
        epoch_loss = 0
        num_batches = 0
        for pairs_chunk, targets_chunk in load_data_in_chunks(pairs_file, targets_file, chunk_size=1000000):
            dataset = TensorDataset(
                torch.tensor(pairs_chunk, dtype=torch.long),
                torch.tensor(targets_chunk, dtype=torch.float32)
            )
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                model.train()
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                if num_batches % 100 == 0:
                    print(f"Batch {num_batches}, Loss: {loss.item():.4f}")
                torch.cuda.empty_cache()
        
        train_losses.append(epoch_loss / num_batches)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | loss: {epoch_loss / num_batches}")
        os.remove(pairs_file)
        os.remove(targets_file)

    return train_losses


def estimate_max_batch_size(model, device, max_memory_gb):
    with torch.no_grad():
        sample_input = torch.zeros((1, 2), dtype=torch.long).to(device)
        traced_model = torch.jit.trace(model, sample_input)
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        start_size = 1
        while True:
            try:
                test_input = torch.zeros((start_size, 2), dtype=torch.long).to(device)
                _ = optimized_model(test_input)
                start_size *= 2
            except RuntimeError:
                break
    max_size = start_size // 2
    return max_size


def load_data_in_chunks(pairs_file, targets_file, chunk_size):
    pairs = np.load(pairs_file, allow_pickle=True)
    targets = np.load(targets_file, allow_pickle=True)
    for i in range(0, len(pairs), chunk_size):
        yield pairs[i:i+chunk_size], targets[i:i+chunk_size]

def showImage(path):
    
    image = Image.open(path)
    
    plt.imshow(image)
    plt.axis('off')  
    plt.show()

def temporal_direction(word1, word2):
    if word1 not in word_to_int or word2 not in word_to_int:
        print(f"One or both words '{word1}' and '{word2}' are not in the vocabulary.") 
        return 0 

    index1 = word_to_int[word1]
    index2 = word_to_int[word2]

    model.eval()
    model.to(device)

    word1_tensor = torch.tensor([index1], dtype=torch.long, device=device)
    word2_tensor = torch.tensor([index2], dtype=torch.long, device=device)

    with torch.no_grad():
        embedding1 = model.embedding(word1_tensor).to(torch.float32)
        embedding2 = model.embedding(word2_tensor).to(torch.float32)

    input_tensor = torch.cat((embedding1, embedding2), dim=1)

    with torch.no_grad():
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


def random_temporal_directions_with_df(word_to_int, num_iterations=10000):
    results = []
    vocab_words = list(word_to_int.keys())

    for _ in range(num_iterations):
        word1, word2 = random.sample(vocab_words, 2)
        indicator = temporal_direction(word1, word2)
        results.append({'Word1': word1, 'Word2': word2, 'Indicator': indicator})

    df_results = pd.DataFrame(results)
    return df_results

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def count_in_ranges(scores):
    bins = [-float('inf'), -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, float('inf')]
    labels = ['менше -1', '-1', '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0', 
              '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1', 'більше 1']
    
    binned_scores = pd.cut(scores, bins=bins, labels=labels[:-1], right=False)
    count_dict = dict(Counter(binned_scores))
    count_in_order = {label: count_dict.get(label, 0) for label in labels}
    
    counts = np.array(list(count_in_order.values()))
    normalized_counts = softmax(counts)
    
    normalized_count_in_order = {label: float(value) for label, value in zip(labels, normalized_counts)}
    
    return normalized_count_in_order


def phi(a):
    return a**4 if -1 <= a <= 1 else 1/(a**4)

def phi2(a):
    return a**3 if -1 <= a <= 1 else 1/(a**3)

def sm(A):
    A_list = A.tolist() if isinstance(A, pd.Series) else A
    return sum(phi(a) for a in A_list) / len(A_list) if A_list else 0

def anti_sm(A):
    A_list = A.tolist() if isinstance(A, pd.Series) else A
    return sum(phi2(a) for a in A_list) / len(A_list) if A_list else 0

def filter_words(list_a):
    voc = list(word_to_int.keys())
    return list(filter(lambda x: x in voc, list_a))

def process_texts(text_a, text_b, graph=False):
    words_a = filter_words(clean_text(text_a, stopwords, morph).split())
    words_b = filter_words(clean_text(text_b, stopwords, morph).split())
    
    evaluations = [
        [word1, word2, score := temporal_direction(word1, word2), 1 if score > 0 else -1]
        for word1, word2 in itertools.product(words_a, words_b)
    ]
    
    df = pd.DataFrame(evaluations, columns=['Слово1', 'Слово2', 'Оцінка', 'Знак'])
    scores, signs = df['Оцінка'], df['Знак']
    
    stats = {
        'text1': text_a,
        'words1': len(words_a),
        'len1': len(text_a),
        'text2': text_b,
        'words2': len(words_b),
        'len2': len(text_a),
        'words1*2': len(words_a)*len(words_b),
        'words1+2': len(words_a)+len(words_b),
        'words1/2': len(words_a)/len(words_b),
        'words2/1': len(words_b)/len(words_a),
        'len1*2': len(text_a)*len(text_b),
        'len1+2': len(text_a)+len(text_b),
        'len1/2': len(text_a)/len(text_b),
        'len2/1': len(text_b)/len(text_a),
        'Mean': scores.mean(),
        'mean_sign': 1 if scores.mean() > 0 else -1,
        'Std Dev': scores.std() if len(scores) > 1 else 0,
        'sm_scores': sm(scores),
        'anti_sm_scores': anti_sm(scores),
        'max+min': scores.max() + scores.min(),
        'max-min': scores.max() - scores.min(),
        'Max': scores.max(),
        'Min': scores.min(),
        'Sum': scores.sum(),
        'sign_mean': signs.mean(),
        'std_dev_sign': signs.std() if len(signs) > 1 else 0,
        'sum_sing': signs.sum()
        }

    stats.update(count_in_ranges(scores))
    
    if graph:
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        
        axs[0, 0].hist(scores, bins=30, edgecolor='black')
        axs[0, 0].set_title('Розподіл значень scores')
        axs[0, 0].set_xlabel('Значення')
        axs[0, 0].set_ylabel('Частота')

        
        phi_scores = [phi(s) for s in scores]
        phi2_scores = [phi2(s) for s in scores]
        axs[0, 1].scatter(scores, phi_scores, label='phi', alpha=0.5)
        axs[0, 1].scatter(scores, phi2_scores, label='phi2', alpha=0.5)
        axs[0, 1].set_title('Залежність phi та phi2 від scores')
        axs[0, 1].set_xlabel('scores')
        axs[0, 1].set_ylabel('phi(scores) / phi2(scores)')
        axs[0, 1].legend()

        
        axs[1, 0].hist(phi_scores, bins=30, edgecolor='black')
        axs[1, 0].set_title('Розподіл phi(scores)')
        axs[1, 0].set_xlabel('phi(scores)')
        axs[1, 0].set_ylabel('Частота')

        
        axs[1, 1].hist(phi2_scores, bins=30, edgecolor='black')
        axs[1, 1].set_title('Розподіл phi2(scores)')
        axs[1, 1].set_xlabel('phi2(scores)')
        axs[1, 1].set_ylabel('Частота')

        plt.tight_layout()
        plt.show()

    return pd.DataFrame([stats])


def compare_texts_with_dates(texts, dates):
    all_stats = pd.DataFrame()

    
    total_comparisons = len(texts)*2  

    
    with tqdm(total=total_comparisons, desc="Порівняння текстів") as pbar:
        for i in range(0, len(texts), 3):  
            
            if i + 2 >= len(texts):
                break  

            id1 = i
            id2 = i+1
            id3 = i+2
            
            text1 = texts[id1]
            text2 = texts[id2]
            text3 = texts[id3]

            
            #1 Порівняння 1 з 2
            stats_df1 = process_texts(text1, text2)
            stats_df1['target'] = 1 if dates[id1] < dates[id2] else -1
            pbar.update(1)

            #2 Порівняння 2 з 1
            stats_df2 = process_texts(text2, text1)
            stats_df2['target'] = 1 if dates[id2] < dates[id1] else -1
            pbar.update(1)
            
            stats_df3 = process_texts(text2, text3)
            stats_df3['target'] = 1 if dates[id2] < dates[id3] else -1
            pbar.update(1)
            
            stats_df4 = process_texts(text3, text2)
            stats_df4['target'] = 1 if dates[id3] < dates[id2] else -1
            pbar.update(1)
            
            stats_df5 = process_texts(text1, text3)
            stats_df5['target'] = 1 if dates[id1] < dates[id3] else -1
            pbar.update(1)
            
            stats_df6 = process_texts(text3, text1)
            stats_df6['target'] = 1 if dates[id3] < dates[id1] else -1
            pbar.update(1)
            
            all_stats = pd.concat([all_stats, stats_df1, stats_df2,stats_df3, stats_df4,stats_df5, stats_df6], ignore_index=True)

    return all_stats

def get_N_texts(texts, dates, N=4):
    indices = random.sample(range(len(texts)), N)
    selected_texts, selected_dates = zip(*[(texts[i], dates[i]) for i in indices if texts[i] != ''])
    return selected_texts, selected_dates


class TemporalModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TemporalModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

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

with open(config['stopwords_file_path'], 'r', encoding='utf-8') as file:
    stopwords = set(line.strip() for line in file)

morph = pymorphy2.MorphAnalyzer(lang='uk')

if os.path.exists(config['text_file']) and os.path.exists(config['date_file']) and os.path.exists(config['vocab_file']):
    print("Loading preprocessed data from files...")
    
    with open(config['text_file'], 'rb') as f:
        encoded_texts = pickle.load(f)
    
    normalized_dates = np.load(config['date_file'], allow_pickle=True)
    
    with open(config['vocab_file'], 'r') as f:
        word_to_int = json.load(f)
        
    
else:
    messages, dates, texts = load_data(config['json_file_path'])

    cleaned_texts, cleaned_dates = preprocess_texts_batch(texts, dates, stopwords, morph)

    word_to_int = create_vocab(cleaned_texts, config['vocab_file'])
    int_to_word = list(word_to_int.keys())

    encode_and_save_texts(cleaned_texts, word_to_int, config['text_file'])
    nd = normalize_and_save_dates(cleaned_dates, config['date_file'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalModel(len(word_to_int), config['embedding_dim']).to(device)
model_file_path = config['load_model_file']

if os.path.exists(model_file_path):
    print(f"Loading model from {model_file_path}...")
    model.load_state_dict(torch.load(model_file_path, map_location=device))
    model.eval()
else:
    print("Initializing new model...")

if config['learning']:
    train_losses = train_model_optimized(model, config['epochs'], num_samples=config['num_samples'], batch_size=config['batch_size'], device=device)
    torch.save(model.state_dict(), "tmodel600epoch.pth")
    print(f"Model saved to tmodel.pth")


if config['learning']:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config['epochs'] + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()


with open(config['vocab_file'], 'r') as f:
    word_to_int = json.load(f)

df_temporal_indicators = random_temporal_directions_with_df(word_to_int)

plt.figure(figsize=(10, 6))
plt.hist(df_temporal_indicators['Indicator'], bins=100, color='skyblue', edgecolor='black')
plt.title('Histogram of Temporal Indicators for Random Word Pairs')
plt.xlabel('Temporal Indicator')
plt.ylabel('Frequency')
plt.show()

messages, dates, texts = load_data('UaOnlii.json')

selected_texts, selected_dates = get_N_texts(texts, dates, 10074)

data = compare_texts_with_dates(selected_texts, selected_dates)
data.to_csv('data.csv', encoding='utf-8', index=False)

data = pd.read_csv('27114.csv')

categorical_columns = ['text1', 'text2']
data = data.drop(categorical_columns, axis=1)


columns_to_scale = ['words1', 'len1', 'words2', 'len2', 'words1*2', 'words1+2', 'words1/2',
       'words2/1', 'len1*2', 'len1+2', 'len1/2', 'len2/1', 'Mean', 'mean_sign',
       'Std Dev', 'sm_scores', 'anti_sm_scores', 'max+min', 'max-min', 'Max',
       'Min', 'Sum', 'sign_mean', 'std_dev_sign', 'sum_sing', 'менше -1', '-1',
       '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1',
       '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1',
       'більше 1' ]

scaler = MinMaxScaler()

data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])


with open('scaler2.pkl', 'wb') as f:
    pickle.dump(scaler, f)


data2 = data[['words1', 'len1', 'words2', 'len2', 'words1*2', 'words1+2', 'words1/2', 'words2/1', 'len1*2', 'len1+2', 'len1/2', 'len2/1', 'Mean', 'mean_sign', 'Std Dev', 'sm_scores', 'anti_sm_scores', 'max+min', 'max-min', 'Max', 'Min', 'Sum', 'sign_mean', 'std_dev_sign', 'sum_sing', 'менше -1', '-1', '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1', 'більше 1', 'target']]



X = data2.drop(columns=['target']) 
y = data2['target'] 

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y, test_size=0.2, random_state=42)

gb_model = GradientBoostingClassifier(
    ccp_alpha=0.0,
    criterion='friedman_mse',
    init=None,
    learning_rate=0.1,
    loss='log_loss',
    max_depth=10,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=4,
    min_samples_split=10,
    min_weight_fraction_leaf=0.0,
    n_estimators=400,
    n_iter_no_change=None,
    random_state=None,
    subsample=1.0,
    tol=0.0001,
    validation_fraction=0.1,
    verbose=0,
    warm_start=False
)


gb_model.fit(X_train_clf, y_train_clf)

test_accuracy = gb_model.score(X_test_clf, y_test_clf)
print("Точність на тестових даних:", test_accuracy)


feature_importance = gb_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['feature'][:10], feature_importance_df['importance'][:10])
plt.xlabel('Ознаки')
plt.ylabel('Важливість')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

with open('gb_model_fixed.pkl', 'wb') as file:
    pickle.dump(gb_model, file)


plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Ознаки')
plt.ylabel('Важливість')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

news = ['example1','example2','example3','...', 'exampleN',]

text1 = news[0]
text2 = news[1]

t1t2comp = process_texts(text1, text2)
t1t2comp.columns


columns_to_scale = ['words1', 'len1', 'words2', 'len2', 'words1*2', 'words1+2', 'words1/2',
       'words2/1', 'len1*2', 'len1+2', 'len1/2', 'len2/1', 'Mean', 'mean_sign',
       'Std Dev', 'sm_scores', 'anti_sm_scores', 'max+min', 'max-min', 'Max',
       'Min', 'Sum', 'sign_mean', 'std_dev_sign', 'sum_sing', 'менше -1', '-1',
       '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1',
       '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1',
       'більше 1' ]

t1t2comp[columns_to_scale] = scaler.transform(t1t2comp[columns_to_scale])

prediction = gb_model.predict(data)
print(f"Prediction: {prediction}")


config2 = {
    'dir': '',
    'stopwords_file_path': 'stopwords_ua.txt',
    'vocab_file': 'vocab.json',
    'model_file': 'word-model.pth',
    'classifier_file': 'text-model.pkl',
    'embedding_dim': 100,
    'max_seq_length': 128
}

file = lambda name: f"{config2['dir']}{config2[name]}"

def classify_texts(text1, text2):

    config = {
        'stopwords_file_path': 'stopwords_ua.txt',
        'vocab_file': 'vocab.json',
        'model_file': 'word-model.pth',
        'classifier_file': 'text-model.pkl',
        'embedding_dim': 100,
        'max_seq_length': 128
    }
    
    with open(config['stopwords_file_path'], 'r', encoding='utf-8') as file:
        stopwords = set(line.strip() for line in file)
        

    with open(config['vocab_file'], 'r') as f:
        word_to_int = json.load(f)

    morph = pymorphy2.MorphAnalyzer(lang='uk')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
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

    def count_in_ranges(scores):
        
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        
        bins = [-float('inf'), -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, float('inf')]
        labels = ['менше -1', '-1', '-0.9', '-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0', 
                  '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1', 'більше 1']

        binned_scores = pd.cut(scores, bins=bins, labels=labels[:-1], right=False)
        count_dict = dict(Counter(binned_scores))
        count_in_order = {label: count_dict.get(label, 0) for label in labels}

        counts = np.array(list(count_in_order.values()))
        normalized_counts = softmax(counts)

        normalized_count_in_order = {label: float(value) for label, value in zip(labels, normalized_counts)}

        return normalized_count_in_order

    def filter_words(list_a):
        return list(filter(lambda x: x in list(word_to_int.keys()), list_a))
    
    
    def process_texts(text_a, text_b, graph=False):
        words_a = filter_words(clean_text(text_a, stopwords, morph).split())
        words_b = filter_words(clean_text(text_b, stopwords, morph).split())
        
        def phi(a):
            return a**4 if -1 <= a <= 1 else 1/(a**4)

        def phi2(a):
            return a**3 if -1 <= a <= 1 else 1/(a**3)

        def sm(A):
            A_list = A.tolist() if isinstance(A, pd.Series) else A
            return sum(phi(a) for a in A_list) / len(A_list) if A_list else 0

        def anti_sm(A):
            A_list = A.tolist() if isinstance(A, pd.Series) else A
            return sum(phi2(a) for a in A_list) / len(A_list) if A_list else 0
        
        evaluations = [
            [word1, word2, score := temporal_direction(word1, word2), 1 if score > 0 else -1]
            for word1, word2 in itertools.product(words_a, words_b)
        ]

        df = pd.DataFrame(evaluations, columns=['Слово1', 'Слово2', 'Оцінка', 'Знак'])
        scores, signs = df['Оцінка'], df['Знак']

        stats = {
            'text1': text_a,
            'words1': len(words_a),
            'len1': len(text_a),
            'text2': text_b,
            'words2': len(words_b),
            'len2': len(text_a),
            'words1*2': len(words_a)*len(words_b),
            'words1+2': len(words_a)+len(words_b),
            'words1/2': len(words_a)/len(words_b),
            'words2/1': len(words_b)/len(words_a),
            'len1*2': len(text_a)*len(text_b),
            'len1+2': len(text_a)+len(text_b),
            'len1/2': len(text_a)/len(text_b),
            'len2/1': len(text_b)/len(text_a),
            'Mean': scores.mean(),
            'mean_sign': 1 if scores.mean() > 0 else -1,
            'Std Dev': scores.std() if len(scores) > 1 else 0,
            'sm_scores': sm(scores),
            'anti_sm_scores': anti_sm(scores),
            'max+min': scores.max() + scores.min(),
            'max-min': scores.max() - scores.min(),
            'Max': scores.max(),
            'Min': scores.min(),
            'Sum': scores.sum(),
            'sign_mean': signs.mean(),
            'std_dev_sign': signs.std() if len(signs) > 1 else 0,
            'sum_sing': signs.sum()
            }

        stats.update(count_in_ranges(scores))

        if graph:
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))

            
            axs[0, 0].hist(scores, bins=30, edgecolor='black')
            axs[0, 0].set_title('Розподіл значень scores')
            axs[0, 0].set_xlabel('Значення')
            axs[0, 0].set_ylabel('Частота')

            
            phi_scores = [phi(s) for s in scores]
            phi2_scores = [phi2(s) for s in scores]
            axs[0, 1].scatter(scores, phi_scores, label='phi', alpha=0.5)
            axs[0, 1].scatter(scores, phi2_scores, label='phi2', alpha=0.5)
            axs[0, 1].set_title('Залежність phi та phi2 від scores')
            axs[0, 1].set_xlabel('scores')
            axs[0, 1].set_ylabel('phi(scores) / phi2(scores)')
            axs[0, 1].legend()

            
            axs[1, 0].hist(phi_scores, bins=30, edgecolor='black')
            axs[1, 0].set_title('Розподіл phi(scores)')
            axs[1, 0].set_xlabel('phi(scores)')
            axs[1, 0].set_ylabel('Частота')

            
            axs[1, 1].hist(phi2_scores, bins=30, edgecolor='black')
            axs[1, 1].set_title('Розподіл phi2(scores)')
            axs[1, 1].set_xlabel('phi2(scores)')
            axs[1, 1].set_ylabel('Частота')

            plt.tight_layout()
            plt.show()

        return pd.DataFrame([stats])
    
    def load_temporal_model(model_path, vocab_size, embedding_dim, device):
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

            
        model = TemporalModel(vocab_size, embedding_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.eval()
        return model
    
    temporal_model = load_temporal_model(config['model_file'], len(word_to_int), config['embedding_dim'], device)

    def temporal_direction(word1, word2):
        if word1 not in word_to_int or word2 not in word_to_int:
            print(f"One or both words '{word1}' and '{word2}' are not in the vocabulary.") 
            return 0 

        index1 = word_to_int[word1]
        index2 = word_to_int[word2]
        
        model = temporal_model

        model.eval()
        model.to(device)

        word1_tensor = torch.tensor([index1], dtype=torch.long, device=device)
        word2_tensor = torch.tensor([index2], dtype=torch.long, device=device)

        with torch.no_grad():
            embedding1 = model.embedding(word1_tensor).to(torch.float32)
            embedding2 = model.embedding(word2_tensor).to(torch.float32)

        input_tensor = torch.cat((embedding1, embedding2), dim=1)

        with torch.no_grad():
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
    
    data = process_texts(text1,text2,False)[['Mean', 'Std Dev', 'sign_mean', 'mean_sign',
       'sm_scores', 'anti_sm_scores', 'max+min', 'max-min', 'Min', 'Sum',
       'Max', 'sum_sing', 'std_dev_sign', 'менше -1', '-1', '-0.9', '-0.8',
       '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0', '0.1',
       '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1', 'більше 1','len1', 'len1*2', 'len1+2', 'len2',  
       'words1', 'words2', 'words1+2', 'words1*2', 'words1/2', 'words2/1']]
    
    selected_columns = ['Mean', 'Std Dev', 'sign_mean', 'mean_sign',
       'sm_scores', 'anti_sm_scores', 'max+min', 'max-min', 'Min', 'Sum',
       'Max', 'sum_sing', 'std_dev_sign', 'len1', 'len1*2', 'len1+2', 'len2', 
       'words1', 'words2', 'words1+2', 'words1*2', 'words1/2', 'words2/1']

    columns_to_scale = ['Mean', 'Std Dev', 'sign_mean', 'mean_sign', 'sm_scores', 'anti_sm_scores', 
                        'max+min', 'max-min', 'Min', 'Max', 'Sum','len1', 'len1*2', 'len1+2', 'len2', 'words1', 'words2', 'words1+2', 'words1*2']

    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    
    data[columns_to_scale] = scaler.transform(data[columns_to_scale])
    

    with open(config['classifier_file'], 'rb') as f:
        classifier = pickle.load(f)
    
    prediction = classifier.predict(data)
    
    ranishe2 =  prediction == 0
    zvit = f"Новина '{text2 if ranishe2 else text1}' опублікована раніше ніж '{text1 if ranishe2 else text2}'" if ranishe2 else f"Новина '{text1 if ranishe2 else text2}' опублікована пізніше ніж '{text2 if ranishe2 else text1}'"
    print(zvit)  

    return prediction[0]

news = ['example1','example2','example3','...',]

for n1 in range(len(news)):
    for n2 in range(len(news)):
        if n1 != n2:
            classify_texts(news[n1], news[n2])

res.columns