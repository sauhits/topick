import fugashi
from collections import Counter
import numpy as np
import os

def tokenize_nouns(text):
    tagger = fugashi.Tagger()
    nouns = [word.surface for word in tagger(text) if '名詞' in word.feature]
    return nouns

def compute_document_frequency(documents):
    doc_count = len(documents)
    word_doc_counts = Counter()
    for doc in documents:
        unique_nouns = set(tokenize_nouns(doc))
        word_doc_counts.update(unique_nouns)
    df = {word: count / doc_count for word, count in word_doc_counts.items()}
    return df

def load_sentan_texts(directory, file_limit=100):
    files = sorted(os.listdir(directory))[:file_limit]
    texts = []
    for file in files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

def save_stopwords(stopwords, filename="stopwords.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        for word in stopwords:
            f.write(word + "\n")

dialogues = load_sentan_texts("/diskthalys/ssd8te/aobata/sentan/nuccre")


df_values = compute_document_frequency(dialogues)
threshold = 0.89
stopwords = [word for word, df in df_values.items() if df >= threshold]

print("ストップワード一覧:", stopwords)
save_stopwords(stopwords,"/diskthalys/ssd8te/aobata/sentan/data/stopwords.txt")