import fugashi
from collections import Counter
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer, util

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tokenize_nouns(text):
    tagger = fugashi.Tagger()
    nouns = [word.surface for word in tagger(text) if '名詞' in word.feature]
    return nouns

def load_persona_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as in_f:
        return [line.strip() for line in in_f if line.strip()]

def load_stopwords(filename="stopwords.txt"):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def compute_embeddings(dialogues, model):
    return model.encode(dialogues, convert_to_tensor=True, device=device)

def compute_full_similarity_matrix(embeddings):
    return util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

def identify_topic_shifts(dialogues, embeddings, threshold=0.8):
    topic_windows = []
    current_window = [dialogues[0]]
    window_start_embedding = embeddings[0]

    for i in range(1, len(dialogues) - 1):
        sim_n_n1 = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()  # 直前の発話との類似度
        sim_n_n2 = util.pytorch_cos_sim(embeddings[i-1], embeddings[i+1]).item()  # 次の発話との類似度
        sim_n_start = util.pytorch_cos_sim(window_start_embedding, embeddings[i]).item()  # 最初の発話との類似度

        # 話題転換の判定
        if (sim_n_n1 < threshold and (i == len(dialogues) - 2 or sim_n_n2 < threshold)) or sim_n_start < threshold:
            topic_windows.append(current_window)
            current_window = []
            window_start_embedding = embeddings[i]

        current_window.append(dialogues[i])

    if current_window:
        topic_windows.append(current_window)

    return topic_windows


def extract_topics_from_windows(topic_windows, stopwords, top_n=5):
    topic_words = []
    for window in topic_windows:
        word_counts = Counter()
        for dialogue in window: 
            nouns = tokenize_nouns(dialogue)
            filtered_nouns = [noun for noun in nouns if noun not in stopwords]
            word_counts.update(filtered_nouns)
        topics = [word for word, _ in word_counts.most_common(top_n)]
        topic_words.append((topics, window))
    return topic_words

# モデルのロード
model = SentenceTransformer("cl-tohoku/bert-base-japanese-whole-word-masking").to(device)

# データの読み込み
dialogues = load_persona_texts("/diskthalys/ssd8te/aobata/sentan/persona_text/00003.txt")
stopwords = load_stopwords("/diskthalys/ssd8te/aobata/sentan/data/stopwords.txt")

# 発話の埋め込み
embeddings = compute_embeddings(dialogues, model)

# 話題転換の識別
topic_windows = identify_topic_shifts(dialogues, embeddings)

# 話題語の抽出
topics_per_window = extract_topics_from_windows(topic_windows, stopwords)

# 全発話間のコサイン類似度行列を計算
full_similarity_matrix = compute_full_similarity_matrix(embeddings)

# 各ウィンドウごとの話題語と発話の表示
print("各ウィンドウごとの話題語と発話:")
for i, (topics, window) in enumerate(topics_per_window):
    print(f"\n--- ウィンドウ {i+1} ---")
    print(f"話題語: {topics}")
    print("対話内容:")
    for idx, dialogue in enumerate(window): 
        print(f"  {idx+1}. {dialogue}")

# コサイン類似度行列の出力
# print("\n全発話間のコサイン類似度行列:")
# np.set_printoptions(precision=2, suppress=True, linewidth=200)
# print(full_similarity_matrix)
