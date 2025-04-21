import fugashi
import os
import torch
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_cosine_similarity(embeddings):
    return cosine_similarity(embeddings)

def get_embeddings(documents):
    embeddings = sbert_model.encode(documents, convert_to_tensor=True)
    return embeddings


tagger = fugashi.Tagger()
documents = []
with open(f'/diskthalys/ssd8te/aobata/sentan/persona_text/00001.txt', 'r', encoding='utf-8') as in_f:
    documents = [line.strip() for line in in_f if line.strip()]


processed_documents = [[word.surface for word in tagger(line)] for line in documents]
model = Word2Vec.load("/diskthalys/ssd8te/aobata/sentan/word2vec_nuccre.model")
sbert_model = SentenceTransformer('cl-tohoku/bert-base-japanese-whole-word-masking', device='cuda')

# 文書の埋め込み
embeddings = get_embeddings(processed_documents)

# コサイン類似度
similarity_matrix = calculate_cosine_similarity(embeddings.cpu().numpy())
import pandas as pd

# DataFrame 形式で表示
pd.set_option('display.float_format', '{:.3f}'.format)
df = pd.DataFrame(similarity_matrix, columns=[f"Doc {i}" for i in range(len(similarity_matrix))])
# print(df)

# 話題継続の判断
threshold = 0.81
topic_continuations = []
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):
        if similarity_matrix[i][j] > threshold:
            topic_continuations.append((i, j))


import csv

# ノード情報をCSVに保存
with open('/diskthalys/ssd8te/aobata/sentan/data/nodes.csv', mode='w', newline='', encoding='utf-8') as node_file:
    writer = csv.writer(node_file)
    writer.writerow(['NodeID', 'Document'])  # ヘッダー
    for i in range(len(documents)):
        writer.writerow([f"Doc {i}", documents[i]])

# エッジ情報をCSVに保存
with open('/diskthalys/ssd8te/aobata/sentan/data/edges.csv', mode='w', newline='', encoding='utf-8') as edge_file:
    writer = csv.writer(edge_file)
    writer.writerow(['Source', 'Target', 'CosineSimilarity'])  # ヘッダー
    for pair in topic_continuations:
        writer.writerow([f"Doc {pair[0]}", f"Doc {pair[1]}", similarity_matrix[pair[0]][pair[1]]])
