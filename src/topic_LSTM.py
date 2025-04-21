import fugashi
from collections import Counter
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer, util

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if lstm_out.dim() == 3:  # LSTMの出力が3次元なら、最後の時刻の出力を取得
            out = self.fc(lstm_out[:, -1, :])
        else:  # 出力が2次元の場合
            out = self.fc(lstm_out)
        return out


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

def compute_similarity_change(embeddings):
    diffs = [0]  # 初回は変化量なし
    for i in range(1, len(embeddings)):
        diff = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
        diffs.append(diff)
    return diffs

def train_lstm_model(train_data, input_dim, hidden_dim, output_dim, num_epochs=50, batch_size=8):
    model = LSTMPredictor(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(train_data[:-1], train_data[1:])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # y_batchの次元をoutputsに合わせる
            y_batch = y_batch[:, :output_dim]  # ターゲットの次元をoutput_dimに揃える
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model


def predict_next_embedding(model, input_sequence):
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device)
        predicted_embedding = model(input_sequence)
    return predicted_embedding.cpu().numpy()

# モデルのロード
model = SentenceTransformer("cl-tohoku/bert-base-japanese-whole-word-masking").to(device)

# データの読み込み
dialogues = load_persona_texts("/diskthalys/ssd8te/aobata/sentan/persona_text/00010.txt")

# 発話の埋め込み
embeddings = compute_embeddings(dialogues, model)

# 発話ごとのコサイン類似度変化量を計算
similarity_changes = compute_similarity_change(embeddings)

# LSTM 用データ作成
data = torch.tensor(np.hstack((embeddings.cpu().numpy(), np.array(similarity_changes).reshape(-1, 1))), dtype=torch.float32)

# LSTM モデルの学習
lstm_model = train_lstm_model(data, input_dim=data.shape[1], hidden_dim=128, output_dim=embeddings.shape[1])

# 指定した発話の次の発話を予測
n = 5  # 例: 5番目の発話の次を予測
input_sequence = data[n].unsqueeze(0)
predicted_embedding = predict_next_embedding(lstm_model, input_sequence)

# 予測した埋め込みに最も近い発話を検索
# 予測した埋め込みをGPUに転送
predicted_embedding = torch.tensor(predicted_embedding)  # numpy.ndarrayをtorch.Tensorに変換
predicted_embedding = predicted_embedding.to(device)  # deviceに転送

# embeddingsもGPUに転送（既にGPUにある場合は不要）
embeddings = embeddings.to(device)

# コサイン類似度を計算
cos_similarities = util.pytorch_cos_sim(predicted_embedding, embeddings).cpu().numpy()
predicted_index = np.argmax(cos_similarities)

print(f"{n}番目の発話の次の発話の予測: {dialogues[predicted_index]}")
