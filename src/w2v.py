from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# サンプル文章データ
test_data=[]
with open(f'/diskthalys/ssd8te/aobata/sentan/persona_text/00001.txt','r',encoding='utf-8') as in_f:
  # test_data=slide_window([s.strip() for s in in_f if s.strip()])
  test_data = [line.strip() for line in in_f if line.strip()]
print(test_data)

# トークン化
tokenized_documents = [doc.split() for doc in test_data]

# Word2Vec モデルの学習
model = Word2Vec(sentences=tokenized_documents, vector_size=100, window=5, min_count=1, sg=0)

# TF-IDF の計算
vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None)
vectorizer.fit(tokenized_documents)
tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

# 話題語を抽出
for i, doc in enumerate(tokenized_documents):
    print(f"\n【Document {i+1}】")
    word_weights = {}
    
    for word in doc:
        if word in model.wv:  # Word2Vec に存在する単語のみ処理
            weight = tfidf_scores.get(word, 1.0)  # TF-IDF のスコア取得
            weighted_vector = model.wv[word] * weight  # 重み付きベクトル
            norm = np.linalg.norm(weighted_vector)  # ベクトルの大きさ（ノルム）を計算
            word_weights[word] = norm  # 話題語候補として登録

    # ノルムが大きい順にソート
    sorted_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)

    # 上位3単語を話題語とする
    topic_words = [word for word, norm in sorted_words[:3]]

    # 結果を表示
    for word, norm in sorted_words:
        print(f"単語: {word}, ベクトルノルム: {norm:.4f}")

    print(f"→ 話題語: {', '.join(topic_words)}")
