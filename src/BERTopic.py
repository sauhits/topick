from bertopic import BERTopic
from sentence_transformers import SentenceTransformer,models
from fugashi import Tagger
from sklearn.feature_extraction.text import CountVectorizer
import torch
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device='cuda' if torch.cuda.is_available() else 'cpu'

def fugashi_tokenizer(text):
  tagger=Tagger()
  # return [mrph.surface for mrph in tagger(text) if mrph.surface not in stop_words]
  target={"動詞","名詞","形容詞"}
  tokens=[]
  for mrph in tagger(text):
    pos =mrph.feature.pos1
    if pos in target:
      token =mrph.feature.lemma if pos=="動詞" else mrph.surface
      if token not in stop_words:
        tokens.append(token)
  return tokens
      
      
# def mecab_tokenizer(text):
#   tagger=mecab.Tagger(r"-Owakati -d /content/drive/MyDrive/Colab Notebooks/Demo/pylibs/mecab-ipadic-neologd")
#   return [word for word in tagger.parse(text).strip().split() if word not in stop_words]

def slide_window(data,size=3):
  if len(data) < size:
    return ["".join(data)]
  return ["".join(data[i:i+size]) for i in range(0,len(data)-size+1,1)]

# テストデータの取得
test_data=[]
with open(f'/diskthalys/ssd8te/aobata/sentan/nuccre/re_after120.txt','r',encoding='utf-8') as in_f:
  test_data=slide_window([s.strip() for s in in_f if s.strip()])

# ストップワードの取得
stop_words=[]
with open(f'/diskthalys/ssd8te/aobata/sentan/stopwords_df.txt','r',encoding='utf-8') as in_f:
  stop_words=set(in_f.read().splitlines())

# モデルの指定
model_name='cl-tohoku/bert-base-japanese-whole-word-masking'
bert=models.Transformer(model_name)
pooling=models.Pooling(bert.get_word_embedding_dimension())
embedding_model=SentenceTransformer(modules=[bert,pooling],device=device)

# model_name='sonoisa/sentence-bert-base-ja-mean-tokens-v2'
# embedding_model=SentenceTransformer(model_name)

vectorizer_model = CountVectorizer(tokenizer=fugashi_tokenizer,ngram_range=(1,1))
# vectorizer_model = CountVectorizer(analyzer="word",tokenizer=mecab_tokenizer,ngram_range=(1,1))

# ベクトルに変換
embeddings=embedding_model.encode(test_data,device=device,batch_size=16)

# BERTopicのモデル指定
model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    verbose=True
    )

# 文埋め込みよりトピックの抽出
topics, probs = model.fit_transform(test_data,embeddings)

# model.reduce_topics(test_data, nr_topics=50)

print(model.get_topic_info())

topic_words = {topic: model.get_topic(topic) for topic in set(topics)}
for topic, words in topic_words.items():
    word_list = [word[0] for word in words if word[0] != ""]
    print(f"Topic {topic}: {', '.join(word_list)}")