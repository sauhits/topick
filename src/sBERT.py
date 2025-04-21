from sentence_transformers import SentenceTransformer,models
import os,torch
from torch.nn.functional import cosine_similarity

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device='cuda' if torch.cuda.is_available() else 'cpu'


def slide_window(data,size=2):
  if len(data) < size:
    return ["".join(data)]
  return ["".join(data[i:i+size]) for i in range(0,len(data)-size+1,1)]


test_data=[]
with open(f'/diskthalys/ssd8te/aobata/sentan/persona_text/00001.txt','r',encoding='utf-8') as in_f:
  # test_data=slide_window([s.strip() for s in in_f if s.strip()])
  test_data = [line.strip() for line in in_f if line.strip()]
  

# model_name='sonoisa/sentence-bert-base-ja-mean-tokens-v2'
# embedding_model=SentenceTransformer(model_name)

model_name='cl-tohoku/bert-base-japanese-whole-word-masking'
bert=models.Transformer(model_name)
pooling=models.Pooling(bert.get_word_embedding_dimension(),pooling_mode_max_tokens=True)
embedding_model=SentenceTransformer(modules=[bert,pooling],device=device)

embeddings=embedding_model.encode(test_data,device=device,convert_to_tensor=True)

similarities=cosine_similarity(embeddings[:-1],embeddings[1:])

threshold=0.85
topic_changes=[i for i, sim in enumerate(similarities) if sim < threshold]


print("遷移index: ",topic_changes)
for i, sim in enumerate(similarities):
    print(f"発話 {i} → 発話 {i+1}")
    print(f"  {test_data[i]}")
    print(f"  {test_data[i+1]}")
    print(f"  類似度: {sim:.4f}")
    print("-" * 50)
    
import matplotlib.pyplot as plt

# 類似度のヒストグラムを描画
plt.hist(similarities.cpu().numpy(), bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Distribution of Similarities")
plt.savefig('/diskthalys/ssd8te/aobata/sentan/sBERTplt.png')