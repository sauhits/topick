import torch
import fugashi
from transformers import BertTokenizerFast, BertModel

# モデルとトークナイザーのロード
MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True)

# 形態素解析器（fugashi）
tagger = fugashi.Tagger("-Owakati")

def get_morphemes(text):
    """fugashiで形態素解析し、単語リストを取得"""
    return tagger.parse(text).strip().split()

def get_attention_scores(text):
    """BERTのAttentionスコアを単語単位で取得"""
    
    # 事前に形態素解析して単語を取得
    words = get_morphemes(text)

    # BERTトークナイザーでサブワード分割
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # 最終レイヤーのAttention取得（形状: (batch, num_heads, seq_len, seq_len)）
    attentions = outputs.attentions[-1]
    
    # ヘッド平均をとる
    attention_weights = attentions.mean(dim=1).squeeze(0).cpu().numpy()
    
    # 自己Attentionスコア（対角成分）
    token_attention = attention_weights.diagonal()

    # トークンIDからサブワードリスト取得
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    # サブワードをfugashiの単語単位に統合
    word_attentions = {}
    token_idx = 0  # トークンリストのインデックス
    
    for word in words:
        score_sum = 0
        count = 0
        
        while token_idx < len(tokens):
            token = tokens[token_idx]
            score_sum += token_attention[token_idx]
            count += 1
            token_idx += 1
            
            # fugashiの単語と一致するタイミングで確定
            if word.endswith(token.replace("##", "")):
                break
        
        # 平均スコアで記録
        word_attentions[word] = score_sum / count if count > 0 else 0
    
    return word_attentions

# テスト用の文
text = "システム自動動く"
word_scores = get_attention_scores(text)

# スコアの高い順に話題語を取得
sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
# print("話題語候補:", sorted_words[:3])  # 上位3つを表示
print(word_scores)
