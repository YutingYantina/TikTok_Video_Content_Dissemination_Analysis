import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import joblib
import numpy as np
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()
model_likes = joblib.load('model_likes_logistic.pkl')
model_comment = joblib.load('model_comment_logistic.pkl')
model_collect = joblib.load('model_collect_logistic.pkl')
model_share = joblib.load('model_share_logistic.pkl')
input_sentence = "你好中草药"
input_embedding = get_bert_embeddings(input_sentence, tokenizer, model).reshape(1, -1)
predicted_likes_label = model_likes.predict(input_embedding)[0]
predicted_comment_label = model_comment.predict(input_embedding)[0]
predicted_collect_label = model_collect.predict(input_embedding)[0]
predicted_share_label = model_share.predict(input_embedding)[0]
likes_bins = pd.qcut(np.log1p(merged_data['likes']), q=5, retbins=True)[1]
comment_bins = pd.qcut(np.log1p(merged_data['comment']), q=5, retbins=True)[1]
collect_bins = pd.qcut(np.log1p(merged_data['collect']), q=3, retbins=True)[1]
share_bins = pd.qcut(np.log1p(merged_data['share']), q=3, retbins=True)[1]
def map_label_to_range(label, bins):
    ranges = [(bins[i], bins[i+1]) for i in range(len(bins) - 1)]
    return ranges[label]
predicted_likes_range = map_label_to_range(predicted_likes_label, likes_bins)
predicted_comment_range = map_label_to_range(predicted_comment_label, comment_bins)
predicted_collect_range = map_label_to_range(predicted_collect_label, collect_bins)
predicted_share_range = map_label_to_range(predicted_share_label, share_bins)
print(f"Predicted Likes: {predicted_likes_range[0]:.2f} - {predicted_likes_range[1]:.2f}")
print(f"Predicted Comments: {predicted_comment_range[0]:.2f} - {predicted_comment_range[1]:.2f}")
print(f"Predicted Collects: {predicted_collect_range[0]:.2f} - {predicted_collect_range[1]:.2f}")
print(f"Predicted Shares: {predicted_share_range[0]:.2f} - {predicted_share_range[1]:.2f}")