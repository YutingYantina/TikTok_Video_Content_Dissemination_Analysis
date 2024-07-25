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
model_likes = joblib.load('model_likes_classifier.pkl')
model_comment = joblib.load('model_comment_classifier.pkl')
model_collect = joblib.load('model_collect_classifier.pkl')
model_share = joblib.load('model_share_classifier.pkl')
likes_bins = [0, 500, 1000, 5000, 10000, np.inf]
likes_labels = [0, 1, 2, 3, 4]
def map_label_to_range(label, bins):
    ranges = [(bins[i], bins[i+1]) for i in range(len(bins) - 1)]
    return ranges[label]
input_sentence = "中华本草，小故事"
input_embedding = get_bert_embeddings(input_sentence, tokenizer, model).reshape(1, -1)
likes_pred_label = model_likes.predict(input_embedding)[0]
comment_pred_label = model_comment.predict(input_embedding)[0]
collect_pred_label = model_collect.predict(input_embedding)[0]
share_pred_label = model_share.predict(input_embedding)[0]
likes_range = map_label_to_range(likes_pred_label, likes_bins)
comment_range = map_label_to_range(comment_pred_label, np.quantile(predict_merged_data['comment'], [0, 0.33, 0.66, 1]))
collect_range = map_label_to_range(collect_pred_label, np.quantile(predict_merged_data['collect'], [0, 0.33, 0.66, 1]))
share_range = map_label_to_range(share_pred_label, np.quantile(predict_merged_data['share'], [0, 0.33, 0.66, 1]))
print(f"Predicted Likes: {likes_range[0]} - {likes_range[1]}")
print(f"Predicted Comments: {comment_range[0]} - {comment_range[1]}")
print(f"Predicted Collects: {collect_range[0]} - {collect_range[1]}")
print(f"Predicted Shares: {share_range[0]} - {share_range[1]}")