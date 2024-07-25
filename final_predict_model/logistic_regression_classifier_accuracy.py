import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import joblib
from sklearn.metrics import accuracy_score
import numpy as np
predict_file_path = '/content/sample_data/predictalldata.csv'
predict_data = pd.read_csv(predict_file_path)
predict_data['Title'] = predict_data['Title'].astype(str)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()
predict_embeddings = []
for title in predict_data['Title']:
    embedding = get_bert_embeddings(title, tokenizer, model)
    predict_embeddings.append(embedding)
predict_embeddings_df = pd.DataFrame(predict_embeddings)
predict_merged_data = pd.concat([predict_data[['likes', 'comment', 'collect', 'share']], predict_embeddings_df], axis=1)
y_likes_true_log = np.log1p(predict_merged_data['likes'])
y_comment_true_log = np.log1p(predict_merged_data['comment'])
y_collect_true_log = np.log1p(predict_merged_data['collect'])
y_share_true_log = np.log1p(predict_merged_data['share'])
y_likes_true = pd.qcut(y_likes_true_log, q=5, labels=[0, 1, 2, 3, 4])
y_comment_true = pd.qcut(y_comment_true_log, q=5, labels=[0, 1, 2, 3, 4])
y_collect_true = pd.qcut(y_collect_true_log, q=3, labels=[0, 1, 2])
y_share_true = pd.qcut(y_share_true_log, q=3, labels=[0, 1, 2])
X_predict = predict_merged_data.drop(columns=['likes', 'comment', 'collect', 'share'])
model_likes = joblib.load('model_likes_logistic.pkl')
model_comment = joblib.load('model_comment_logistic.pkl')
model_collect = joblib.load('model_collect_logistic.pkl')
model_share = joblib.load('model_share_logistic.pkl')
y_likes_pred = model_likes.predict(X_predict)
y_comment_pred = model_comment.predict(X_predict)
y_collect_pred = model_collect.predict(X_predict)
y_share_pred = model_share.predict(X_predict)
accuracy_likes = accuracy_score(y_likes_true, y_likes_pred)
accuracy_comment = accuracy_score(y_comment_true, y_comment_pred)
accuracy_collect = accuracy_score(y_collect_true, y_collect_pred)
accuracy_share = accuracy_score(y_share_true, y_share_pred)
print("Likes - Accuracy Score:", accuracy_likes)
print("Comments - Accuracy Score:", accuracy_comment)
print("Collects - Accuracy Score:", accuracy_collect)
print("Shares - Accuracy Score:", accuracy_share)