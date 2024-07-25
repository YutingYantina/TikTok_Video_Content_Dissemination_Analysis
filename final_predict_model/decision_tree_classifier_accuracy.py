import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import joblib
from sklearn.metrics import accuracy_score
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
model_likes = joblib.load('model_likes_classifier.pkl')
model_comment = joblib.load('model_comment_classifier.pkl')
model_collect = joblib.load('model_collect_classifier.pkl')
model_share = joblib.load('model_share_classifier.pkl')
#likes_bins = [0, 500, 1000, 5000, 10000, np.inf]
#likes_labels = [0, 1, 2, 3, 4]
#y_likes_true = pd.cut(predict_merged_data['likes'], bins=likes_bins, labels=likes_labels)
X_predict = predict_merged_data.drop(columns=['likes', 'comment', 'collect', 'share'])
y_likes_true = pd.qcut(predict_merged_data['likes'], q=3, labels=[0, 1, 2])
y_comment_true = pd.qcut(predict_merged_data['comment'], q=3, labels=[0, 1, 2])
y_collect_true = pd.qcut(predict_merged_data['collect'], q=3, labels=[0, 1, 2])
y_share_true = pd.qcut(predict_merged_data['share'], q=3, labels=[0, 1, 2])
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