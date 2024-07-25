import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
file_path = '/content/sample_data/cleaned_mergeddata.csv'
merged_data = pd.read_csv(file_path)
X = merged_data.drop(columns=['likes', 'comment', 'collect', 'share'])
y_likes_log = np.log1p(merged_data['likes'])
y_comment_log = np.log1p(merged_data['comment'])
y_collect_log = np.log1p(merged_data['collect'])
y_share_log = np.log1p(merged_data['share'])
y_likes = pd.qcut(y_likes_log, q=5, labels=[0, 1, 2, 3, 4])
y_comment = pd.qcut(y_comment_log, q=5, labels=[0, 1, 2, 3, 4])
y_collect = pd.qcut(y_collect_log, q=3, labels=[0, 1, 2])
y_share = pd.qcut(y_share_log, q=3, labels=[0, 1, 2])
X_train, X_test, y_likes_train, y_likes_test = train_test_split(X, y_likes, test_size=0.2, random_state=42)
X_train, X_test, y_comment_train, y_comment_test = train_test_split(X, y_comment, test_size=0.2, random_state=42)
X_train, X_test, y_collect_train, y_collect_test = train_test_split(X, y_collect, test_size=0.2, random_state=42)
X_train, X_test, y_share_train, y_share_test = train_test_split(X, y_share, test_size=0.2, random_state=42)
model_likes = LogisticRegression(random_state=42, max_iter=10000).fit(X_train, y_likes_train)
model_comment = LogisticRegression(random_state=42, max_iter=10000).fit(X_train, y_comment_train)
model_collect = LogisticRegression(random_state=42, max_iter=10000).fit(X_train, y_collect_train)
model_share = LogisticRegression(random_state=42, max_iter=10000).fit(X_train, y_share_train)
joblib.dump(model_likes, 'model_likes_logistic.pkl')
joblib.dump(model_comment, 'model_comment_logistic.pkl')
joblib.dump(model_collect, 'model_collect_logistic.pkl')
joblib.dump(model_share, 'model_share_logistic.pkl')
y_likes_pred = model_likes.predict(X_test)
y_comment_pred = model_comment.predict(X_test)
y_collect_pred = model_collect.predict(X_test)
y_share_pred = model_share.predict(X_test)