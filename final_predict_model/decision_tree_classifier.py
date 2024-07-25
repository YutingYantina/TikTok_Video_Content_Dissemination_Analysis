import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
file_path = '/content/sample_data/cleaned_mergeddata.csv'
merged_data = pd.read_csv(file_path)
X = merged_data.drop(columns=['likes', 'comment', 'collect', 'share'])
#likes_bins = [0, 500, 1000, 5000, 10000, np.inf]
#likes_labels = [0, 1, 2, 3, 4]
#y_likes = pd.cut(merged_data['likes'],  bins=likes_bins, labels=likes_labels)
y_likes = pd.qcut(merged_data['likes'], q=3, labels=[0, 1, 2])
y_comment = pd.qcut(merged_data['comment'], q=3, labels=[0, 1, 2])
y_collect = pd.qcut(merged_data['collect'], q=3, labels=[0, 1, 2])
y_share = pd.qcut(merged_data['share'], q=3, labels=[0, 1, 2])
X_train, X_test, y_likes_train, y_likes_test = train_test_split(X, y_likes, test_size=0.2, random_state=42)
X_train, X_test, y_comment_train, y_comment_test = train_test_split(X, y_comment, test_size=0.2, random_state=42)
X_train, X_test, y_collect_train, y_collect_test = train_test_split(X, y_collect, test_size=0.2, random_state=42)
X_train, X_test, y_share_train, y_share_test = train_test_split(X, y_share, test_size=0.2, random_state=42)
model_likes = DecisionTreeClassifier(random_state=42).fit(X_train, y_likes_train)
model_comment = DecisionTreeClassifier(random_state=42).fit(X_train, y_comment_train)
model_collect = DecisionTreeClassifier(random_state=42).fit(X_train, y_collect_train)
model_share = DecisionTreeClassifier(random_state=42).fit(X_train, y_share_train)
joblib.dump(model_likes, 'model_likes_classifier.pkl')
joblib.dump(model_comment, 'model_comment_classifier.pkl')
joblib.dump(model_collect, 'model_collect_classifier.pkl')
joblib.dump(model_share, 'model_share_classifier.pkl')