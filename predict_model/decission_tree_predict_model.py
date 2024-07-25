import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
file_path = './intern/predict_how_viral_each_video_is/cleaned_mergeddata.csv'
merged_data = pd.read_csv(file_path)
X = merged_data.drop(columns=['likes', 'comment', 'collect', 'share'])
y_likes = merged_data['likes']
y_comment = merged_data['comment']
y_collect = merged_data['collect']
y_share = merged_data['share']

X_train, X_test, y_likes_train, y_likes_test = train_test_split(X, y_likes, test_size=0.2, random_state=42)
X_train, X_test, y_comment_train, y_comment_test = train_test_split(X, y_comment, test_size=0.2, random_state=42)
X_train, X_test, y_collect_train, y_collect_test = train_test_split(X, y_collect, test_size=0.2, random_state=42)
X_train, X_test, y_share_train, y_share_test = train_test_split(X, y_share, test_size=0.2, random_state=42)

model_likes = DecisionTreeRegressor(random_state=42).fit(X_train, y_likes_train)
model_comment = DecisionTreeRegressor(random_state=42).fit(X_train, y_comment_train)
model_collect = DecisionTreeRegressor(random_state=42).fit(X_train, y_collect_train)
model_share = DecisionTreeRegressor(random_state=42).fit(X_train, y_share_train)

joblib.dump(model_likes, 'model_likes_decision_tree.pkl')
joblib.dump(model_comment, 'model_comment_decision_tree.pkl')
joblib.dump(model_collect, 'model_collect_decision_tree.pkl')
joblib.dump(model_share, 'model_share_decision_tree.pkl')

y_likes_pred = model_likes.predict(X_test)
y_comment_pred = model_comment.predict(X_test)
y_collect_pred = model_collect.predict(X_test)
y_share_pred = model_share.predict(X_test)

print("Likes - Mean Squared Error:", mean_squared_error(y_likes_test, y_likes_pred))
print("Comments - Mean Squared Error:", mean_squared_error(y_comment_test, y_comment_pred))
print("Collects - Mean Squared Error:", mean_squared_error(y_collect_test, y_collect_pred))
print("Shares - Mean Squared Error:", mean_squared_error(y_share_test, y_share_pred))
