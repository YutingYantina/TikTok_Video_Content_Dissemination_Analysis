from transformers import BertTokenizer, BertModel
import torch
import joblib
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()
new_title = "男人为了把控药材质量以身试药，却被刚入行的学徒一眼看出端倪！#国产动画 #怀旧动画 #中医动画 #老动画 #动画"#predicted Title
new_title_embedding = get_bert_embeddings(new_title, tokenizer, model).reshape(1, -1)
model_likes = joblib.load('/Users/fengliyinghua/Desktop/intern/predict_how_viral_each_video_is/model_likes_decision_tree.pkl')
model_comment = joblib.load('/Users/fengliyinghua/Desktop/intern/predict_how_viral_each_video_is/model_comment_decision_tree.pkl')
model_collect = joblib.load('/Users/fengliyinghua/Desktop/intern/predict_how_viral_each_video_is/model_collect_decision_tree.pkl')
model_share = joblib.load('/Users/fengliyinghua/Desktop/intern/predict_how_viral_each_video_is/model_share_decision_tree.pkl')
predicted_likes = model_likes.predict(new_title_embedding)
predicted_comment = model_comment.predict(new_title_embedding)
predicted_collect = model_collect.predict(new_title_embedding)
predicted_share = model_share.predict(new_title_embedding)
print(f"Predicted Likes: {predicted_likes[0]}")
print(f"Predicted Comments: {predicted_comment[0]}")
print(f"Predicted Collects: {predicted_collect[0]}")
print(f"Predicted Shares: {predicted_share[0]}")