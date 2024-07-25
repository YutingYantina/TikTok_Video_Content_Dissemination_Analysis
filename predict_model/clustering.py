import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
from collections import Counter
file_path = './intern/predict_how_viral_each_video_is/cleaned_alldata.csv'
data = pd.read_csv(file_path)
data['Title'] = data['Title'].astype(str)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings
embeddings = []
for title in data['Title']:
    embedding = get_bert_embeddings(title, tokenizer, model)
    embeddings.append(embedding.numpy())
embeddings_df = pd.DataFrame([embedding.flatten() for embedding in embeddings])
num_clusters = 50
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(embeddings_df)
cluster_keywords = []
cluster_likes = []

for i in range(num_clusters):
    cluster_titles = data[data['cluster'] == i]['Title']
    cluster_likes.append(data[data['cluster'] == i]['likes'].mean())
    all_words = ' '.join(cluster_titles).split()
    most_common_words = Counter(all_words).most_common(5)
    cluster_keywords.append(most_common_words)
for i in range(num_clusters):
    print(f"Cluster {i}:")
    print("Most common words:", cluster_keywords[i])
    print("Average likes:", cluster_likes[i])
    print("\n")