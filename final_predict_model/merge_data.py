import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
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
    return embeddings.numpy().flatten()
embeddings = []
for title in data['Title']:
    embedding = get_bert_embeddings(title, tokenizer, model)
    embeddings.append(embedding)
embeddings_df = pd.DataFrame(embeddings)
merged_data = pd.concat([data[['likes', 'comment', 'collect', 'share']], embeddings_df], axis=1)
merged_data.to_csv('./intern/predict_how_viral_each_video_is/cleaned_mergeddata.csv', index=False)