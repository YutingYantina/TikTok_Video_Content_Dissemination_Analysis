from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].flatten()
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) * (input_ids != 101) * (input_ids != 102) * (input_ids != 0)  # [CLS], [SEP], [PAD]
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        input_ids[selection] = 103 
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }
train_texts, val_texts = train_test_split(scripts, test_size=0.1)
train_dataset = CustomDataset(train_texts, tokenizer, max_len=128)
val_dataset = CustomDataset(val_texts, tokenizer, max_len=128)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4)