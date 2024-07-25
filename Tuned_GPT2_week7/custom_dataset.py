def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask'])
from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['labels'] = item['input_ids'].clone()  # 将标签设置为输入的克隆
        return item
train_dataset = CustomDataset(tokenized_datasets['train'])
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)