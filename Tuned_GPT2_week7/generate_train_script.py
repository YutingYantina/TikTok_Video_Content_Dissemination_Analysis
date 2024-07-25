import os
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer
def read_scripts(directory):
    scripts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                scripts.append(file.read())
    return scripts
scripts = read_scripts('/content/sample_data')
combined_script = "\n\n".join(scripts)
with open('/content/combined_script.txt', 'w', encoding='utf-8') as f:
    f.write(combined_script)
dataset = load_dataset('text', data_files={'train': '/content/combined_script.txt'})