from transformers import BartTokenizer, BartForCausalLM
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForCausalLM.from_pretrained('facebook/bart-large')