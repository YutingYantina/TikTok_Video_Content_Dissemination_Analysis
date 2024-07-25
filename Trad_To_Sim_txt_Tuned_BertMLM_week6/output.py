from transformers import TextGenerationPipeline, BertForMaskedLM, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_bert_model')
model = BertForMaskedLM.from_pretrained('./fine_tuned_bert_model')
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
new_herb_description = "人参的用途，其中包含，小故事"
generated_scripts = generator(new_herb_description, max_length=300, num_return_sequences=1)
for i, script in enumerate(generated_scripts):
    print(script['generated_text'])