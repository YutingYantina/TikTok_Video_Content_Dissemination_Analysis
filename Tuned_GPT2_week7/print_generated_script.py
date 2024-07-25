from transformers import TextGenerationPipeline, GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('/content/fine_tuned_gpt2_model')
model = GPT2LMHeadModel.from_pretrained('/content/fine_tuned_gpt2_model')
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
new_herb_description = "黄金"
generated_scripts = generator(new_herb_description, max_length=300, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95)
for i, script in enumerate(generated_scripts):
    print(script['generated_text'])