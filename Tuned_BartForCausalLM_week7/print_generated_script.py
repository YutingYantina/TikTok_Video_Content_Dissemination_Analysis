from transformers import TextGenerationPipeline, BartForCausalLM, BartTokenizer
tokenizer = BartTokenizer.from_pretrained('./fine_tuned_bart_model')
model = BartForCausalLM.from_pretrained('./fine_tuned_bart_model')
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
new_herb_description = "可见子苏的价值非常好"
generated_scripts = generator(new_herb_description, max_length=300, num_return_sequences=1)
for i, script in enumerate(generated_scripts):
    print(script['generated_text'])