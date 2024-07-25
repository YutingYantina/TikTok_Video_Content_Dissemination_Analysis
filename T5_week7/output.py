import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_text = "风血痢”，书中还记载了关于桂花的外用法，谓：“凡阴寒冷气，瘕疝奔豚，腹内一切冷痛，蒸热布裹熨之。”《陆川本草》称桂花“治痰饮喘咳”"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
outputs = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)