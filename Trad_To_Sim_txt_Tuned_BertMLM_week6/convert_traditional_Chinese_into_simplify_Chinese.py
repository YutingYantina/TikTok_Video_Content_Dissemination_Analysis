import os
import opencc
converter = opencc.OpenCC('t2s.json') 
def read_and_convert_scripts(directory):
    scripts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                simplified_text = converter.convert(text) 
                scripts.append(simplified_text)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(simplified_text)
    return scripts
scripts = read_and_convert_scripts('/content/txt')