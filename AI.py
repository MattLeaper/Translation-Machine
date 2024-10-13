from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

source_language = input("Enter the source language: ")
target_language = input("Enter the target language: ")
line = input("Enter the text you want to translate: ")

text = f"translate {source_language} to {target_language}: {line}"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt")

# Load fine-tuned model


model_name = f"./Trainingcheckpoints/{target_language}"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Generate translation
outputs = model.generate(inputs.input_ids, max_length=40, num_beams=4, early_stopping=True)

# Decode the output tokens to text
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

###### prints all information#####
print(source_language)
print(target_language)
print(text)
print("Translation:", translation)