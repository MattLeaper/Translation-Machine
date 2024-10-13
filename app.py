from flask import Flask, render_template, request, jsonify
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from views import views

app = Flask(__name__)
app.register_blueprint(views)


def translate_text(text, target_language, origin_language):
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    model_name = f"./Trainingcheckpoints/{target_language}"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    input_text = f"Translate {origin_language} to {target_language}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=40, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(origin_language)
    print(target_language)
    print("Hello")
    print(model_name)
    print(translation)
    return translation

def define_word(word):
    synsets = wordnet.synsets(word)
    definitions = [syn.definition() for syn in synsets]
    return definitions

@app.route('/')
def index():
    return render_template('site.html')

@app.route('/translate', methods=['POST'])
def translate_text_route():
    text_to_translate = request.form['text']
    target_language = request.form['target_language']
    origin_language = request.form['origin_language']
    
    translated_text = translate_text(text_to_translate, target_language, origin_language)

    return jsonify({'translated_text': translated_text})

@app.route('/define', methods=['POST'])
def define_route():
    word_to_define = request.form['text']
    definitions = define_word(word_to_define)
    return jsonify({'definitions': definitions})

if __name__ == '__main__':
    app.run(debug=True)