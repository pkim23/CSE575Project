from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the model
model = load_model('model/finalized_model.keras')

# Load the tokenizer
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define a reverse mapping dictionary
reverse_label_mapping = {0:'hate', 1:'neutral', 2:'anger', 3:'love', 4:'worry', 5:'relief', 6:'happiness',
       7:'fun', 8:'empty', 9:'enthusiasm', 10:'sadness', 11:'surprise', 12:'boredom'}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        sequences = tokenizer.texts_to_sequences([text])
        x = pad_sequences(sequences, maxlen=200)
        predictions = model.predict(x)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_emotion = reverse_label_mapping[predicted_labels[0]]
        return render_template('index.html', emotion=predicted_emotion)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)


