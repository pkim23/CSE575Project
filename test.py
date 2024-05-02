import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('model/finalized_model.keras')

# Load the tokenizer
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Prepare your own data
texts = ["i am feeling great", "i feel jealous"]
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, maxlen=200)

# Use the model to predict the emotion
predictions = model.predict(x)

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)

# Define a reverse mapping dictionary
reverse_label_mapping = {0:'hate', 1:'neutral', 2:'anger', 3:'love', 4:'worry', 5:'relief', 6:'happiness',
       7:'fun', 8:'empty', 9:'enthusiasm', 10:'sadness', 11:'surprise', 12:'boredom'}

# Convert the predicted label numbers to their corresponding emotion text
predicted_emotions = [reverse_label_mapping[label] for label in predicted_labels]

# Print the predicted emotions
for text, emotion in zip(texts, predicted_emotions):
    print(f'Text: {text} --> Predicted Emotion: {emotion}')
