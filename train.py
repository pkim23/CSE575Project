import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, BatchNormalization, Dropout, Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load data
data = pd.read_csv('dataset/emotion_sentimen_dataset.csv')

data['Emotion'].value_counts()

plt.figure(figsize=(12, 8))
sns.countplot(x='Emotion', data=data)
plt.title('Emotion Visualization')
plt.show()

def clean_data(text):
    # Convert all letters to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    return text


# Clean the text data
data['text'] = data['text'].apply(clean_data)

print(data)

# Define a mapping dictionary
label_mapping = {'hate':0, 'neutral':1, 'anger':2, 'love':3, 'worry':4, 'relief':5, 'happiness':6,
       'fun':7, 'empty':8, 'enthusiasm':9, 'sadness':9, 'surprise':10, 'boredom':11}

# Rename the 'Label' column using the mapping dictionary
data['Emotion'] = data['Emotion'].map(label_mapping)

# Prepare text data
tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
maxlen = max(len(tokens) for tokens in sequences)
x = pad_sequences(sequences, maxlen=maxlen, padding='post')

# Prepare target data
encoder = LabelEncoder()
encoder.fit(data['Emotion'])
encoded_y = encoder.transform(data['Emotion'])
y = to_categorical(encoded_y)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

input_size = np.max(x_train) + 1

# Define model
model = Sequential()
model.add(Embedding(input_dim=input_size, output_dim=100,input_shape=(69,)))
model.add(Bidirectional(LSTM(128)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

best_epoch = np.argmin(history.history['val_loss'])
print(f'Best epoch: {best_epoch + 1}')


import os

# Create the 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the model
model.save('model/finalized_model.h5')

# Save the tokenizer and the encoder
with open('model/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model/encoder.pkl', 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("The model, tokenizer, and encoder are successfully saved to the disk.")

# Load the model, tokenizer, and encoder from disk
with open('model/finalized_model.h5', 'rb') as f:
    model = pickle.load(f)

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Prepare your own data
texts = ["i am feeling great", "i feel jealous"]
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, maxlen=200)

# Use the model to predict the emotion
predictions = model.predict(x)

# Convert predictions to labels
predicted_labels = encoder.inverse_transform(np.argmax(predictions, axis=1))

# Print the predicted labels
for text, label in zip(texts, predicted_labels):
    print(f'Text: {text} --> Predicted Emotion: {label}')