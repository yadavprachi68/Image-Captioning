import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Embedding, LSTM, Dense, add
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os

def load_inception_model():
    model = InceptionV3(weights='imagenet')
    model = Model(model.input, model.layers[-2].output)
    return model

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode_image(model, img_path):
    img = preprocess_image(img_path)
    feature_vector = model.predict(img)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

# Paths to your image and the pre-trained models
img_path = 'path_to_your_image.jpg'
tokenizer_path = 'tokenizer.pkl'
caption_model_path = 'caption_model.h5'

# Load the models and tokenizer
inception_model = load_inception_model()
tokenizer = pickle.load(open(tokenizer_path, 'rb'))
caption_model = tf.keras.models.load_model(caption_model_path)
max_length = 34

# Process the image and generate the caption
img_features = encode_image(inception_model, img_path)
caption = generate_caption(caption_model, tokenizer, img_features, max_length)
print('Generated Caption:', caption)
