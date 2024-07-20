import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import time

# 1. Load the Dataset
def load_dataset():
    data, info = tfds.load('coco/2014', with_info=True)
    train_data = data['train']
    val_data = data['validation']
    return train_data, val_data, info

# 2. Preprocess Images and Captions
def preprocess_image(image):
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

def preprocess_caption(caption):
    caption = caption.numpy().decode('utf-8')
    return caption

def map_func(image, caption):
    image = preprocess_image(image)
    caption = preprocess_caption(caption)
    return image, caption

# 3. Load Pre-trained Model (Encoder-Decoder with Attention)
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Define max_length and attention_features_shape
max_length = 52  # You need to set this based on your dataset
attention_features_shape = 64  # Typically the flattened size of the feature map (8x8=64)


# Load tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("Tokenizer file not found. Please ensure 'tokenizer.pickle' is in the same directory as the script.")
    exit(1)

# Load pre-trained model
try:
    model = tf.keras.models.load_model('image_captioning_model.h5')
except FileNotFoundError:
    print("Pre-trained model file not found. Please ensure 'image_captioning_model.h5' is in the same directory as the script.")
    exit(1)

# 4. Generate Caption
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = model.reset_states()
    temp_input = tf.expand_dims(preprocess_image(image), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = model(dec_input, img_tensor_val, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)

        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

# 5. Main Function
def main(image_path):
    result, attention_plot = evaluate(image_path)
    result = ' '.join(result[:-1])
    print('Prediction Caption:', result)
    plot_attention(image_path, result, attention_plot)

if __name__ == "__main__":
    train_data, val_data, info = load_dataset()
    # Select an image from validation data
    for img, cap in val_data.take(1):
        image_path = 'example.jpg'  # Update with the actual path
        main(image_path)

