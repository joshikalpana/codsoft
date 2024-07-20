import tensorflow as tf
import tensorflow_datasets as tfds
import pickle

# Load the Dataset
def load_dataset():
    data, info = tfds.load('coco/2014', with_info=True)
    train_data = data['train']
    val_data = data['validation']
    return train_data, val_data, info

def preprocess_caption(caption):
    caption = caption.numpy().decode('utf-8')
    return caption

def create_tokenizer(train_data):
    captions = []
    for img, cap in train_data:
        caption = preprocess_caption(cap)
        captions.append(caption)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    
    return tokenizer

# Save the tokenizer
if __name__ == "__main__":
    train_data, val_data, info = load_dataset()
    tokenizer = create_tokenizer(train_data)
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Tokenizer created and saved to tokenizer.pickle")
