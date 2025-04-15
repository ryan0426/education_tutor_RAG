import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import utils

# Initialize random seed and suppress TensorFlow logging
tf.keras.utils.set_random_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def preprocess_data(data_dir, encoder_maxlen=256):
    '''
    Loads and preprocesses the dataset, tokenizing the documents.
    Assumes data is a list of dictionaries with a "document" key.
    '''
    train_data, test_data = utils.get_train_test_data(data_dir)
    document = [entry['document'] for entry in train_data]
    document_test = [entry['document'] for entry in test_data]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n', 
        oov_token='[UNK]', 
        lower=False
    )
    tokenizer.fit_on_texts(document)
    inputs = tokenizer.texts_to_sequences(document)
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
    inputs = tf.cast(inputs, dtype=tf.int32)

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    return tf.data.Dataset.from_tensor_slices(inputs).shuffle(BUFFER_SIZE).batch(BATCH_SIZE), document, document_test, tokenizer

# Positional Encoding Function
def positional_encoding(positions, d_model):
    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Create Padding Mask Function
def create_padding_mask(seq):
    """
    Creates a padding mask for the input sequence. The padding mask will mark the padded indices as zero.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)  
    return seq[:, tf.newaxis, :]  

# Masked Loss Function (for sequence-to-sequence tasks)
def masked_loss(real, pred):
    """
    Computes the loss with a mask to ignore padding tokens during training.
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # Padding mask
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask  # Mask the loss
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

# Encoder Layer for Transformer Model
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(fully_connected_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        return self.layernorm2(x + ffn_output)

# Full Encoder for Transformer
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size, max_pos_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(max_pos_encoding, embedding_dim)
        self.enc_layers = [EncoderLayer(embedding_dim, num_heads, fully_connected_dim, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        x = self.embedding(x) * tf.math.sqrt(tf.cast(x.shape[-1], tf.float32)) + self.pos_encoding
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x=x, training=training, mask=mask)
        return x

# Transformer Model (Encoder only for retrieval)
@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size, max_pos_encoding, dropout_rate=0.1,**kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.fully_connected_dim = fully_connected_dim
        self.vocab_size = vocab_size
        self.max_pos_encoding = max_pos_encoding
        self.dropout_rate = dropout_rate
        self.encoder = Encoder(num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size, max_pos_encoding, dropout_rate)

    def call(self, input_sentence, training, enc_padding_mask):
        return self.encoder(x=input_sentence, training=training, mask=enc_padding_mask)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "fully_connected_dim": self.fully_connected_dim,
            "vocab_size": self.vocab_size,
            "max_pos_encoding": self.max_pos_encoding,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Cosine Similarity for Retrieval
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_similar_docs(query, doc_embeddings, tokenizer, model, top_k=3):
    query_tokens = tokenizer.texts_to_sequences([query])
    query_tokens = tf.keras.preprocessing.sequence.pad_sequences(query_tokens, maxlen=256, padding='post', truncating='post')
    query_input = tf.expand_dims(query_tokens[0], 0)  
    query_embed = model(query_input, training=False, enc_padding_mask=None)  
    query_embed = tf.reduce_mean(query_embed, axis=1)  
    query_embed = tf.squeeze(query_embed, axis=0)  
    doc_embeddings = tf.squeeze(doc_embeddings, axis=1)  
    doc_embeddings_flat = tf.reduce_mean(doc_embeddings, axis=1) 
    query_embed = query_embed.numpy().reshape(1, -1)  
    doc_embeddings_flat = doc_embeddings_flat.numpy()  
    similarities = cosine_similarity(query_embed, doc_embeddings_flat) 
    top_k_idx = np.argsort(similarities[0])[-top_k:][::-1]
    
    return top_k_idx

# Training Process
def train_model(dataset, vocab_size, num_layers, embedding_dim, num_heads, fully_connected_dim, epochs=5):
    transformer = Transformer(num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size, max_pos_encoding=256)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for inp in dataset:
            enc_padding_mask = create_padding_mask(inp)
            with tf.GradientTape() as tape:
                predictions = transformer(input_sentence=inp, training=True, enc_padding_mask=enc_padding_mask)
                loss = masked_loss(inp, predictions)
            
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            
        print(f"Epoch {epoch+1}: Loss {loss.numpy()}")
    return transformer

# Example of how to integrate the model with the data and retrieve top K documents
if __name__ == '__main__':
    data_dir = '../data'
    dataset, document, document_test, tokenizer = preprocess_data(data_dir)

    vocab_size = len(tokenizer.word_index) + 1
    transformer = train_model(dataset, vocab_size, num_layers=2, embedding_dim=128, num_heads=2, fully_connected_dim=128)
    transformer.save("transformer_model.keras")

    loaded_transformer = tf.keras.models.load_model(
        "transformer_model.keras",
        custom_objects={'Transformer': Transformer, 'masked_loss': masked_loss}
    )

    query = "What is deep learning?"
    doc_embeddings = np.array([
        loaded_transformer(tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences([doc]), maxlen=256, padding='post', truncating='post')), 
        training=False, enc_padding_mask=None) 
        for doc in document
    ])
    
    top_k_docs = retrieve_similar_docs(query, doc_embeddings, tokenizer, loaded_transformer)
    
    print("Top 3 relevant documents for the query:")
    for idx in top_k_docs:
        print(f"Document {idx}: {document[idx]}")