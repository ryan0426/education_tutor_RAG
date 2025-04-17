import os
import numpy as np
import tensorflow as tf
import utils
import random

# Initialize random seed and suppress TensorFlow logging
tf.keras.utils.set_random_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def preprocess_data_contrastive(data_dir, encoder_maxlen=256, batch_size=64):
    raw_data = utils.get_train_test_data(data_dir)
    documents = [entry["document"] for entry in raw_data]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n',
        oov_token='[UNK]',
        lower=False
    )
    tokenizer.fit_on_texts(documents)

    sequences = tokenizer.texts_to_sequences(documents)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=encoder_maxlen, padding='post', truncating='post'
    )
    sequences = tf.constant(sequences, dtype=tf.int32)

    def generate_triplets():
        num_docs = sequences.shape[0]
        for i in range(num_docs):
            anchor = sequences[i]
            positive = augment_sequence(sequences[i])
            neg_index = (i + 1000) % num_docs
            negative = sequences[neg_index]
            yield anchor, positive, negative

    triplet_dataset = tf.data.Dataset.from_generator(
        generate_triplets,
        output_signature=(
            tf.TensorSpec(shape=(encoder_maxlen,), dtype=tf.int32),
            tf.TensorSpec(shape=(encoder_maxlen,), dtype=tf.int32),
            tf.TensorSpec(shape=(encoder_maxlen,), dtype=tf.int32),
        )
    ).shuffle(10000).batch(batch_size)

    return triplet_dataset, documents, tokenizer

def augment_sequence(seq, drop_prob=0.1):
    """Randomly drops some tokens (not special tokens) from the sequence."""
    seq = seq.numpy()
    # Keep SOS (assumed 0 index) and EOS (assumed last non-zero)
    non_zero = seq[seq != 0]
    if len(non_zero) <= 2:
        return tf.convert_to_tensor(seq)
    
    sos = non_zero[0]
    eos = non_zero[-1]
    middle = non_zero[1:-1]

    # Randomly drop tokens
    keep_mask = np.random.rand(len(middle)) > drop_prob
    middle = middle[keep_mask]

    new_seq = [sos] + list(middle) + [eos]
    # Pad back to original length
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        [new_seq], maxlen=len(seq), padding='post'
    )
    return tf.convert_to_tensor(padded[0])

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
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # Padding mask
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask  # Mask the loss
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super(AttentionPooling, self).__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, mask):
        # inputs: (batch_size, seq_len, embedding_dim)
        # mask: (batch_size, 1, seq_len)
        scores = self.dense(inputs)  # (batch_size, seq_len, 1)
        scores = tf.squeeze(scores, axis=-1)  # (batch_size, seq_len)

        if mask is not None:
            mask = tf.squeeze(mask, axis=1)  # (batch_size, seq_len)
            scores += (mask * -1e9)  # apply mask: padding gets -inf

        weights = tf.nn.softmax(scores, axis=-1)  # (batch_size, seq_len)
        weights = tf.expand_dims(weights, axis=-1)  # (batch_size, seq_len, 1)

        # Weighted sum
        pooled = tf.reduce_sum(inputs * weights, axis=1)  # (batch_size, embedding_dim)
        return pooled
    
# Encoder Layer for Transformer Model
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.3):
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
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size, max_pos_encoding, dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(max_pos_encoding, embedding_dim)
        self.enc_layers = [EncoderLayer(embedding_dim, num_heads, fully_connected_dim, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.embedding_dim = embedding_dim

    def call(self, x, training, mask):
        # x = self.embedding(x) * tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32)) + self.pos_encoding
        x = self.embedding(x)
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x=x, training=training, mask=mask)
        return x

# Transformer Model (Encoder only for retrieval)
@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size, max_pos_encoding, dropout_rate=0.3,**kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.fully_connected_dim = fully_connected_dim
        self.vocab_size = vocab_size
        self.max_pos_encoding = max_pos_encoding
        self.dropout_rate = dropout_rate
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        self.cls_proj = tf.keras.layers.Dense(embedding_dim, activation='tanh')
        self.encoder = Encoder(num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size, max_pos_encoding, dropout_rate)
        self.attn_pool = AttentionPooling(embedding_dim)

    def call(self, input_sentence, training, enc_padding_mask):
        encoder_output = self.encoder(x=input_sentence, training=training, mask=enc_padding_mask)
        final_output = self.final_layer(encoder_output)  # project to vocab size
        return final_output
    
    def encode(self, input_sentence, training, enc_padding_mask):
        encoder_output = self.encoder(x=input_sentence, training=training, mask=enc_padding_mask)
        pooled = self.attn_pool(encoder_output, mask=enc_padding_mask)
        return pooled
    
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

def rerank_by_length(documents, indices, scores, min_len=100, penalty=0.1):
    reranked = []
    for i in range(len(indices)):
        doc = documents[indices[i]]
        length = len(doc.split())
        score = scores[i]
        if length < min_len:
            score *= penalty
        reranked.append((indices[i], score))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in reranked]

def retrieve_similar_docs(query, doc_embeddings, tokenizer, model, document, top_k=10):
    # Step 1: Tokenize and pad query
    query_tokens = tokenizer.texts_to_sequences([query])
    query_tokens = tf.keras.preprocessing.sequence.pad_sequences(
        query_tokens, maxlen=256, padding='post', truncating='post'
    )
    query_input = tf.convert_to_tensor(query_tokens)  # (1, 256)

    # Step 2: Get query embedding (already mean-pooled inside model.encode)
    query_embedding = model.encode(query_input, training=False, enc_padding_mask=None)  # (1, emb_dim)

    # Step 3: Normalize both embeddings
    query_embedding = tf.math.l2_normalize(query_embedding, axis=1).numpy()  # shape (1, emb_dim)
    doc_embeddings_np = tf.math.l2_normalize(doc_embeddings, axis=1).numpy()  # shape (num_docs, emb_dim)

    # Step 4: Compute cosine similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings_np)  # shape (1, num_docs)

    # Step 5: Get top-k result indices
    top_k_idx = np.argsort(similarities[0])[-top_k:][::-1]
    top_k_scores = similarities[0][top_k_idx]
    reranked_idx = rerank_by_length(documents=document, indices=top_k_idx, scores=top_k_scores)

    return top_k_idx

def contrastive_loss(anchor, positive, negative, margin=0.3):
    # Normalize embeddings
    anchor = tf.math.l2_normalize(anchor, axis=1)
    positive = tf.math.l2_normalize(positive, axis=1)
    negative = tf.math.l2_normalize(negative, axis=1)

    # Cosine similarities
    pos_sim = tf.reduce_sum(anchor * positive, axis=1)
    neg_sim = tf.reduce_sum(anchor * negative, axis=1)

    # Loss: max(0, margin - (pos - neg))
    loss = tf.maximum(0.0, margin - pos_sim + neg_sim)
    return tf.reduce_mean(loss)

def train_model_contrastive(dataset, vocab_size, num_layers, embedding_dim, num_heads, fully_connected_dim, epochs=10):
    transformer = Transformer(num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size, max_pos_encoding=256)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = []

        for anchor, positive, negative in dataset:
            anchor_mask = create_padding_mask(anchor)
            positive_mask = create_padding_mask(positive)
            negative_mask = create_padding_mask(negative)

            with tf.GradientTape() as tape:
                anchor_embed = transformer.encode(anchor, training=True, enc_padding_mask=anchor_mask)
                positive_embed = transformer.encode(positive, training=True, enc_padding_mask=positive_mask)
                negative_embed = transformer.encode(negative, training=True, enc_padding_mask=negative_mask)

                loss = contrastive_loss(anchor_embed, positive_embed, negative_embed)
                epoch_loss.append(loss.numpy())

            grads = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(grads, transformer.trainable_variables))

        print(f"Epoch {epoch+1}: Loss {np.mean(epoch_loss):.4f}")
    return transformer

def clean_tokens(text):
    return text.replace("[CLS]", "").replace("[SOS]", "").replace("[EOS]", "").strip()

def training():
    data_dir = '../data'
    dataset, document, tokenizer = preprocess_data_contrastive(data_dir)

    vocab_size = len(tokenizer.word_index) + 1
    transformer = train_model_contrastive(dataset, vocab_size, num_layers=2, embedding_dim=128, num_heads=2, fully_connected_dim=128)

    _ = transformer(tf.ones((1, 256), dtype=tf.int32), training=False, enc_padding_mask=None)
    transformer.save("transformer_model.keras")
    print("model saved")

    doc_embeddings = tf.stack([
        tf.squeeze(transformer.encode(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences([doc]), maxlen=256, padding='post', truncating='post')),
            training=False,
            enc_padding_mask=None
        ), axis=0)  # remove the singleton dimension
        for doc in document
    ], axis=0)

    query = "list operations in Python"
    top_k_docs = retrieve_similar_docs(query, doc_embeddings, tokenizer, transformer, document)
    
    print("Top 10 relevant documents for the query:")
    for idx in top_k_docs:
        print(f"Document {idx}: {clean_tokens(document[idx])}")

def testing():
    data_dir = '../data'
    dataset, document, tokenizer = preprocess_data_contrastive(data_dir)

    loaded_transformer = tf.keras.models.load_model(
        "transformer_model.keras",
        custom_objects={'Transformer': Transformer, 'masked_loss': masked_loss}
    )
    print("model loaded")

    doc_embeddings = tf.stack([
        loaded_transformer.encode(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences([doc]), maxlen=256, padding='post', truncating='post')),
            training=False,
            enc_padding_mask=None
        )
        for doc in document
    ])

    query = "list operations in Python"
    top_k_docs = retrieve_similar_docs(query, doc_embeddings, tokenizer, loaded_transformer)
    
    print("Top 10 relevant documents for the query:")
    for idx in top_k_docs:
        print(f"Document {idx}: {document[idx]}")


if __name__ == '__main__':
    training()
    #testing()
