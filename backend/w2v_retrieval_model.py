import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# Q1: Preprocess the dataset
def preprocess_text(corpus):
    """
    Tokenizes the corpus and returns a list of tokens.
    
    Args:
        corpus (list of str): List of document titles (strings).
        
    Returns:
        list of list of str: List of tokenized words for each document.
    """
    return [title.lower().split() for title in corpus]

# Q2: Create training pairs using a sliding window approach
def create_training_pairs(tokens, window_size=5):
    """
    Creates training pairs using a sliding window approach.
    
    Args:
        tokens (list of list of str): List of tokenized documents.
        window_size (int): The size of the sliding window to define context.
        
    Returns:
        list of tuple: Training pairs of (target_word, context_word).
    """
    pairs = []
    for sentence in tokens:
        for i, target_word in enumerate(sentence):
            # Define the window of context words around the target word
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(sentence))
            for j in range(start, end):
                if i != j:  # Ensure we're not pairing the word with itself
                    pairs.append((target_word, sentence[j]))
    return pairs

# Q3: Initialize embeddings
def initialize_embeddings(vocab_size, embedding_size):
    """
    Initializes the word embeddings matrices.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_size (int): Size of each embedding vector.
        
    Returns:
        np.ndarray: Initialized word embeddings matrices (input and output).
    """
    return np.random.randn(vocab_size, embedding_size) * 0.01

# Q4: Negative sampling method
def negative_sample(vocab_size, num_samples):
    """
    Returns negative samples for a given word.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        num_samples (int): Number of negative samples to generate.
        
    Returns:
        list: List of negative sample indices.
    """
    return random.sample(range(vocab_size), num_samples)

# Q5: Training Word2Vec using negative sampling
def train_word2vec(training_pairs, vocab_size, embedding_size, learning_rate=0.01, epochs=1, window_size=5, negative_samples=5):
    """
    Trains a Word2Vec model using negative sampling.
    
    Args:
        training_pairs (list of tuple): List of (target_word, context_word) pairs.
        vocab_size (int): Size of the vocabulary.
        embedding_size (int): Dimensionality of word embeddings.
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of training epochs.
        window_size (int): Context window size.
        negative_samples (int): Number of negative samples to use.
        
    Returns:
        np.ndarray: Final word embeddings (input and output).
    """
    W1 = initialize_embeddings(vocab_size, embedding_size)  # Input embeddings (target word embeddings)
    W2 = initialize_embeddings(vocab_size, embedding_size)  # Output embeddings (context word embeddings)
    
    for epoch in range(epochs):
        total_loss = 0
        for target_word, context_word in training_pairs:
            target_idx = word2index[target_word]
            context_idx = word2index[context_word]
            
            # Positive sampling (target and context words)
            vi = W1[target_idx]  # Target word vector (input)
            vo = W2[context_idx]  # Context word vector (output)
            
            # Negative sampling
            neg_samples = negative_sample(vocab_size, negative_samples)
            Vn = W2[neg_samples]  # Negative sample word vectors
            
            # Compute the gradients
            grad_vi, grad_vo, grad_Vn = w2vgrads(vi, vo, Vn)
            
            # Update the embeddings using the gradients
            W1[target_idx] -= learning_rate * grad_vi
            W2[context_idx] -= learning_rate * grad_vo
            W2[neg_samples] -= learning_rate * grad_Vn
            
            # Compute and accumulate the loss
            positive_loss = -np.log(sigmoid(np.dot(vo, vi)))
            negative_loss = -np.sum(np.log(sigmoid(-np.dot(Vn, vi))))
            total_loss += (positive_loss + negative_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")
    
    return W1, W2

# Q6: Compute the gradients
def w2vgrads(vi, vo, Vns):
    """
    Computes gradients for vi, vo, and negative samples.
    
    Args:
        - vi:  Vector of shape (d,), a sample in the input word vector matrix.
        - vo:  Vector of shape (d,), a positive sample in the output word vector matrix.
        - vns: Vector of shape (d, k), k negative samples in the output word vector matrix.
    
    Returns:
        - dvi, dvo, dVns: Gradients of J with respect to vi, vo, and vns.
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    dvi = (1 - sigmoid(np.dot(vo, vi))) * vo - np.sum((sigmoid(np.dot(Vns, vi)))[:, None] * Vns, axis=0)
    dvo = (1 - sigmoid(np.dot(vo, vi))) * vi
    dVns = -sigmoid(np.dot(Vns, vi))[:, None] * vi

    return dvi, dvo, dVns

# Q7: Helper function for sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(x))

# Q8: Function to find the most similar words (neighbors) to a query
def find_similar(query, word2index, W1, W2, top_k=10):
    """
    Given a query word, return the most similar words from the embedding matrix using cosine similarity.
    
    Args:
        query (str): The query word.
        word2index (dict): Mapping of words to indices.
        W1 (np.ndarray): Word embeddings.
        W2 (np.ndarray): Context word embeddings.
        top_k (int): Number of top similar words to return.
        
    Returns:
        list: List of tuples with (word, similarity).
    """
    if query not in word2index:
        return []

    query_idx = word2index[query]
    query_vec = W1[query_idx]  # Get the embedding vector of the query word
    
    # Calculate cosine similarity between query vector and all word vectors
    similarities = cosine_similarity([query_vec], W1)[0]
    
    # Get the indices of the top_k most similar words
    top_indices = similarities.argsort()[-top_k-1:-1][::-1]
    similar_words = [(index2word[i], similarities[i]) for i in top_indices]
    
    return similar_words

# Main Function
if __name__ == "__main__":
    # Assuming you have preprocessed text
    corpus = ["Example sentence 1.", "Another example sentence."]  # Replace with your actual corpus
    tokens = preprocess_text(corpus)
    
    # Create word-to-index mapping
    word2index = {word: idx for idx, word in enumerate(set([word for sentence in tokens for word in sentence]))}
    index2word = {idx: word for word, idx in word2index.items()}
    
    # Create training pairs
    training_pairs = create_training_pairs(tokens)
    
    # Train the model
    embedding_size = 100
    learning_rate = 0.01
    epochs = 10
    negative_samples = 5
    W1, W2 = train_word2vec(training_pairs, len(word2index), embedding_size, learning_rate, epochs, negative_samples=negative_samples)
    
    # Save the embeddings
    np.save("W1_embeddings.npy", W1)
    np.save("W2_embeddings.npy", W2)
    
    print("Training complete!")
    
    # Example: Find the most similar words to 'example'
    similar_words = find_similar('example', word2index, W1, W2)
    print("Top similar words to 'example':")
    for word, sim in similar_words:
        print(f"{word}: {sim}")