# qa_model.py

import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformer_retrieval_model import (
    Transformer,
    retrieve_similar_docs,
    masked_loss,
    preprocess_data_contrastive
)

def load_retrieval_model(model_path):
    """
    Load the custom Transformer-based retrieval model from disk.
    """
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"Transformer": Transformer, "masked_loss": masked_loss}
    )

def load_retrieval_assets(data_dir):
    """
    Use preprocess_data_contrastive to obtain:
      - A tf.data.Dataset (unused here)
      - documents: list of raw document strings
      - tokenizer: the same Tokenizer instance used during training
    """
    _, documents, tokenizer = preprocess_data_contrastive(
        data_dir, encoder_maxlen=256, batch_size=1
    )
    return documents, tokenizer

def encode_documents(documents, tokenizer, retrieval_model):
    """
    Convert each document into its embedding by:
      1. Converting text to token IDs
      2. Prepending [CLS] token ID
      3. Padding/truncating to fixed length
      4. Passing through retrieval_model.encode to get CLS embedding
    Returns a tensor of shape (num_documents, embedding_dim).
    """
    # Retrieve [CLS] token ID
    cls_id = tokenizer.word_index["[CLS]"]

    # Convert documents to sequences and add CLS
    sequences = tokenizer.texts_to_sequences(documents)
    sequences = [[cls_id] + seq for seq in sequences]

    # Pad or truncate to length 256
    padded = pad_sequences(sequences, maxlen=256, padding='post', truncating='post')
    inputs = tf.convert_to_tensor(padded, dtype=tf.int32)

    # Encode each document and collect embeddings
    embeddings = []
    for vector in inputs:
        vector = tf.expand_dims(vector, axis=0)  # shape (1, 256)
        encoded = retrieval_model.encode(vector, training=False, enc_padding_mask=None)
        embeddings.append(tf.squeeze(encoded, axis=0))  # shape (embedding_dim,)
    return tf.stack(embeddings)  # shape (num_documents, embedding_dim)

def load_generation_model(model_name="google/flan-t5-base"):
    """
    Load the local Seq2Seq generation model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def retrieve_contexts(query, retrieval_model, tokenizer, doc_embeddings, documents, top_k=1):
    """
    Use the retrieval_model and its tokenizer to find the top_k most similar documents.
    Returns a list of document strings.
    """
    indices = retrieve_similar_docs(
        query=query,
        doc_embeddings=doc_embeddings,
        tokenizer=tokenizer,
        model=retrieval_model,
        top_k=top_k
    )
    return [documents[i] for i in indices]

def generate_answer(question, contexts, gen_tokenizer, gen_model):
    """
    Build a prompt from the retrieved contexts and the question,
    then generate an answer using the Seq2Seq model.
    """
    prompt = "Answer the question based on the following contexts:\n\n"
    for idx, ctx in enumerate(contexts, start=1):
        prompt += f"Context {idx}: {ctx}\n\n"
    prompt += f"Question: {question}\nAnswer:"

    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = gen_model.generate(**inputs, max_new_tokens=512)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer_question(
    question,
    retrieval_model, retrieval_tokenizer,
    doc_embeddings, documents,
    gen_tokenizer, gen_model
):
    """
    End-to-end function that takes a question string and returns a generated answer.
    """
    contexts = retrieve_contexts(
        question, retrieval_model, retrieval_tokenizer,
        doc_embeddings, documents, top_k=5
    )
    return generate_answer(question, contexts, gen_tokenizer, gen_model)

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "transformer_model.keras"
    DATA_DIR = "../data"
    QUESTION = "How to Remove Duplicates From a Python List?"

    # Start execution
    print(">> Start running qa_model.py")

    # Load retrieval model
    print(">> Loading retrieval model from:", MODEL_PATH)
    retrieval_model = load_retrieval_model(MODEL_PATH)
    print("✅ Retrieval model loaded")

    # Load documents and tokenizer
    print(">> Loading documents and tokenizer")
    documents, retrieval_tokenizer = load_retrieval_assets(DATA_DIR)
    print(f"✅ Loaded {len(documents)} documents")

    # Encode documents into embeddings
    print(">> Encoding documents into embeddings (this may take some time)…")
    doc_embeddings = encode_documents(documents, retrieval_tokenizer, retrieval_model)
    print(f"✅ Encoded documents shape: {doc_embeddings.shape}")

    # Load generation model
    print(">> Loading generation model (flan-t5-base)…")
    gen_tokenizer, gen_model = load_generation_model()
    print("✅ Generation model loaded")

    # Display question
    print(">> Question to ask:", QUESTION)

    # Retrieve contexts
    print(">> Retrieving top-k contexts")
    contexts = retrieve_contexts(
        QUESTION,
        retrieval_model, retrieval_tokenizer,
        doc_embeddings, documents,
        top_k=5
    )
    print(f"✅ Retrieved {len(contexts)} contexts:")
    for i, ctx in enumerate(contexts, start=1):
        preview = ctx.replace("\n", " ")[:60] + "..."
        print(f"   Context {i} preview: {preview}")

    # Generate answer
    print(">> Generating answer from contexts")
    answer = generate_answer(QUESTION, contexts, gen_tokenizer, gen_model)
    print("✅ Answer generated")

    # Final output
    print("\n=== Final QA Output ===")
    print("Question:", QUESTION)
    print("Answer:", answer)
    print(">> End of qa_model.py")