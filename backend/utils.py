import pandas as pd
import re

def get_train_test_data(data_dir):
    data = pd.read_json(f"{data_dir}/data.json")
    data['document'] = data['document'].apply(preprocess_text)
    data = data[data['document'].str.strip().str.len() > 20]
    train_data = data.to_dict(orient="records")
    return train_data

def preprocess(input_data):
    # Define the custom preprocessing function
    def preprocess_util(input_data):
        # Convert all text to lowercase
        lowercase = input_data.lower()
        # Remove newlines and double spaces
        removed_newlines = re.sub("\n|\r|\t", " ", lowercase)
        removed_double_spaces = ' '.join(removed_newlines.split(' '))
        # Add start of sentence and end of sentence tokens
        s = '[SOS] ' + removed_double_spaces + ' [EOS]'
        return s

    # Extract only the document column from input data
    document = input_data['document'].apply(lambda row : preprocess_util(row))
    
    return document

def preprocess_text(input_data):
        # Convert all text to lowercase
        lowercase = input_data.lower()
        # Remove newlines and double spaces
        removed_newlines = re.sub("\n|\r|\t", " ", lowercase)
        removed_double_spaces = ' '.join(removed_newlines.split(' '))
        # Add start of sentence and end of sentence tokens
        s = '[SOS] ' + removed_double_spaces + ' [EOS]'
        return s