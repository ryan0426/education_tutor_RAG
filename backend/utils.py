import pandas as pd
import re

def get_train_test_data(data_dir):
    # train_data = pd.read_json(f"{data_dir}/train.json")
    # train_data = train_data.to_dict(orient="records")
    # test_data = pd.read_json(f"{data_dir}/test.json")
    # test_data = test_data.to_dict(orient="records")

    # example 
    data = pd.read_json(f"{data_dir}/example.json")
    train_data = data.to_dict(orient="records")
    test_data = data.to_dict(orient="records")

    return train_data, test_data

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
