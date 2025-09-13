import re
import time
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split

# Set path
train_data_path = "./data/classify/train_data.txt"
test_data_path = "./data/classify/test_data.txt"
model_path = "./model/fasttext_model.bin"
data_path = "./data/classify/dataset.csv"
stopwords_path = "./library/stopwords.txt"

with open(file=stopwords_path, mode="r", encoding="utf-8") as f:
    stopwords = f.read().splitlines()

# Text preprocessing function
def preprocess_text(text):
    """
    Preprocess text for classification.
    
    Args:
        text (str): Input text to be preprocessed
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return "" # Return an empty string for missing values
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # # Remove punctuation
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

# Load and preprocess the dataset
data = pd.read_csv(data_path, encoding="utf-8", sep=";")

# preprocess the datase
data.dropna(subset=['text'], inplace=True)  # Remove rows where 'text' is NaN
data['text'] = data['text'].apply(preprocess_text)

# Data Format for FastText
data['formatted'] = data['label'].astype(str) + ' ' + data['text']

# Split the data into training and test sets
train_data, test_data = train_test_split(data['formatted'], test_size=0.2, random_state=42)
train_data = train_data.str.strip()
test_data = test_data.str.strip()

# Save preprocessed data to files
train_data.to_csv(train_data_path, index=False, header=False)
test_data.to_csv(test_data_path, index=False, header=False)

start_time = time.time()

# Train model
model = fasttext.train_supervised(input=train_data_path, epoch=50, lr=1.0, wordNgrams=3, verbose=2, minCount=1, dim=100)
print(f'Time: {time.time() - start_time} seconds')

# Save model
model.save_model(model_path)
print("Model has been saved successfully!")
