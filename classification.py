import re
import fasttext

model_path = "./model/fasttext_model.bin"
model = fasttext.load_model(model_path)

binary_model_path = "./model/binary_model.bin"
binary_model = fasttext.load_model(binary_model_path)

def preprocess_text(text):
    if not isinstance(text, str):
        return "" # Return an empty string for missing values
    text = text.lower()  # Lowercase
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    return text

def classify_text(query):
    query = preprocess_text(query)
    labels, probabilities = model.predict(query, k=1)
    
    if probabilities[0] >= 0.8:
        return labels[0]
    else:
        return "__label__Khac"
    
def classify_binary(query):
    query = preprocess_text(query)
    labels, probabilities = binary_model.predict(query, k=1)
 
    if probabilities[0] >= 0.8:
        return labels[0]
    else:
        return None
    
if __name__ == "__main__":
    while True:
        test_query = input("Input: ")

        binary_result = classify_binary(test_query)
        print(f"Binary Classification Result: {binary_result}")
        
        if binary_result == "__label__Academic": 
            result = classify_text(test_query)
            print(f"Classification Result: {result}")