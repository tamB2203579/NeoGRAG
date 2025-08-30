import fasttext

model_path = "./model/fasttext_model.bin"
model = fasttext.load_model(model_path)

labels_list = [label.replace('__label__', '') for label in model.labels]

def classify_text(query):
    labels, probabilities = model.predict(query, k=1)
    
    if probabilities[0] >= 0.9:
        return labels[0]
    else:
        return None