import fasttext

model_path = "./model/fasttext_model.bin"
model = fasttext.load_model(model_path)

binary_model_path = "./model/binary_model.bin"
binary_model = fasttext.load_model(binary_model_path)

def classify_text(query):
    labels, probabilities = model.predict(query, k=1)
    
    if probabilities[0] >= 0.9:
        return labels[0]
    else:
        return None
    
def classify_binary(query):
    labels, probabilities = binary_model.predict(query, k=1)

    if probabilities[0] >= 0.9:
        print(labels[0])
        return labels[0]
    else:
        return None