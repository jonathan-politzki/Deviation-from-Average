import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Set up device for computations
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model and tokenizer
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your dataset
df = pd.read_csv('/Users/jonathanpolitzki/Desktop/Coding/Deviation from Average/Data/substack_data_two_columns.csv')
essays = df['Text'].tolist()
titles = df['Title'].tolist()

# Function to get word progressions
def get_word_progressions(text, tokenizer, model):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    actual_embeddings = []
    predicted_embeddings = []
    word_labels = []

    for i in range(len(tokens) - 1):  # Up to the second to last token
        input_ids = torch.tensor([token_ids[:i+1]]).to(device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][0, -1, :].cpu().numpy()  # Last token's hidden state
        logits = outputs.logits[0, -1, :]  # Logits for the last token
        predicted_token_id = logits.argmax().item()
        predicted_embedding = model.transformer.wte.weight[predicted_token_id].cpu().numpy()

        actual_embeddings.append(hidden_states)
        predicted_embeddings.append(predicted_embedding)
        word_labels.append(tokens[i+1])  # The actual next word

    return np.array(actual_embeddings), np.array(predicted_embeddings), word_labels

# Function to visualize progressions
def visualize_progressions(actual_embeddings, predicted_embeddings, word_labels, title):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(np.vstack((actual_embeddings, predicted_embeddings)))

    # Split the PCA results
    actual_pca = pca_result[:len(actual_embeddings)]
    predicted_pca = pca_result[len(actual_embeddings):]

    # Plot the points and lines
    plt.figure(figsize=(12, 8))
    for i, (actual_point, predicted_point, word) in enumerate(zip(actual_pca, predicted_pca, word_labels)):
        plt.scatter(*actual_point, color='blue')
        plt.scatter(*predicted_point, color='red')
        plt.plot([actual_point[0], predicted_point[0]], [actual_point[1], predicted_point[1]], 'grey', alpha=0.5)
        plt.text(actual_point[0], actual_point[1], word, fontsize=8, ha='right')

    plt.title(f'PCA of Actual vs Predicted Word Progressions for {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(['Actual Word', 'Predicted Word'], loc='upper right')
    plt.grid(True)
    plt.show()

# Process only the first non-empty essay
for essay, title in zip(essays, titles):
    if essay.strip():  # Check if the essay is not empty
        actual_embeddings, predicted_embeddings, word_labels = get_word_progressions(essay, tokenizer, model)
        visualize_progressions(actual_embeddings, predicted_embeddings, word_labels, title)
        break  # Exit after processing the first non-empty essay
