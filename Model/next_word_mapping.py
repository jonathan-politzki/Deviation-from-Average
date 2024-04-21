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
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    for i in range(15):  # Limiting to 15 tokens to avoid out-of-range errors
        input_ids = torch.tensor([token_ids[:i+1]]).to(device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
        logits = model.lm_head(outputs.hidden_states[-1][0, :, :]).detach().cpu().numpy()
        predicted_token_id = logits.argmax(axis=1)[-1]  # This ensures we are getting the last token predicted id correctly
        predicted_embedding = model.model.embed_tokens.weight[predicted_token_id].detach().cpu().numpy()

        print(f"Token {i}: {tokens[i]} - Predicted Token ID: {predicted_token_id}")  # Debug print

        actual_embeddings.append(hidden_states)
        predicted_embeddings.append(predicted_embedding)
        if i < len(tokens) - 1:
            word_labels.append(tokens[i+1])

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
    actual_plot, = plt.plot(*zip(*actual_pca), marker='o', color='blue', linestyle='-', markersize=5, label='Actual Words')
    predicted_plot, = plt.plot(*zip(*predicted_pca), marker='x', color='red', linestyle='-', markersize=5, label='Predicted Words')

    for i, word in enumerate(word_labels):
        plt.text(actual_pca[i][0], actual_pca[i][1], word, fontsize=8, ha='right')

    plt.title(f'PCA of Actual vs Predicted Word Progressions for {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(handles=[actual_plot, predicted_plot], loc='upper right')
    plt.grid(True)
    plt.show()

# Process only the first non-empty essay
for essay, title in zip(essays, titles):
    if essay.strip():  # Check if the essay is not empty
        actual_embeddings, predicted_embeddings, word_labels = get_word_progressions(essay, tokenizer, model)
        visualize_progressions(actual_embeddings, predicted_embeddings, word_labels, title)
        break  # Exit after processing the first non-empty essay
