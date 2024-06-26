import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

# Set the default device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32 if device == 'cpu' else torch.cuda.FloatTensor)
torch.set_default_device(device)

# Load the dataset
df = pd.read_csv('Data/medium-data.csv')
df.dropna(subset=['Content'], inplace=True)  # Ensure there are no NaNs in the content to process
texts = df['Content'].tolist()
titles = df['Title'].tolist()  # Load essay titles for labeling points on the plot

# Load model and tokenizer with trust_remote_code enabled
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Function to get hidden states
def get_hidden_states(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    mean_hidden_states = hidden_states.mean(dim=1).squeeze().cpu().detach().numpy()
    return mean_hidden_states

# Collect all hidden states for texts
all_hidden_states = [get_hidden_states(text) for text in texts if text.strip()]  # Filter out empty texts

# Process hidden states if any valid ones collected
if all_hidden_states:
    all_hidden_states = np.stack(all_hidden_states)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(all_hidden_states)

    plt.figure(figsize=(12, 8))
    for i, title in enumerate(titles):
        plt.scatter(reduced_features[i, 0], reduced_features[i, 1])
        plt.text(reduced_features[i, 0], reduced_features[i, 1], f'{title}', fontsize=9)
    plt.title('PCA of Text Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
else:
    print("No valid hidden states were collected.")
