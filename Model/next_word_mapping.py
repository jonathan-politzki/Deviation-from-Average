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
df = pd.read_csv('/Users/jonathanpolitzki/Desktop/Coding/Deviation from Average/Data/substack_data.csv')
texts = df['text'].tolist()

# Load model and tokenizer with trust_remote_code enabled
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Example function to get hidden states
def get_hidden_states(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    # Remove singleton dimensions and ensure it is 1D
    mean_hidden_states = hidden_states.mean(dim=1).squeeze().cpu().detach().numpy()
    print("Shape of hidden states:", mean_hidden_states.shape)  # Should now print (2560,)
    return mean_hidden_states

# Collect all hidden states for texts
all_hidden_states = [get_hidden_states(text) for text in texts if text.strip()]  # Ensure text is not empty

# Stack all hidden states to create a 2D array (samples, features)
if all_hidden_states:
    all_hidden_states = np.stack(all_hidden_states)
    print("Stacked hidden states shape:", all_hidden_states.shape)  # Should print (number of texts, 2560)

    # Perform PCA on the 2D array of hidden states
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(all_hidden_states)

    # Plot the PCA results
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:,0], reduced_features[:,1], alpha=0.5)
    plt.title('PCA of Text Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
else:
    print("No valid hidden states were collected.")
