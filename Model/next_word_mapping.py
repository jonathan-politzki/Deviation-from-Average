import pandas as pd
import torch
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
    mean_hidden_states = hidden_states.mean(dim=1).cpu().detach().numpy()  # Average over the sequence length dimension
    print("Shape of hidden states:", mean_hidden_states.shape)  # Debugging line
    return mean_hidden_states


# Collect hidden states for all texts
all_hidden_states = [get_hidden_states(text) for text in texts if text.strip()]

# Check if all_hidden_states is not empty and all have the same shape
if not all_hidden_states:
    print("No valid hidden states were collected.")
else:
    first_shape = all_hidden_states[0].shape
    if all(state.shape == first_shape for state in all_hidden_states):
        # Perform PCA on the hidden states
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
        print("Error: Not all hidden states have the same shape.")
