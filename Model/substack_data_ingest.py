import os
from bs4 import BeautifulSoup
import pandas as pd

# Specify the directory containing the HTML files
data_directory = 'Data/Essays'
output_file_path = 'Data/substack_data.csv'

# List to hold all extracted text
all_texts = []

# Check if any files are being processed
print(f"Processing files in {data_directory}...")

# Iterate over each file in the Essays directory
for filename in os.listdir(data_directory):
    if filename.endswith('.html'):  # Adjusted to look for .html files
        file_path = os.path.join(data_directory, filename)
        print(f"Reading file: {filename}")  # Print the name of the current file
        
        # Read the HTML content from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract text from the HTML content
            text = soup.get_text(separator=' ', strip=True)
            if text:
                all_texts.append(text)
            else:
                print(f"Warning: No text extracted from {filename}.")
# Convert the list of texts into a DataFrame
output_df = pd.DataFrame(all_texts, columns=['text'])

if not output_df.empty:
    # Write the DataFrame to a new CSV file
    output_df.to_csv(output_file_path, index=False)
    print(f"Aggregated data written to {output_file_path}")
else:
    print("No data was aggregated. The output CSV file has not been created.")


# I may have to go through again and clean it up and tokenize it. There may be a need to also organize it by essay or structure it in some way, or to just keep it as a giant clump of te