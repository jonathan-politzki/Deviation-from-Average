import os
import pandas as pd
from bs4 import BeautifulSoup

# Specify the directory containing the HTML files
data_directory = '/path/to/html/files'
output_file_path = '/path/to/output/substack_data.csv'

# Prepare a list to hold data
all_data = []

# Iterate over each file in the directory
for filename in os.listdir(data_directory):
    if filename.endswith('.html'):  # Ensuring only HTML files are processed
        file_path = os.path.join(data_directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            
            # Example to extract title - adjust the selector as per your HTML structure
            title = soup.find('h1').text if soup.find('h1') else 'No Title'
            
            # Extract text from the HTML content
            text = soup.get_text(separator=' ', strip=True)
            
            # Append title and text as a tuple to the list
            all_data.append((title, text))

# Create a DataFrame and specify column names
df = pd.DataFrame(all_data, columns=['Title', 'Text'])

# Save DataFrame to CSV
df.to_csv(output_file_path, index=False)

print(f'Data written to {output_file_path}')