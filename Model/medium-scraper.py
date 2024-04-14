import os
import feedparser
from bs4 import BeautifulSoup
import pandas as pd

# URL of the RSS feed
rss_url = 'https://medium.com/feed/@travismay'

# Parse the feed
feed = feedparser.parse(rss_url)

# Create a list to hold all entries
entries = []

# Loop through each post
for post in feed.entries:
    # Clean the summary content to remove HTML tags
    soup = BeautifulSoup(post.summary, 'html.parser')
    cleaned_text = soup.get_text(separator=' ', strip=True)
    
    # Append to the list as a tuple
    entries.append((post.title, post.link, cleaned_text))

# Convert the list of tuples into a DataFrame
df = pd.DataFrame(entries, columns=['Title', 'URL', 'Content'])

# Ensure the directory exists
directory = 'Data'
if not os.path.exists(directory):
    os.makedirs(directory)

# Write the DataFrame to a CSV file
output_file_path = os.path.join(directory, 'medium-data.csv')
df.to_csv(output_file_path, index=False)

print(f'Data written to {output_file_path}')