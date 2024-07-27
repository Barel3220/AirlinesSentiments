import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load the CSV file
file_path = 'airline-sentiment-data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Select relevant columns
data = data[['text', 'airline_sentiment']]

# Map sentiment to numerical values
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['label'] = data['airline_sentiment'].map(sentiment_mapping)

# List of airline Twitter handles
airline_handles = [
    '@AmericanAir', '@SouthwestAir', '@united', '@JetBlue', '@USAirways', '@VirginAmerica'
]


# Function to clean text
def clean_text(text):
    # Remove airline mentions
    for handle in airline_handles:
        text = text.replace(handle, '')
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s#.,?!]', '', text)  # Remove non-alphanumeric characters
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Clean the text column
data['text'] = data['text'].apply(clean_text)

# Drop any rows with missing values
data = data.dropna(subset=['text', 'label'])

# Split the data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the preprocessed data to CSV files
train_data.to_csv('data/train.csv', index=False)
val_data.to_csv('data/val.csv', index=False)
