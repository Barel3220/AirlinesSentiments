import pandas as pd
import re
from sklearn.model_selection import train_test_split

file_path = 'airline-sentiment-data.csv'
data = pd.read_csv(file_path)

print(data.head())

data = data[['text', 'airline_sentiment']]

# Map sentiment to numerical values
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['label'] = data['airline_sentiment'].map(sentiment_mapping)

airline_handles = [
    '@AmericanAir', '@SouthwestAir', '@united', '@JetBlue', '@USAirways', '@VirginAmerica'
]


def clean_text(text):
    """
    Clean the text by removing airline mentions, URLs, and non-alphanumeric characters.

    Parameters:
    - text: The original text.

    Returns:
    - text: The cleaned text.
    """
    # Remove airline mentions
    for handle in airline_handles:
        text = text.replace(handle, '')
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove non-alphanumeric characters except specified punctuation
    text = re.sub(r'[^a-zA-Z0-9\s#.,?!]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


data['text'] = data['text'].apply(clean_text)

data = data.dropna(subset=['text', 'label'])

test_data = data[-10:]
data = data[:-10]

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Remove any overlap between training and validation sets
train_texts = set(train_data['text'])
val_data = val_data[~val_data['text'].isin(train_texts)]

train_data.to_csv('data/train.csv', index=False)
val_data.to_csv('data/val.csv', index=False)
test_data.to_csv('data/test.csv', index=False)

# Display the sizes of the datasets
print(f'Training set size: {len(train_data)}')
print(f'Validation set size: {len(val_data)}')
print(f'Test set size: {len(test_data)}')
