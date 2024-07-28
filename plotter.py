import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - data: DataFrame containing the loaded data.
    """
    data = pd.read_csv(file_path)
    return data


def plot_airline_distribution(data):
    """
    Plot a pie chart showing the distribution of tweets by airline.

    Parameters:
    - data: DataFrame containing the data with an 'airline' column.
    """
    airline_counts = data['airline'].value_counts()

    def autopct_format(values):
        """
        Format the labels of the pie chart to show percentages and counts.
        """

        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}% ({val:d})'

        return my_format

    plt.figure(figsize=(10, 10))
    plt.pie(airline_counts, labels=airline_counts.index, autopct=autopct_format(airline_counts), startangle=140,
            colors=sns.color_palette('pastel'))
    plt.title('Distribution of Tweets by Airline')
    plt.show()


def plot_sentiment_over_time(data):
    """
    Plot the sentiment distribution over time.

    Parameters:
    - data: DataFrame containing the data with 'tweet_created' and 'airline_sentiment' columns.
    """
    # Convert 'tweet_created' to datetime
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])

    # Extract the date part for grouping
    data['date'] = data['tweet_created'].dt.date

    # Group by date and sentiment
    sentiment_over_time = data.groupby(['date', 'airline_sentiment']).size().unstack().fillna(0)

    # Plot sentiment over time
    sentiment_over_time.plot(kind='line', figsize=(12, 8), marker='o')
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.show()


def plot_data_distribution(data):
    """
    Plot a pie chart showing the distribution of sentiment labels.

    Parameters:
    - data: DataFrame containing the data with an 'airline_sentiment' column.
    """
    sentiment_counts = data['airline_sentiment'].value_counts()

    def autopct_format(values):
        """
        Format the labels of the pie chart to show percentages and counts.
        """

        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}% ({val:d})'

        return my_format

    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct=autopct_format(sentiment_counts), startangle=140,
            colors=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Sentiment Distribution')
    plt.show()


def plot_training_summary(history_path):
    """
    Plot the training summary, including training and validation loss, accuracy, and F1-score over epochs.

    Parameters:
    - history_path: Path to the JSON file containing the training history.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot validation F1-score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_f1'], label='Validation F1-Score')
    plt.title('Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes):
    """
    Plot a confusion matrix using a heatmap.

    Parameters:
    - cm: Confusion matrix.
    - classes: List of class names.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    """
    Main function to load data and generate plots.
    """
    data = load_data('airline-sentiment-data.csv')
    plot_data_distribution(data)
    plot_airline_distribution(data)
    plot_sentiment_over_time(data)
    plot_training_summary('data/history.json')


if __name__ == "__main__":
    main()
