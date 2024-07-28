import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from model import ABSAModel
from dataset import ABSADataset
from train import train_model
from predict import predict
import logging

# Set logging level to ERROR to reduce verbosity
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - DataFrame containing 'text' and 'label' columns.
    """
    data = pd.read_csv(file_path)
    return data[['text', 'label']]


def calculate_class_weights(data, weight_factor=1.2):
    """
    Calculate class weights to handle class imbalance.

    Parameters:
    - data: DataFrame containing the data with 'label' column.
    - weight_factor: Factor to adjust the class weights.

    Returns:
    - class_weights: Tensor containing the calculated class weights.
    """
    class_counts = data['label'].value_counts().sort_index().values
    total_samples = len(data)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights *= weight_factor  # Adjust the weights by multiplying with a factor
    return torch.tensor(class_weights, dtype=torch.float)


def main():
    """
    Main function to load data, create datasets and dataloaders, initialize the model,
    and train and evaluate the model.
    """
    train_data = load_data('data/train.csv')
    val_data = load_data('data/val.csv')
    test_data = load_data('data/test.csv')

    # Print intersections to check for data leakage
    print(set(train_data['text']).intersection(set(val_data['text'])))
    print(set(train_data['text']).intersection(set(test_data['text'])))
    print(set(val_data['text']).intersection(set(test_data['text'])))

    class_weights = calculate_class_weights(train_data)
    print(f"Class weights: {class_weights}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = ABSADataset(train_data, tokenizer)
    val_dataset = ABSADataset(val_data, tokenizer)
    test_dataset = ABSADataset(test_data, tokenizer)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize DistilBERT model
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = ABSAModel(distilbert_model)

    # Select device (MPS for Apple Silicon, otherwise CPU)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device chosen: {device}')

    model = train_model(model, train_loader, val_loader, class_weights, device)

    predictions = predict(model, test_loader, device)

    test_data['predictions'] = predictions
    print(test_data)


if __name__ == "__main__":
    main()
