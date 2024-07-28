import torch
from torch import nn, optim
from tqdm import tqdm
from evaluate import evaluate_model
from plotter import plot_confusion_matrix
import json


def train_model(model, train_loader, val_loader, class_weights, device, epochs=10, learning_rate=5e-6):
    """
    Train the model with the given training and validation data loaders.

    Parameters:
    - model: The model to be trained.
    - train_loader: DataLoader for the training data.
    - val_loader: DataLoader for the validation data.
    - class_weights: Tensor containing class weights for handling class imbalance.
    - device: Device to run the training on (e.g., 'cpu', 'cuda').
    - epochs: Number of epochs to train (default: 10).
    - learning_rate: Learning rate for the optimizer (default: 5e-6).

    Returns:
    - model: The trained model.
    """
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3  # Number of epochs to wait before early stopping

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', colour='green')

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(float(avg_train_loss))
        print(f'Training loss: {avg_train_loss:.4f}')

        # Evaluate the model on the validation set
        val_loss, val_accuracy, val_f1, cm = evaluate_model(model, val_loader, loss_fn, device)
        history['val_loss'].append(float(val_loss))
        history['val_accuracy'].append(float(val_accuracy))
        history['val_f1'].append(float(val_f1))
        print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} '
              f'| Validation F1 Score: {val_f1:.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.bin')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            plot_confusion_matrix(cm, ['negative', 'neutral', 'positive'])
            print("Early stopping triggered")
            break

    # Save the training history
    with open('data/history.json', 'w') as f:
        json.dump(history, f)

    print("Training complete")

    return model
