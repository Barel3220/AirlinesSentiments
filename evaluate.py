import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_model(model, data_loader, loss_fn, device):
    """
    Evaluate the model on the given data loader.

    Parameters:
    - model: The model to evaluate.
    - data_loader: DataLoader containing the evaluation data.
    - loss_fn: Loss function to compute the loss.
    - device: Device to run the evaluation on (e.g., 'cpu', 'cuda').

    Returns:
    - avg_loss: Average loss over the evaluation dataset.
    - accuracy: Accuracy of the model on the evaluation dataset.
    - f1: F1-score of the model on the evaluation dataset.
    - cm: Confusion matrix of the model's predictions.
    """
    model.to(device)
    model.eval()

    total_loss = 0
    correct_predictions = 0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(data_loader, desc=f'Evaluating', colour='green')

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.float() / len(data_loader.dataset)

    print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    return avg_loss, accuracy, f1, cm
