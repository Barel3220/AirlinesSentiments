import torch
from torch.utils.data import Dataset


class ABSADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        """
        Initialize the ABSA Dataset.

        Parameters:
        - data: DataFrame containing the text and labels.
        - tokenizer: Tokenizer to convert text into tokens.
        - max_len: Maximum length of the tokenized sequences (default: 64).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the tokenized text and label for a given index.

        Parameters:
        - idx: Index of the sample to retrieve.

        Returns:
        - A dictionary containing 'input_ids', 'attention_mask', and 'label'.
        """
        # Extract text and label from the DataFrame
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        # Tokenize the text
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            max_length=self.max_len,  # Pad or truncate text to max_len
            padding='max_length',  # Pad to the maximum length
            truncation=True,  # Truncate if the text is longer than max_len
            return_attention_mask=True,  # Return attention mask
            return_tensors='pt',  # Return PyTorch tensors
        )

        # Return a dictionary with the tokenized inputs and the label
        return {
            'input_ids': inputs['input_ids'].flatten(),  # Flatten to 1D tensor
            'attention_mask': inputs['attention_mask'].flatten(),  # Flatten to 1D tensor
            'label': torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        }
