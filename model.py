from torch import nn


class ABSAModel(nn.Module):
    def __init__(self, distilbert_model, num_labels=3, dropout_rate=0.3):
        """
        Initialize the ABSA model with a pre-trained DistilBERT model and additional layers.

        Parameters:
        - distilbert_model: Pre-trained DistilBERT model.
        - num_labels: Number of output labels (default: 3 for positive, neutral, negative).
        - dropout_rate: Dropout rate for regularization (default: 0.3).
        """
        super(ABSAModel, self).__init__()
        self.distil_bert = distilbert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.distil_bert.config.hidden_size, num_labels)
        self.layer_norm = nn.LayerNorm(self.distil_bert.config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the model.

        Parameters:
        - input_ids: Tensor containing input token IDs.
        - attention_mask: Tensor containing attention masks to differentiate between real tokens and padding tokens.

        Returns:
        - logits: Output logits for each class.
        """
        # Obtain the last hidden state from DistilBERT
        outputs = self.distil_bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # Use the [CLS] token's representation

        # Apply dropout for regularization
        cls_output = self.dropout(cls_output)

        # Apply layer normalization
        cls_output = self.layer_norm(cls_output)

        # Apply activation function (ReLU)
        cls_output = self.activation(cls_output)

        # Get logits from the classifier layer
        logits = self.classifier(cls_output)

        return logits
