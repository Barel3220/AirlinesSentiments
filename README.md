# Airlines Sentiment Analysis

## Overview

This project aims to perform Aspect-Based Sentiment Analysis (ABSA) on tweets related to airlines. The goal is to classify the sentiments of tweets into positive, neutral, or negative categories using a pre-trained DistilBERT model fine-tuned on the specific dataset.

This project is developed and written by Angelina (Lina) Rozentzvig and Barel Hatuka.

## Project Structure
```
.
├── data/
│   ├── airline-sentiment-data.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── history.json
├── models/
│   └── best_model.bin
├── plots/
│   └── confusion_matrix.png
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plotter.py
│   ├── main.py
├── README.md
└── requirements.txt
```

## Installation

requirements.txt
```
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
torch==1.9.0
transformers==4.10.0
tqdm==4.62.3
matplotlib==3.4.3
seaborn==0.11.2
```

command to install all the requirements
```sh
pip install -r requirements.txt
```

## Dataset

The dataset consists of tweets related to various airlines. Each tweet is labeled with a sentiment: positive, neutral, or negative. The dataset is preprocessed and split into training, validation, and test sets.

## Preprocessing

The preprocess.py script performs the following steps:

    •	Load the raw dataset.
    •	Clean the text by removing airline mentions, URLs, and non-alphanumeric characters.
    •	Map sentiment labels to numerical values.
    •	Split the data into training, validation, and test sets.
    •	Save the processed datasets to CSV files.

## Model

The model is implemented in model.py and uses a pre-trained DistilBERT model with additional layers for classification:

	•	Dropout layer for regularization.
	•	Layer normalization for stable learning.
	•	Linear layer for sentiment classification.

## Training

The train.py script handles the training process:

	•	Loads the training and validation data.
	•	Initializes the model, tokenizer, optimizer, and loss function.
	•	Trains the model with early stopping based on validation loss.
	•	Saves the best model and training history.

## Evaluation

The evaluate.py script evaluates the model on the validation set:

	•	Computes the loss, accuracy, F1-score, and confusion matrix.
	•	Prints the classification report and confusion matrix.

The evaluation is integrated within the training loop to monitor performance after each epoch.

## Plotting

The plotter.py script provides functions to visualize:

	•	Data distribution by airline.
	•	Sentiment distribution over time.
	•	Training and validation metrics.
	•	Confusion matrix.

## Main Script

The main.py script integrates all components:

	•	Loads the preprocessed data.
	•	Trains the model and evaluates its performance.
	•	Makes predictions on the test set.
	•	Prints the results.

## Results

The results include:

	•	Model performance metrics (loss, accuracy, F1-score).
	•	Confusion matrix for visualizing prediction errors.
	•	Data distribution plots and sentiment trends.

## Conclusion

This project demonstrates the application of ABSA using a pre-trained transformer model. It covers data preprocessing, model training, evaluation, and visualization of results, providing a comprehensive pipeline for sentiment analysis.
