# Spam Email Classifier

This project implements a pipeline to classify emails as spam or non-spam using machine learning techniques. It includes data preprocessing, feature extraction, and model training with hyperparameter tuning.

## Features

- Downloads and processes the [SpamAssassin public corpus](http://spamassassin.apache.org/old/publiccorpus/).
- Converts emails into feature vectors using custom transformers.
- Builds a classification model using a Gradient Boosting Classifier.
- Evaluates model performance with metrics like accuracy, precision, recall, and F1-score.
- Includes grid search for hyperparameter optimization.

## Installation

### Prerequisites

Make sure you have the following installed:

- Python (>= 3.7)
- Required libraries:
  - `numpy`
  - `scikit-learn`
  - `nltk`
  - `matplotlib`
  - `urlextract`

Install the required libraries using:

```bash
pip install numpy scikit-learn nltk matplotlib urlextract
```

### Download NLTK Data

Run the following command to download necessary NLTK resources:

```python
import nltk
nltk.download('punkt')
```

## Usage

1. Clone the repository or download the script.
2. Ensure the working directory contains the `datasets` folder where the spam and ham datasets will be downloaded.
3. Run the script to execute the pipeline:

```bash
python spam_classifier.py
```

## Project Structure

### Key Functions and Classes

- **`fetch_spam_data()`**: Downloads and extracts spam and ham datasets.
- **`email_to_text(email)`**: Extracts plain text from email content.
- **`GetCounterOfWords`**: Custom transformer to preprocess and tokenize email text. It processes the email content by:
  1. Extracting text from HTML or plain text parts of the email.
  2. Lowercasing all text for uniformity.
  3. Replacing URLs with a placeholder (e.g., "URL").
  4. Removing numeric values by replacing them with a placeholder (e.g., "NUMBER").
  5. Removing special symbols and punctuation to retain only meaningful words.
  6. Using stemming (via NLTK's `PorterStemmer`) to reduce words to their root forms, enhancing generalization.
  The output is a `Counter` object containing word frequencies for each email.

- **`WordCounterToVectors`**: Converts tokenized words into sparse feature vectors. It works as follows:
  1. Analyzes the word counts across all emails and selects the most common words based on a predefined vocabulary size (e.g., 1000 words).
  2. Maps each word to a unique index in the vocabulary.
  3. Constructs a sparse matrix where each row corresponds to an email and each column represents the count of a specific word from the vocabulary.

- **`our_pipe`**: Pipeline combining text preprocessing and vectorization.
- **`GridSearchCV`**: Performs hyperparameter tuning for the classifier.

### Model Training and Evaluation

The script trains a `GradientBoostingClassifier` using the processed data and evaluates its performance with metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results

The script outputs the following evaluation metrics:

- **Accuracy**: Overall percentage of correct predictions.
- **Precision**: Percentage of correctly identified spam emails.
- **Recall**: Percentage of actual spam emails identified.
- **F1-score**: Harmonic mean of precision and recall.

Example output:
```
accuracy -> 0.98
--------------------------------------------------
precision -> 0.95
--------------------------------------------------
recall -> 0.97
--------------------------------------------------
f1_score -> 0.96
--------------------------------------------------
|confusion_matrix
[[890   10]
 [ 15  885]]
```

## Contributing

Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.


## Developer
***Youssef khaled***

