# Neural Network Models and Data Analysis

This repository contains Python code demonstrating various aspects of neural network models, data preprocessing, and analysis using `scikit-learn`, `imbalanced-learn`, and `tensorflow.keras`.

## Description

The repository is structured as follows:

### Neural Network Models

#### MLP Classifier and Regression

The repository showcases the usage of `MLPClassifier` and `MLPRegressor` from `sklearn.neural_network`. The classifiers are trained on various datasets such as `load_diabetes` and `load_digits`. Evaluation metrics such as accuracy scores, confusion matrices, and classification reports are used to assess model performance.

#### Keras Sequential Models

Demonstrates the creation of various neural network architectures using `tensorflow.keras.Sequential`. Different configurations of hidden layers and activation functions are explored and analyzed using `Sequential` models.

### Data Preprocessing and Analysis

The repository includes data preprocessing techniques such as:

- Scaling data using `StandardScaler` from `sklearn.preprocessing`.
- Handling missing values and cleaning data using pandas.
- Feature engineering to extract company names from a car dataset.
- Replacing misspelled or incorrect company names for consistency.
- Calculating correlations between features and visualizing them using heatmaps.
- Creating scatter plots to visualize relationships between specific features and the target variable (`price`).

### File Structure

- `neural_networks.ipynb`: Jupyter notebook containing neural network model implementations and evaluations.
- `data_preprocessing.ipynb`: Notebook focusing on data preprocessing and analysis.
- `confusion_matrix.png`: Saved confusion matrix heatmap.
- `requirements.txt`: Python packages and versions required to run the code.

## Instructions

### Setup

1. Install the required packages:
