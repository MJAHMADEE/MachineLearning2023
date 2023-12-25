# Radial Basis Function Network (RBF)

This repository contains an implementation of a Radial Basis Function Network (RBF) in Python.

## Overview

The RBF class implements functionalities to create, train, and use a Radial Basis Function Network for classification tasks. The network architecture includes:

- Initialization and fitting of centers using KMeans clustering
- Training the RBF network with specified parameters
- Prediction capabilities
- Monitoring loss and accuracy during training
- Visualization of training metrics (Loss and Accuracy)

## Usage

### Instantiating the RBF Model

```python
from RBF import RBF

# Create an RBF instance
model = RBF('My_First_RBF')
```

### Training the RBF Model
   ```python
   # Fit the model with input data X and one-hot encoded labels OHY
   model.fit(X, OHY, nH=3, nEpoch=50, lr=1e-3)
   ```
### Making Predictions
   ```python
   # Obtain predictions for input data X
   predictions = model.predict(X)
   ```
### Visualizing Training Metrics

   ```python
# Access training history for Loss and Accuracy
losses = model.history['loss']
accuracies = model.history['accuracy']

# Plotting Loss and Accuracy
# (Two separate plots side by side)
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.plot(losses, lw=1.2, c='crimson', marker='o', ms=3)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(accuracies, lw=1.2, c='teal', marker='o', ms=3)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()
   ```

