# Image Processing and Hamming Network

This repository contains code for image processing and an implementation of a Hamming network for image classification. Below are the key functionalities:

## Functionality 1: Image to Binary Conversion

### `convertImageToBinary(path)`
Converts an image to a binary representation based on pixel intensity.

- `path`: File path to the input image.
- Returns a binary representation where white is represented by -1 and black is represented by 1.

## Functionality 2: Generating Noisy Images

### `generateNoisyImages()`
Generates noisy images based on given images.

- Modifies images by adding noise and saving them.
- Utilizes `getNoisyBinaryImage()`.

### `getNoisyBinaryImage(input_path, output_path)`
Adds noise to an image and saves it as a new file.

- `input_path`: File path to the input image.
- `output_path`: File path to save the noisy image.

### `getNoisyBinaryImage(input_path, output_path, num_missing_points, conversion_percentage)`
Modifies images with missing points and black-to-white conversion.

- `input_path`: File path to the input image.
- `output_path`: File path to save the modified image.
- `num_missing_points`: Number of missing points to generate.
- `conversion_percentage`: Percentage of black pixels to convert to white.

## Functionality 3: Hamming Network Implementation

### Hamming Network for Image Classification
An implementation using a Hamming network to classify an input image among a set of example images.

- `show(matrix)`: Displays a matrix in a formatted manner.
- `change(vector, a, b)`: Transforms a vector into a matrix of specified dimensions.
- `product(matrix, vector, T)`: Multiplies a matrix by a vector.
- `action(vector, T, Emax)`: Activation function to process a vector.
- `mysum(vector, j)`: Calculates the sum of vector values excluding an element.
- `norm(vector, p)`: Calculates the Euclidean norm of the difference between two vectors.
- Determines class based on the highest output value from the Hamming network.

## Example Usage
- Convert images to binary representations.
- Generate noisy images.
- Implement the Hamming network for image classification.

Refer to the code for further details and usage examples.
