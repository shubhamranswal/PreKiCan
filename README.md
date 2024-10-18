## PreKiCan: A PARAM Utkarsh-Powered Model for Early Kidney Cancer Identification

### Project Overview

PreKiCan is a convolutional neural network (CNN) model designed to detect kidney cancer from x-ray images. This project leverages deep learning to enhance the accuracy and efficiency of medical diagnostics, aiming to provide faster, non-invasive diagnostic tools that can potentially save lives through early detection and treatment of kidney cancer.

### Features

- **Deep Learning Model**: A CNN architecture designed specifically for binary classification of x-ray images into cancerous and non-cancerous categories.
- **Supercomputing**: Trained on the PARAM Utkarsh Supercomputer, providing the computational power required for handling large datasets and complex model architectures.
- **Robust Training**: Includes extensive data preprocessing, augmentation, and the use of advanced optimization techniques to ensure high accuracy and reliability.
- **Easy Deployment**: The trained model can be easily loaded and used for predictions on new x-ray images using provided scripts.

### Data Preparation

1. **Load Dataset**: Import the x-ray image dataset, which includes 10,000 images (5,000 cancerous and 5,000 non-cancerous).
2. **Preprocessing**:
   - Resize images to 150x150 pixels for consistency.
   - Normalize pixel values to the range 0 to 1.
   - Apply data augmentation techniques such as rotation, zoom, and horizontal flipping to increase dataset variability and model robustness.

### Model Architecture

The PreKiCan model consists of several layers designed to extract and learn features from x-ray images:

1. **Conv2D Layers**: Apply filters of size 3x3 with ReLU activation.
   - First layer: 32 filters.
   - Second layer: 64 filters.
   - Third and fourth layers: 128 filters each.
2. **MaxPooling2D Layers**: Reduce dimensionality with a 2x2 pool size after each Conv2D layer.
3. **Flatten Layer**: Convert 3D feature maps to 1D feature vectors.
4. **Dense Layers**:
   - First dense layer: 512 neurons with ReLU activation.
   - Output layer: 1 neuron with sigmoid activation for binary classification.

### Model Training

- **Epochs**: 250
- **Batch Size**: 32
- **Optimizer**: Adam optimizer for efficient gradient descent and adaptive learning rate.
- **Loss Function**: Binary crossentropy, suitable for binary classification tasks.
- **Validation**: A portion of the dataset is reserved for validation to monitor and prevent overfitting.

### Model Evaluation and Prediction

- **Save Trained Model**: The model weights and architecture are saved in `model.h5`.
- **Load Model for Prediction**: Use `predict.py` to load the trained model and make predictions.
- **Make Predictions**: The script preprocesses new x-ray images and predicts whether they are cancerous or non-cancerous.

### Usage

1. **Training the Model**:
   ```sh
   python main.py
   ```
2. **Making Predictions**:
   ```sh
   python predict.py
   ```

### Conclusion

The PreKiCan model demonstrates the application of advanced deep learning techniques in the medical field, specifically for the early detection of kidney cancer. By utilizing the computational power of the PARAM Utkarsh Supercomputer, this project aims to contribute significantly to non-invasive medical diagnostics, improving early detection rates and potentially saving lives.
