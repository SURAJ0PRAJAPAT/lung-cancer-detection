This is a Pneumonia Classification workflow using a Convolutional Neural Network (CNN) in Keras. The workflow consists of several key steps:

1. Import Necessary Libraries
The code starts by importing essential libraries like:

numpy, pandas: Data manipulation
matplotlib.pyplot, seaborn: Data visualization
cv2: Image processing
os: File handling
keras and sklearn: Deep learning and model evaluation
2. Load and Preprocess Data
a. Define Labels and Image Size
labels = ['PNEUMONIA', 'NORMAL']
Images are resized to 150×150 pixels.
b. Load Images
The function get_training_data(data_dir):

Iterates through the train, test, and val directories.
Reads grayscale images using cv2.imread().
Resizes them to 150×150 pixels.
Assigns labels (0 = Pneumonia, 1 = Normal).
Returns a NumPy array of images and their labels.
c. Visualizing Class Distribution
Uses sns.countplot() to plot class distribution.
d. Splitting Features and Labels
Extracts images (x_train, x_test, x_val) and labels (y_train, y_test, y_val).
Normalizes pixel values to [0,1] by dividing by 255.
Reshapes images to (150,150,1) to match CNN input.
3. Data Augmentation
To improve generalization, ImageDataGenerator() applies:

Rotation (30 degrees)
Zoom (20%)
Width & height shift (10%)
Horizontal flip
This generates more diverse images to prevent overfitting.

4. Define CNN Model
A Sequential CNN model with:

Convolutional Layers (Conv2D): Extracts features using filters (32, 64, 128, 256).
Batch Normalization: Normalizes activations.
MaxPooling (MaxPool2D): Reduces dimensions.
Dropout: Prevents overfitting (10-20%).
Flatten Layer: Converts feature maps to a 1D vector.
Dense Layers:
128 neurons (ReLU activation).
1 neuron (Sigmoid activation) for binary classification.
Compilation
Optimizer: rmsprop
Loss Function: binary_crossentropy
Metric: accuracy
5. Train the Model
The model is trained for 12 epochs using model.fit() with:

datagen.flow() for data augmentation.
ReduceLROnPlateau() to adjust learning rate if validation accuracy stops improving.
6. Evaluate Performance
Loss & Accuracy on test data: model.evaluate(x_test, y_test).
Plot Training vs. Validation Accuracy & Loss using matplotlib.
7. Predictions & Model Evaluation
Uses model.predict_classes(x_test) to get predictions.
Confusion Matrix: Evaluates model errors.
Classification Report: Shows precision, recall, and F1-score.
Visualizing Predictions:
Correct Predictions: Plots first 6 correctly classified images.
Incorrect Predictions: Plots first 6 misclassified images.
Final Summary
Load and preprocess X-ray images.
Apply data augmentation.
Train a CNN for pneumonia detection.
Evaluate performance using accuracy, confusion matrix, and sample predictions.
