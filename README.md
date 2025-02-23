# Real-Time Hand Gesture Recognition using Deep Learning

This repository contains a deep learning model for real-time hand gesture recognition, trained to classify 24 American Sign Language (ASL) alphabets (A-Y, excluding J and Z). The system leverages TensorFlow/Keras and OpenCV for real-time inference via webcam.

## Features
- **24 ASL Classes**: Recognizes static gestures for letters A-Y (excluding J and Z).
- **CNN Architecture**: Uses a Convolutional Neural Network (CNN) with dropout for robust performance.
- **High Accuracy**: Achieves 99% training accuracy and 93.9% test accuracy.
- **Webcam Integration**: Real-time predictions using OpenCV and Google Colab's webcam capture.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- scikit-learn
- Jupyter/Google Colab

Install dependencies:
```bash
pip install tensorflow opencv-python numpy scikit-learn
```

## Dataset
The model was trained on the [Hand Sign Images Dataset](https://www.kaggle.com/datasets/ash2703/handsignimages) from Kaggle:
- **24 classes** (ASL letters A-Y)
- **Train**: 27,455 images
- **Test**: 7,172 images

## Model Architecture
```python
Model: "Sequential"
┌─────────────────────────────────┬─────────────────────────────┐
│ Layer (type)                    │ Output Shape                │
├─────────────────────────────────┼─────────────────────────────┤
│ Conv2D (32 filters)             │ (None, 26, 26, 32)          │
│ MaxPooling2D                    │ (None, 13, 13, 32)          │
│ Conv2D (64 filters)             │ (None, 11, 11, 64)          │
│ MaxPooling2D                    │ (None, 5, 5, 64)            │
│ Flatten                         │ (None, 1600)                │
│ Dense (128 units)               │ (None, 128)                 │
│ Dropout (50%)                   │ (None, 128)                 │
│ Dense (24 units, softmax)       │ (None, 24)                  │
└─────────────────────────────────┴─────────────────────────────┘
```

## Usage

### 1. Training
```python
# Load data
train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_directory, target_size=(28,28), batch_size=32, class_mode='categorical'
)

# Train model
model.fit(train_generator, epochs=25, validation_data=test_generator)
```

### 2. Real-Time Prediction (Google Colab)
```python
from IPython.display import display, Javascript
# Webcam capture and prediction code included in notebook
```

1. Run the `take_photo()` function to capture a hand gesture via webcam.
2. The model preprocesses the image (resize to 28x28, normalize pixels).
3. Predictions are displayed with the inferred ASL letter.

## Results
| Metric        | Value   |
|---------------|---------|
| Training Acc  | 99.02%  |
| Test Acc      | 93.91%  |
| Test Loss     | 0.4557  |

## Limitations
- Static gestures only (no support for dynamic letters like J/Z).
- Performance may vary under low-light conditions.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Dataset by [Aashay Sachdeva](https://www.kaggle.com/ash2703)
- Built with TensorFlow and OpenCV
