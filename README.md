# Constellation Recognition Model (CNN)

## Overview

The **Constellation Recognition Model** is a Convolutional Neural Network (CNN) designed to recognize and classify star constellations from images of the night sky. This model takes as input an image of a star field, identifies the stars, and classifies the specific constellation to which the star pattern belongs. It leverages deep learning techniques to process star patterns and generate accurate results even under various noise conditions.

This project is ideal for astronomy enthusiasts, machine learning practitioners, or anyone interested in using neural networks for image recognition tasks related to the night sky.

---

## Features

- **CNN-based Architecture**: The model uses convolutional layers to extract spatial hierarchies of features from star fields.
- **Star Constellation Classification**: Classifies an image of stars into one of several predefined constellations.
- **Robust Performance**: Handles noisy star fields and partial occlusions, making it versatile for real-world data.
- **Training with Augmented Data**: The model is trained with various augmentations, such as rotation and scaling, to increase robustness and generalization.

---

## Table of Contents

1. [Installation](#installation)
2. [Model Overview](#model-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Training the Model](#training-the-model)
5. [Model Evaluation](#model-evaluation)
6. [Usage](#usage)
7. [Example](#example)
8. [Dependencies](#dependencies)
9. [Contributing](#contributing)
10. [License](#license)

---

## Installation

To get started with this project, you will need to clone this repository and install the required dependencies.

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/constellation-recognition-cnn.git
    cd constellation-recognition-cnn
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

---

## Model Overview

The model architecture is based on a typical CNN structure with several convolutional layers followed by fully connected layers. Key components include:

- **Convolutional Layers**: To extract features like star locations and patterns.
- **Max Pooling Layers**: To reduce dimensionality and retain key features.
- **Fully Connected Layers**: To classify the extracted features into constellations.
- **Softmax Output Layer**: To output the probability distribution over possible constellations.

### Hyperparameters

- Number of convolutional layers: 3
- Kernel size: (3, 3)
- Number of fully connected layers: 2
- Dropout rate: 0.5
- Optimizer: Adam
- Loss function: Categorical Crossentropy

---

## Data Preprocessing

### Dataset

The model is trained on a dataset of star field images labeled with the corresponding constellation names. Each image contains a set of stars that correspond to a particular constellation. The dataset should be structured as follows:

```
/data
    /train
        /constellation_name_1
            image_1.jpg
            image_2.jpg
            ...
        /constellation_name_2
            image_1.jpg
            ...
    /test
        /constellation_name_1
            image_1.jpg
            ...
        /constellation_name_2
            image_1.jpg
            ...
```

### Image Preprocessing

1. **Resizing**: Images are resized to a fixed dimension (e.g., 224x224 pixels) to standardize input sizes.
2. **Normalization**: Pixel values are normalized to the range [0, 1].
3. **Augmentation**: Data augmentation techniques, such as random rotations, flipping, and scaling, are applied to improve generalization.

---

## Training the Model

To train the model, use the following command:

```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

- `epochs`: Number of epochs for training.
- `batch_size`: Batch size for each training step.
- `learning_rate`: Learning rate for the optimizer.

Training logs will be saved in the `logs/` directory.

---

## Model Evaluation

After training, you can evaluate the model’s performance on the test dataset using the following command:

```bash
python evaluate.py --model checkpoint/model.h5 --test_data data/test
```

This will output the accuracy, precision, recall, and F1-score of the model on the test data.

---

## Usage

Once the model is trained and evaluated, you can use it to predict constellations from new images of star fields. The following command will load the model and make a prediction on a new image:

```bash
python predict.py --model checkpoint/model.h5 --image path/to/your/image.jpg
```

The output will be the predicted constellation label and its associated probability.

---

## Example

Here’s an example of how to use the trained model to classify a new star field image:

1. **Train the model**:
    ```bash
    python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
    ```

2. **Evaluate the model**:
    ```bash
    python evaluate.py --model checkpoint/model.h5 --test_data data/test
    ```

3. **Make predictions**:
    ```bash
    python predict.py --model checkpoint/model.h5 --image data/test/andromeda/image_1.jpg
    ```

The output will display:
```
Predicted Constellation: Andromeda
Probability: 0.92
```

---

## Dependencies

The following Python libraries are required:

- `tensorflow` (>=2.5)
- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `pillow`
- `tqdm`

You can install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Contributing

We welcome contributions to improve the Constellation Recognition Model. To contribute:

1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Open a pull request describing your changes and why they improve the project.

Please ensure your code follows the PEP 8 style guide, and write tests to cover new features or bug fixes
