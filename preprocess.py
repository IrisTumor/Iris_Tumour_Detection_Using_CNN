import os
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet50

# Set TensorFlow options
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def get_data_generators(train_dir, test_dir, batch_size=32):
    # Normalize to zero mean using ResNet50 preprocessing
    preprocess_fn = resnet50.preprocess_input

    # Data Augmentation for training set
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,  # Use ResNet50 preprocessing
        rotation_range=20,  # Moderate rotation
        width_shift_range=0.1,  # Slight width shift
        height_shift_range=0.1,  # Slight height shift
        shear_range=0.15,  # Moderate shear
        zoom_range=0.15,  # Moderate zoom
        horizontal_flip=True,  # Horizontal flip
        brightness_range=[0.9, 1.1],  # Subtle brightness changes
    )

    # Test set normalization only
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    # Load images from directories (Train)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Resize to fit ResNet50 input
        batch_size=batch_size,
        class_mode='binary',  # Binary classification
        shuffle=True  # Shuffle for better generalization
    )

    # Load images from directories (Test)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),  # Resize to fit ResNet50 input
        batch_size=batch_size,
        class_mode='binary',  # Binary classification
        shuffle=False  # Do not shuffle for evaluation
    )

    return train_generator, test_generator


# Class weights calculation (to handle class imbalance)
def compute_class_weights(generator):
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',  # Balance class weights
        classes=np.unique(generator.classes),  # Unique classes in the generator
        y=generator.classes  # The class labels
    )
    return dict(enumerate(class_weights))
