import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import get_data_generators, compute_class_weights

# Build Model Function
def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Paths to your dataset
train_dir = 'D:/iris-tumour-detection/dataset/train'
test_dir = 'D:/iris-tumour-detection/dataset/test'

# Data Generators for training and testing data
train_generator, test_generator = get_data_generators(train_dir, test_dir)

# Compute Class Weights to handle class imbalance
class_weights = compute_class_weights(train_generator)

# Initialize the model
model = build_model()

# Early Stopping and Model Checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    'models/best_model.keras',  # Updated file extension to `.keras`
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint]
)

# Save the final model
model.save('models/custom_cnn_final.keras')  # Updated file extension to `.keras`

# Model Summary
model.summary()
