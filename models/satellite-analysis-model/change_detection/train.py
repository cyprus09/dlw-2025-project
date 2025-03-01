import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from model import build_forest_detection_model
import ssl

# Disable SSL verification temporarily
ssl._create_default_https_context = ssl._create_unverified_context


def load_dataset(data_dir):
    """
    Load dataset from X_features.npy and y_masks.npy

    Args:
        data_dir (str): Path to directory containing dataset

    Returns:
        tuple: X (features), y (masks)
    """
    # Load features
    X_path = os.path.join(data_dir, "X_features.npy")
    y_path = os.path.join(data_dir, "y_masks.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Features file not found at {X_path}")

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Masks file not found at {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    return X, y


def augment_dataset(X, y, augmentation_factor=3):
    """
    Perform data augmentation on the dataset

    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Input masks
        augmentation_factor (int): Multiplicative factor for dataset augmentation

    Returns:
        tuple: Augmented X and y
    """
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        augmented_X.append(X[i])
        augmented_y.append(y[i])

        # Additional augmentations
        for _ in range(augmentation_factor - 1):
            # Random horizontal flip
            if np.random.rand() > 0.5:
                aug_image = np.fliplr(X[i])
                aug_mask = np.fliplr(y[i])
            else:
                aug_image = X[i]
                aug_mask = y[i]

            # Random rotation
            rotation_angle = np.random.randint(-30, 30)
            aug_image = tf.keras.preprocessing.image.random_rotation(
                aug_image, rotation_angle
            )
            aug_mask = tf.keras.preprocessing.image.random_rotation(
                aug_mask, rotation_angle
            )

            augmented_X.append(aug_image)
            augmented_y.append(aug_mask)

    return np.array(augmented_X), np.array(augmented_y)


def plot_augmentation_examples(
    X_orig, y_orig, X_aug, y_aug, num_examples=3, save_path=None
):
    """
    Visualize original and augmented images

    Args:
        X_orig (np.ndarray): Original features
        y_orig (np.ndarray): Original masks
        X_aug (np.ndarray): Augmented features
        y_aug (np.ndarray): Augmented masks
        num_examples (int): Number of examples to plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(15, 5 * num_examples))

    for i in range(min(num_examples, len(X_orig))):
        # Original feature
        plt.subplot(num_examples, 4, 4 * i + 1)
        plt.imshow(X_orig[i])
        plt.title(f"Original Feature {i+1}")
        plt.axis("off")

        # Original mask
        plt.subplot(num_examples, 4, 4 * i + 2)
        plt.imshow(y_orig[i], cmap="gray")
        plt.title(f"Original Mask {i+1}")
        plt.axis("off")

        # Augmented feature
        plt.subplot(num_examples, 4, 4 * i + 3)
        plt.imshow(X_aug[4 * i])
        plt.title(f"Augmented Feature {i+1}")
        plt.axis("off")

        # Augmented mask
        plt.subplot(num_examples, 4, 4 * i + 4)
        plt.imshow(y_aug[4 * i], cmap="gray")
        plt.title(f"Augmented Mask {i+1}")
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def main():
    # Configuration
    DATA_DIR = "../../../data/processed/processed_ndvi_rgb/image_datasets"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    MODEL_DIR = "models/change_patterns/models"
    LOG_DIR = "models/change_patterns/logs"

    # Print current directory for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Using data directory: {DATA_DIR}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Validation directory: {VAL_DIR}")

    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Training parameters
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4

    # Load datasets first to determine input shape
    print("Loading training dataset...")
    X_train, y_train = load_dataset(TRAIN_DIR)

    # Dynamically set input shape
    INPUT_SHAPE = X_train.shape[1:]
    print(f"Input shape: {INPUT_SHAPE}")

    print(f"Original training set: {X_train.shape}, {y_train.shape}")

    # Print class distribution information
    forest_pixel_ratio = np.mean(y_train)
    print(f"Forest pixel percentage: {forest_pixel_ratio * 100:.2f}%")
    print(
        f"Class distribution - Forest: {forest_pixel_ratio:.4f}, Non-forest: {1-forest_pixel_ratio:.4f}"
    )

    print("Augmenting training dataset...")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, 3)
    print(f"Augmented training set: {X_train_aug.shape}, {y_train_aug.shape}")

    # Plot augmentation examples
    aug_viz_path = os.path.join(LOG_DIR, "augmentation_examples.png")
    plot_augmentation_examples(
        X_train,
        y_train,
        X_train_aug,
        y_train_aug,
        num_examples=3,
        save_path=aug_viz_path,
    )
    print(f"Augmentation examples saved to {aug_viz_path}")

    print("Loading validation dataset...")
    X_val, y_val = load_dataset(VAL_DIR)
    print(f"Validation set: {X_val.shape}, {y_val.shape}")

    # Build model
    print("Building model...")
    model = build_forest_detection_model(input_shape=INPUT_SHAPE, use_pretrained=True)
    model.summary()

    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f"change_patterns_{timestamp}.h5"),
        save_best_only=True,
        monitor="val_iou_threshold_0.1",  # Use lower threshold IoU for monitoring
        mode="max",
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou_threshold_0.1",  # Use lower threshold IoU for early stopping
        patience=15,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(LOG_DIR, f"change_patterns_{timestamp}"), histogram_freq=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks = [model_checkpoint, early_stopping, tensorboard_callback, reduce_lr]

    # Train model
    print(f"Training model for up to {EPOCHS} epochs with early stopping...")
    history = model.fit(
        X_train_aug,
        y_train_aug,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Plot and save training history
    history_path = os.path.join(LOG_DIR, f"training_history_{timestamp}.png")
    plot_training_history(history, save_path=history_path)
    print(f"Training history saved to {history_path}")

    # Visualize sample predictions
    viz_path = os.path.join(LOG_DIR, f"predictions_visualization_{timestamp}.png")
    visualize_predictions(model, X_val, y_val, num_samples=5, save_path=viz_path)
    print(f"Prediction visualization saved to {viz_path}")

    # Save the final model
    final_model_path = os.path.join(MODEL_DIR, f"change_patterns_final{timestamp}.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save training history to file
    np.save(os.path.join(LOG_DIR, f"training_history_{timestamp}.npy"), history.history)

    print("Training complete!")


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics

    Args:
        history (tf.keras.callbacks.History): Training history
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Available metrics
    metrics = list(history.history.keys())
    print("Available metrics:", metrics)

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(2, 2, 2)
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    # Precision plot
    plt.subplot(2, 2, 3)
    if "precision" in history.history:
        plt.plot(history.history["precision"], label="Training Precision")
        if "val_precision" in history.history:
            plt.plot(history.history["val_precision"], label="Validation Precision")
        plt.title("Precision")
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.legend()

    # Recall plot
    plt.subplot(2, 2, 4)
    if "recall" in history.history:
        plt.plot(history.history["recall"], label="Training Recall")
        if "val_recall" in history.history:
            plt.plot(history.history["val_recall"], label="Validation Recall")
        plt.title("Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_predictions(model, X_val, y_val, num_samples=5, save_path=None):
    """
    Visualize model predictions on validation set

    Args:
        model (tf.keras.Model): Trained model
        X_val (np.ndarray): Validation images
        y_val (np.ndarray): Validation masks
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the plot
    """
    # Predict on validation set
    predictions = model.predict(X_val[:num_samples])

    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Original image
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(X_val[i])
        plt.title(f"Original Image {i+1}")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(y_val[i], cmap="gray")
        plt.title(f"Ground Truth Mask {i+1}")
        plt.axis("off")

        # Predicted mask
        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(predictions[i].squeeze(), cmap="gray")
        plt.title(f"Predicted Mask {i+1}")
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    print("Starting forest detection training...")
    main()
