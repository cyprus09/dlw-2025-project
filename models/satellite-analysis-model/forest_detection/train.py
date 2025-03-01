import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from model import build_forest_detection_model, custom_iou, make_iou_threshold
import ssl

# Disable SSL verification temporarily
ssl._create_default_https_context = ssl._create_unverified_context

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_dataset(data_dir):
    """Load features and masks from .npy files"""
    try:
        features_path = os.path.join(data_dir, "X_features.npy")
        masks_path = os.path.join(data_dir, "y_masks.npy")

        print(f"Looking for features at: {features_path}")
        print(f"Looking for masks at: {masks_path}")

        if not os.path.exists(features_path) or not os.path.exists(masks_path):
            print(f"Files not found in {data_dir}")
            print(f"Directory contents: {os.listdir(data_dir)}")
            raise FileNotFoundError(f"Dataset files not found in {data_dir}")

        features = np.load(features_path)
        masks = np.load(masks_path)

        print(f"Loaded features shape: {features.shape}")
        print(f"Loaded masks shape: {masks.shape}")

        return features, masks
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def create_data_augmentation_model():
    """Create a sequential model for data augmentation."""
    data_augmentation = tf.keras.Sequential(
        [
            # Apply random rotation
            tf.keras.layers.RandomRotation(0.2, fill_mode="reflect"),
            # Apply random flip
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
            # Apply random zoom
            tf.keras.layers.RandomZoom(
                height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="reflect"
            ),
            # Apply random contrast
            tf.keras.layers.RandomContrast(0.2),
            # Optional: apply a slight brightness variation
            tf.keras.layers.RandomBrightness(0.1),
        ]
    )
    return data_augmentation


def augment_dataset(features, masks, augmentation_factor=3):
    """Augment the dataset by creating multiple augmented versions of each image."""
    data_augmentation = create_data_augmentation_model()

    # Create augmented versions
    augmented_features = [features]  # Start with original features
    augmented_masks = [masks]  # Start with original masks

    for i in range(augmentation_factor):
        print(f"Creating augmentation set {i+1}/{augmentation_factor}...")

        # Apply same augmentation to both features and masks
        # Use a different random seed for each augmentation set
        tf.random.set_seed(42 + i)

        # Apply augmentation
        aug_features = data_augmentation(features, training=True)

        # For masks, we need to handle separately to ensure binary values
        # Apply only geometric transformations to masks
        geometric_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(0.2, fill_mode="reflect"),
                tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42 + i),
                tf.keras.layers.RandomZoom(
                    height_factor=(-0.2, 0.2),
                    width_factor=(-0.2, 0.2),
                    fill_mode="reflect",
                ),
            ]
        )

        # Apply geometric augmentation to masks
        aug_masks = geometric_augmentation(masks, training=True)

        # Threshold masks back to binary values
        aug_masks = tf.cast(aug_masks > 0.5, tf.float32)

        augmented_features.append(aug_features)
        augmented_masks.append(aug_masks)

    # Concatenate all augmented datasets
    augmented_features = np.concatenate(augmented_features, axis=0)
    augmented_masks = np.concatenate(augmented_masks, axis=0)

    return augmented_features, augmented_masks


def plot_augmentation_examples(
    original_features,
    original_masks,
    augmented_features,
    augmented_masks,
    num_examples=3,
    save_path=None,
):
    """Plot examples of original and augmented images."""
    # Select random indices
    indices = np.random.choice(len(original_features), num_examples, replace=False)

    fig, axes = plt.subplots(num_examples, 6, figsize=(20, 4 * num_examples))

    for i, idx in enumerate(indices):
        # Original RGB image
        rgb_img = original_features[idx, :, :, :3]
        axes[i, 0].imshow(np.clip(rgb_img, 0, 1))
        axes[i, 0].set_title("Original RGB")
        axes[i, 0].axis("off")

        # Original NDVI
        ndvi_img = original_features[idx, :, :, 3]
        axes[i, 1].imshow(ndvi_img, cmap="viridis")
        axes[i, 1].set_title("Original NDVI")
        axes[i, 1].axis("off")

        # Original mask
        axes[i, 2].imshow(original_masks[idx, :, :, 0], cmap="viridis")
        axes[i, 2].set_title("Original Mask")
        axes[i, 2].axis("off")

        # Augmented RGB image
        aug_idx = idx + len(original_features)  # First augmented batch
        if aug_idx < len(augmented_features):
            aug_rgb = augmented_features[aug_idx, :, :, :3]
            axes[i, 3].imshow(np.clip(aug_rgb, 0, 1))
            axes[i, 3].set_title("Augmented RGB")
            axes[i, 3].axis("off")

            # Augmented NDVI
            aug_ndvi = augmented_features[aug_idx, :, :, 3]
            axes[i, 4].imshow(aug_ndvi, cmap="viridis")
            axes[i, 4].set_title("Augmented NDVI")
            axes[i, 4].axis("off")

            # Augmented mask
            axes[i, 5].imshow(augmented_masks[aug_idx, :, :, 0], cmap="viridis")
            axes[i, 5].set_title("Augmented Mask")
            axes[i, 5].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_training_history(history, save_path=None):
    """Plot and optionally save training metrics"""
    # Find the IoU metrics in history
    iou_metrics = [
        key for key in history.history.keys() if "iou" in key and "val" not in key
    ]
    val_iou_metrics = [
        key for key in history.history.keys() if "iou" in key and "val" in key
    ]

    # Create figure with appropriate number of subplots
    n_plots = (
        2
        + (1 if iou_metrics else 0)
        + (1 if len(history.history.get("precision", [])) > 0 else 0)
    )
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots))

    # Plot loss
    axes[0].plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plot accuracy
    axes[1].plot(history.history["accuracy"], label="Training Accuracy")
    if "val_accuracy" in history.history:
        axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # Plot IoU if available
    if iou_metrics:
        plot_idx = 2
        for metric in iou_metrics:
            axes[plot_idx].plot(history.history[metric], label=f"Training {metric}")
            val_metric = f"val_{metric}"
            if val_metric in history.history:
                axes[plot_idx].plot(
                    history.history[val_metric], label=f"Validation {metric}"
                )
        axes[plot_idx].set_title("IoU Metrics")
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("IoU")
        axes[plot_idx].legend()
        plot_idx += 1

    # Plot precision and recall
    if "precision" in history.history:
        plot_idx = 3 if iou_metrics else 2
        axes[plot_idx].plot(history.history["precision"], label="Training Precision")
        if "val_precision" in history.history:
            axes[plot_idx].plot(
                history.history["val_precision"], label="Validation Precision"
            )
        axes[plot_idx].plot(history.history["recall"], label="Training Recall")
        if "val_recall" in history.history:
            axes[plot_idx].plot(
                history.history["val_recall"], label="Validation Recall"
            )
        axes[plot_idx].set_title("Precision & Recall")
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Value")
        axes[plot_idx].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_predictions(model, features, masks, num_samples=5, save_path=None):
    """Visualize model predictions vs ground truth with different thresholds"""
    # Get random indices
    indices = np.random.choice(
        len(features), min(num_samples, len(features)), replace=False
    )

    # Get predictions
    predictions = model.predict(features[indices])

    # Plot the results with multiple thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]
    fig, axes = plt.subplots(
        num_samples,
        len(thresholds) + 3,
        figsize=(5 * (len(thresholds) + 3), 4 * num_samples),
    )

    for i, idx in enumerate(indices):
        # Display RGB image
        rgb_img = features[idx, :, :, :3]
        axes[i, 0].imshow(np.clip(rgb_img, 0, 1))
        axes[i, 0].set_title("RGB Image")
        axes[i, 0].axis("off")

        # Display NDVI
        ndvi_img = features[idx, :, :, 3]
        axes[i, 1].imshow(ndvi_img, cmap="viridis")
        axes[i, 1].set_title("NDVI")
        axes[i, 1].axis("off")

        # Display ground truth mask
        axes[i, 2].imshow(masks[idx, :, :, 0], cmap="viridis")
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

        # Display predictions with different thresholds
        raw_pred = predictions[i, :, :, 0]

        for t_idx, threshold in enumerate(thresholds):
            thresholded_pred = (raw_pred > threshold).astype(np.float32)
            axes[i, 3 + t_idx].imshow(thresholded_pred, cmap="viridis")
            axes[i, 3 + t_idx].set_title(f"Threshold {threshold}")
            axes[i, 3 + t_idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    # Configuration
    DATA_DIR = "../../../data/processed/processed_ndvi_rgb/image_datasets"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    MODEL_DIR = "models/forest_detection/models"
    LOG_DIR = "models/forest_detection/logs"

    # Print current directory for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Using data directory: {DATA_DIR}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Validation directory: {VAL_DIR}")

    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Training parameters
    EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    INPUT_SHAPE = (64, 64, 4)  # RGB + NDVI (removed Land Cover)
    AUGMENTATION_FACTOR = 3  # Multiply dataset size by this factor

    # Load datasets
    print("Loading training dataset...")
    X_train, y_train = load_dataset(TRAIN_DIR)
    print(f"Original training set: {X_train.shape}, {y_train.shape}")

    # Print class distribution information
    forest_pixel_ratio = np.mean(y_train)
    print(f"Forest pixel percentage: {forest_pixel_ratio * 100:.2f}%")
    print(
        f"Class distribution - Forest: {forest_pixel_ratio:.4f}, Non-forest: {1-forest_pixel_ratio:.4f}"
    )

    print("Augmenting training dataset...")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, AUGMENTATION_FACTOR)
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
        filepath=os.path.join(MODEL_DIR, f"forest_detection_model_{timestamp}.h5"),
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
        log_dir=os.path.join(LOG_DIR, f"forest_detection_{timestamp}"), histogram_freq=1
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
    final_model_path = os.path.join(MODEL_DIR, f"forest_detection_final_{timestamp}.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save training history to file
    np.save(os.path.join(LOG_DIR, f"training_history_{timestamp}.npy"), history.history)

    print("Training complete!")


if __name__ == "__main__":
    print("Starting forest detection training...")
    main()
