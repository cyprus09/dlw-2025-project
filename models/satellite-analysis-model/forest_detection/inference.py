import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import seaborn as sns
import argparse

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


# Define custom IoU metric with threshold
def custom_iou(y_true, y_pred, threshold=0.5):
    # Apply threshold to predictions
    y_pred = tf.cast(y_pred > threshold, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        - intersection
    )

    # Add small epsilon to avoid division by zero
    iou = tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))
    return iou


# Create a function to make a custom IoU with a specific threshold
def make_iou_threshold(threshold):
    def iou_threshold(y_true, y_pred):
        return custom_iou(y_true, y_pred, threshold)

    iou_threshold.__name__ = f"iou_threshold_{threshold}"
    return iou_threshold


def load_model_with_custom_objects(model_path):
    custom_objects = {
        "dice_loss": dice_loss,
        "combined_loss": combined_loss,
        "custom_iou": custom_iou,
        "iou_threshold_0.1": make_iou_threshold(0.1),
        "iou_threshold_0.5": make_iou_threshold(0.5),
    }

    try:
        # Try loading the model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully with custom objects")
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            # Try loading without compiling
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded without compilation")

            # Recompile with custom objects
            model.compile(
                optimizer="adam",
                loss=combined_loss,
                metrics=[
                    "accuracy",
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.Precision(name="precision"),
                    make_iou_threshold(0.1),
                    make_iou_threshold(0.5),
                ],
            )
            print("Model recompiled with custom loss and metrics")
        except Exception as e2:
            raise Exception(f"Failed to load model: {e2}")

    return model


def load_dataset(data_dir):
    features = np.load(os.path.join(data_dir, "X_features.npy"))
    masks = np.load(os.path.join(data_dir, "y_masks.npy"))
    return features, masks


def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test, verbose=1)

    print("Test Metrics:")
    for name, value in zip(model.metrics_names, metrics):
        print(f"{name}: {value:.4f}")

    y_pred = model.predict(X_test)

    return metrics, y_pred


def plot_precision_recall_curve(y_true_flat, y_pred_flat, save_path=None):
    y_true_binary = (y_true_flat > 0.5).astype(int)

    # Keep predictions as continuous for precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_flat)
    average_precision = average_precision_score(y_true_binary, y_pred_flat)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2, label=f"PR curve (AP = {average_precision:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return average_precision


def plot_confusion_matrix(y_true_flat, y_pred_flat, threshold=0.5, save_path=None):
    y_true_binary = (y_true_flat > 0.5).astype(int)
    y_pred_binary = (y_pred_flat > threshold).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Forest", "Forest"],
        yticklabels=["Non-Forest", "Forest"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Threshold = {threshold})")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"Confusion Matrix Metrics (Threshold = {threshold}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return cm


def visualize_predictions(X_test, y_test, y_pred, num_samples=10, save_dir=None):
    # Get random indices
    indices = np.random.choice(
        len(X_test), min(num_samples, len(X_test)), replace=False
    )

    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        rgb_img = X_test[idx, :, :, :3]
        if rgb_img.max() > 1.0:  # Normalize if needed
            rgb_img = rgb_img / 255.0
        axes[0].imshow(rgb_img)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")

        # Display ground truth mask
        axes[1].imshow(y_test[idx, :, :, 0], cmap="viridis")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Display prediction
        axes[2].imshow(y_pred[idx, :, :, 0], cmap="viridis")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"prediction_{i}.png"))
            plt.close()
        else:
            plt.show()


def calculate_forest_percentage(masks):
    forest_pixels = np.sum(masks > 0.5)
    total_pixels = masks.size
    return (forest_pixels / total_pixels) * 100


def predict_on_new_image(model, image_path, output_path=None):
    print(
        f"[WARNING] Using random tensor for demonstration. Implement actual image preprocessing."
    )
    image = np.random.random((1, 64, 64, 4))  # RGB + NDVI

    prediction = model.predict(image)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image[0, :, :, :3])
    plt.title("Input Image (RGB channels)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(prediction[0, :, :, 0], cmap="viridis")
    plt.title("Forest Prediction")
    plt.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

    forest_percentage = np.mean(prediction > 0.5) * 100
    print(f"Predicted forest coverage: {forest_percentage:.2f}%")

    return prediction, forest_percentage


def main():
    parser = argparse.ArgumentParser(description="Forest Detection Model Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/processed/image_datasets/test",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/forest_detection/inference_results",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification",
    )
    parser.add_argument(
        "--single_image",
        type=str,
        default=None,
        help="Path to a single image for inference",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = load_model_with_custom_objects(args.model_path)

    if args.single_image:
        output_path = os.path.join(args.output_dir, "single_prediction.png")
        predict_on_new_image(model, args.single_image, output_path)
    else:
        print(f"Loading test dataset from {args.test_dir}")
        X_test, y_test = load_dataset(args.test_dir)
        print(f"Test set: {X_test.shape}, {y_test.shape}")

        metrics, y_pred = evaluate_model(model, X_test, y_test)

        y_true_flat = y_test.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        print(f"y_true_flat shape: {y_true_flat.shape}")
        print(f"y_true_flat range: Min={y_true_flat.min()}, Max={y_true_flat.max()}")
        print(f"y_pred_flat shape: {y_pred_flat.shape}")
        print(f"y_pred_flat range: Min={y_pred_flat.min()}, Max={y_pred_flat.max()}")

        pr_curve_path = os.path.join(args.output_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(y_true_flat, y_pred_flat, save_path=pr_curve_path)
        print(f"Precision-recall curve saved to {pr_curve_path}")

        cm_path = os.path.join(
            args.output_dir, f"confusion_matrix_t{args.threshold}.png"
        )
        plot_confusion_matrix(
            y_true_flat, y_pred_flat, threshold=args.threshold, save_path=cm_path
        )
        print(f"Confusion matrix saved to {cm_path}")

        # Visualize predictions
        viz_dir = os.path.join(args.output_dir, "visualization")
        visualize_predictions(X_test, y_test, y_pred, num_samples=10, save_dir=viz_dir)
        print(f"Visualization samples saved to {viz_dir}")

        # Calculate forest percentage
        true_forest_percentage = calculate_forest_percentage(y_test)
        pred_forest_percentage = calculate_forest_percentage(y_pred)

        print(f"True forest coverage: {true_forest_percentage:.2f}%")
        print(f"Predicted forest coverage: {pred_forest_percentage:.2f}%")
        print(
            f"Difference: {abs(true_forest_percentage - pred_forest_percentage):.2f}%"
        )

        with open(os.path.join(args.output_dir, "summary_results.txt"), "w") as f:
            f.write(f"Test Dataset: {args.test_dir}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Classification Threshold: {args.threshold}\n\n")

            f.write("Evaluation Metrics:\n")
            for name, value in zip(model.metrics_names, metrics):
                f.write(f"{name}: {value:.4f}\n")

            f.write(f"\nTrue forest coverage: {true_forest_percentage:.2f}%\n")
            f.write(f"Predicted forest coverage: {pred_forest_percentage:.2f}%\n")
            f.write(
                f"Difference: {abs(true_forest_percentage - pred_forest_percentage):.2f}%\n"
            )

        print(f"All inference results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
