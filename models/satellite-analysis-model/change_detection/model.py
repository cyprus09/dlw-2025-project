import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Dropout,
    Lambda,
    Resizing,
)
from tensorflow.keras.models import Model


def iou_threshold(threshold=0.1):
    """
    Create an IoU metric with a specific threshold

    Args:
        threshold (float): Threshold for binary classification

    Returns:
        function: IoU metric function
    """

    def metric(y_true, y_pred):
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        intersection = tf.reduce_sum(y_pred_binary * y_true)
        union = tf.reduce_sum(y_pred_binary + y_true) - intersection

        iou = tf.math.divide_no_nan(intersection, union)
        return iou

    metric.__name__ = f"iou_threshold_{threshold}"
    return metric


def build_forest_detection_model(input_shape=(64, 64, 4), use_pretrained=True):
    """
    Build a U-Net inspired model for forest detection with shape-matching layers

    Args:
        input_shape (tuple): Input image shape
        use_pretrained (bool): Use pretrained weights for backbone

    Returns:
        tf.keras.Model: Compiled forest detection model
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # Custom preprocessing to handle 4-channel input
    def preprocess_input(x):
        # Assume first 3 channels are RGB, last channel (NDVI) is additional
        rgb_channels = x[..., :3]
        return tf.keras.applications.efficientnet.preprocess_input(rgb_channels)

    # Preprocess input for EfficientNet
    preprocessed_input = Lambda(preprocess_input)(inputs)

    # Resize for EfficientNet
    resized_inputs = Resizing(224, 224)(preprocessed_input)

    # Pretrained backbone (EfficientNetB0)
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet" if use_pretrained else None,
        input_tensor=resized_inputs,
    )

    # Encoder (Downsampling)
    # Extract feature maps from different stages
    layer_names = [
        "block2a_expand_activation",
        "block3a_expand_activation",
        "block4a_expand_activation",
        "top_activation",
    ]
    encoder_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Decoder (Upsampling) with skip connections
    x = encoder_outputs[-1]  # Last feature map

    # Decoder blocks with skip connections
    for i, encoder_output in reversed(list(enumerate(encoder_outputs[:-1]))):
        # Upsample
        x = UpSampling2D(size=(2, 2))(x)

        # Convolutional layers to match encoder output shape
        x = Conv2D(encoder_output.shape[-1], (3, 3), activation="relu", padding="same")(
            x
        )
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Resize to match encoder output shape
        x = Resizing(encoder_output.shape[1], encoder_output.shape[2])(x)

        # Skip connection
        x = Concatenate()([x, encoder_output])

        # Additional convolution
        x = Conv2D(64 * (2**i), (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

    # Resize back to original input shape
    x = Resizing(input_shape[0], input_shape[1])(x)

    # Final classification layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model with custom metrics and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            # IoU metrics at different thresholds
            iou_threshold(0.1),
            iou_threshold(0.3),
            iou_threshold(0.5),
        ],
    )

    return model