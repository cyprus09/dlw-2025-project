import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import (
    concatenate,
    Conv2DTranspose,
    Dropout,
    BatchNormalization,
    Resizing,
)
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def custom_iou(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        - intersection
    )

    iou = tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))
    return iou


# Define Dice loss
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


# Combined loss function
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


# Create a function to make a custom IoU with a specific threshold
def make_iou_threshold(threshold):
    def iou_threshold(y_true, y_pred):
        return custom_iou(y_true, y_pred, threshold)

    iou_threshold.__name__ = f"iou_threshold_{threshold}"
    return iou_threshold


def build_forest_detection_model(input_shape=(64, 64, 4), use_pretrained=True):
    inputs = Input(input_shape)
    print(f"Input shape: {input_shape}")

    rgb_input = tf.keras.layers.Lambda(lambda x: x[:, :, :, :3])(inputs)

    # Convert non-RGB channels (NDVI) to features with 1x1 convolutions
    if input_shape[2] > 3:
        other_channels = tf.keras.layers.Lambda(lambda x: x[:, :, :, 3:])(inputs)
        other_features = Conv2D(16, 1, activation="relu")(other_channels)
        print(f"NDVI features shape: {other_features.shape}")
    else:
        other_features = None

    # Pre-trained encoder (MobileNetV2)
    if use_pretrained:
        base_model = MobileNetV2(
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=False,
            weights="imagenet",
        )

        for layer in base_model.layers:
            try:
                output_shape = layer.output_shape
                if output_shape is None:
                    output_shape = "Unknown"
                print(f"Layer: {layer.name}, Output shape: {output_shape}")
            except (AttributeError, IndexError):
                print(f"Layer: {layer.name}, Output shape: Could not determine")

        # Process RGB input through the model
        features = base_model(rgb_input)
        print(f"Base model output shape: {features.shape}")

        # Bridge is the bottleneck feature map (smallest spatial dimension)
        bridge = features
        print(f"Bridge shape: {bridge.shape}")

        # Upsampling path
        x = Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(
            bridge
        )
        x = BatchNormalization()(x)
        print(f"Upsampling 1 shape: {x.shape}")

        # Upsampling block 2
        x = Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        print(f"Upsampling 2 shape: {x.shape}")

        # Upsampling block 3
        x = Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        print(f"Upsampling 3 shape: {x.shape}")

        # Upsampling block 4
        x = Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        print(f"Upsampling 4 shape: {x.shape}")

        # Final upsampling block 5 - to ensure 64x64 output
        x = Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        print(f"Upsampling 5 shape: {x.shape}")

        # Add NDVI features if available
        if other_features is not None:
            current_size = x.shape[1]
            other_resized = Resizing(current_size, current_size)(other_features)
            x = concatenate([x, other_resized])
            print(f"After concatenation with NDVI features: {x.shape}")

        x = Conv2D(64, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

    else:
        x = Conv2D(64, 3, activation="relu", padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, 3, activation="relu", padding="same")(x)
        skip1 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 3, activation="relu", padding="same")(x)
        skip2 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(256, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, 3, activation="relu", padding="same")(x)
        skip3 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(512, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, 3, activation="relu", padding="same")(x)
        skip4 = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Bridge
        x = Conv2D(1024, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(1024, 3, activation="relu", padding="same")(x)

        # Decoder with skip connections
        x = Conv2DTranspose(512, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = concatenate([x, skip4])
        x = Conv2D(512, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = concatenate([x, skip3])
        x = Conv2D(256, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = concatenate([x, skip2])
        x = Conv2D(128, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = concatenate([x, skip1])
        x = Conv2D(64, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)

    # Output
    outputs = Conv2D(1, 1, activation="sigmoid")(x)
    print(f"Final output shape: {outputs.shape}")

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=combined_loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            make_iou_threshold(0.1),
            make_iou_threshold(0.5),
        ],
    )

    return model
