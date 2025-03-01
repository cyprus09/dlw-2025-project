import tensorflow as tf
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from model import build_forest_detection_model

def train_forest_model(data_dir, model_save_dir, batch_size=16, epochs=50):
    """
    Train the forest detection model.
    
    Args:
        data_dir: Directory with processed training data
        model_save_dir: Directory to save trained model
        batch_size: Batch size for training
        epochs: Number of training epochs
    """
    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load training data
    train_dir = os.path.join(data_dir, 'train')
    
    # Create data generator
    train_gen = DataGenerator(train_dir, batch_size=batch_size)
    
    # Load validation data
    val_dir = os.path.join(data_dir, 'test')
    val_gen = DataGenerator(val_dir, batch_size=batch_size)
    
    # Build model
    model = build_forest_detection_model()
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            monitor='val_loss'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_save_dir, 'logs')
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(model_save_dir, 'final_model.h5'))
    
    # Save training history
    pd.DataFrame(history.history).to_csv(
        os.path.join(model_save_dir, 'training_history.csv'), 
        index=False
    )
    
    return model, history

class DataGenerator(tf.keras.utils.Sequence):
    """Data generator for loading forest detection data."""
    
    def __init__(self, data_dir, batch_size=16, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get list of all feature files
        self.feature_files = sorted(glob.glob(os.path.join(data_dir, '*_features.npy')))
        self.indexes = np.arange(len(self.feature_files))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.feature_files) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate indexes for this batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_features = []
        batch_labels = []
        
        # Load data for this batch
        for i in batch_indexes:
            feature_file = self.feature_files[i]
            # Derive label file name from feature file name
            label_file = feature_file.replace('_features.npy', '_label.npy')
            
            # Load features and labels
            features = np.load(feature_file)
            label = np.load(label_file)
            
            batch_features.append(features)
            batch_labels.append(label)
        
        # Convert to arrays
        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)
        
        return batch_features, batch_labels
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch if shuffle is True."""
        if self.shuffle:
            np.random.shuffle(self.indexes)