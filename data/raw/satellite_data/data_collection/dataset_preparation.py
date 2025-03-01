import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_dataset_without_landcover(metadata, data_dir, output_dir, patch_size=64):
    """
    Prepare dataset for U-Net training using only RGB and NDVI (no landcover)
    
    Args:
        metadata: DataFrame with metadata
        data_dir: Directory containing GeoTIFF files
        output_dir: Directory to save processed data
        patch_size: Size of image patches to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = []  # Feature stacks
    y = []  # Labels
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Preparing forest detection dataset"):
        try:
            rgb_path = os.path.join(data_dir, row['rgb_file'])
            ndvi_path = os.path.join(data_dir, row['ndvi_file'])
            
            # Check if files exist
            if not os.path.exists(rgb_path) or not os.path.exists(ndvi_path):
                print(f"Skipping {row['id']}_{row['year']} - missing files")
                continue
            
            # Read RGB data
            with rasterio.open(rgb_path) as src:
                rgb = src.read()
                height, width = rgb.shape[1], rgb.shape[2]
            
            # Read NDVI data
            with rasterio.open(ndvi_path) as src:
                ndvi = src.read(1)
                
                # Reshape to match RGB dimensions if needed
                if ndvi.shape != (height, width):
                    from skimage.transform import resize
                    ndvi = resize(ndvi, (height, width), preserve_range=True)
            
            # Normalize data
            rgb = rgb / 10000.0  # Typical scaling for reflectance values
            ndvi = (ndvi + 1) / 2.0  # Scale NDVI from [-1,1] to [0,1]
            
            # Stack features: RGB (3) + NDVI (1)
            features = np.vstack([rgb, ndvi[np.newaxis, :, :]])
            
            # Create binary mask based on known_forest (image-level label)
            is_forest = row['known_forest']
            
            if is_forest == 1:
                # If it's a forest image, create a mask highlighting probable forest areas
                # Use NDVI to guide this since we don't have landcover
                forest_mask = np.zeros((height, width), dtype=np.float32)
                
                # Areas with high NDVI are likely forest
                forest_mask[ndvi > 0.6] = 1.0
                forest_mask[(ndvi > 0.4) & (ndvi <= 0.6)] = 0.7
                
                # If the mask is still empty (unlikely), fill with forest label
                if np.sum(forest_mask) < 100:
                    forest_mask.fill(1.0)
            else:
                # Non-forest image - create mostly empty mask
                forest_mask = np.zeros((height, width), dtype=np.float32)
                
                # Small areas might still have some vegetation
                forest_mask[ndvi > 0.7] = 0.3  # Possible isolated trees
            
            # Extract center patch
            h_start = (height - patch_size) // 2
            w_start = (width - patch_size) // 2
            
            if h_start < 0 or w_start < 0:
                print(f"Warning: Image {row['id']}_{row['year']} is smaller than patch size")
                continue
                
            feature_patch = features[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
            mask_patch = forest_mask[h_start:h_start+patch_size, w_start:w_start+patch_size]
            
            # Add to dataset
            X.append(feature_patch)
            y.append(mask_patch)
            
        except Exception as e:
            print(f"Error processing {row['id']}_{row['year']}: {e}")
            continue
    
    # Convert to arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for Keras: (samples, height, width, channels)
    X = np.transpose(X, (0, 2, 3, 1))
    y = y[:, :, :, np.newaxis]  # Add channel dimension
    
    # Save dataset
    np.save(os.path.join(output_dir, 'X_features.npy'), X)
    np.save(os.path.join(output_dir, 'y_masks.npy'), y)
    
    print(f"Saved dataset with {len(X)} samples to {output_dir}")
    print(f"Feature shape: {X.shape}, Mask shape: {y.shape}")
    
    return X, y

def visualize_sample_data(X, y, num_samples=3, save_path=None):
    """
    Visualize sample data from the prepared dataset
    """
    if num_samples > len(X):
        num_samples = len(X)
    
    # Select random indices
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        # Display RGB
        rgb = X[idx, :, :, :3]
        axes[i, 0].imshow(np.clip(rgb, 0, 1))
        axes[i, 0].set_title(f"RGB Image")
        axes[i, 0].axis('off')
        
        # Display NDVI
        ndvi = X[idx, :, :, 3]
        axes[i, 1].imshow(ndvi, cmap='viridis')
        axes[i, 1].set_title(f"NDVI")
        axes[i, 1].axis('off')
        
        # Display mask
        mask = y[idx, :, :, 0]
        axes[i, 2].imshow(mask, cmap='viridis')
        axes[i, 2].set_title(f"Forest Mask")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def process_and_split_data(metadata_csv, data_dir, output_dir):
    """
    Process the dataset and split into train/val/test sets
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_datasets/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_datasets/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_datasets/test'), exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_csv)
    
    # Filter out rows with missing files or issues
    valid_rows = []
    for idx, row in metadata.iterrows():
        rgb_path = os.path.join(data_dir, row['rgb_file'])
        ndvi_path = os.path.join(data_dir, row['ndvi_file'])
        
        if os.path.exists(rgb_path) and os.path.exists(ndvi_path):
            valid_rows.append(row)
    
    metadata_filtered = pd.DataFrame(valid_rows)
    print(f"Filtered metadata: {len(metadata_filtered)} valid entries")
    
    # Save filtered metadata
    metadata_filtered.to_csv(os.path.join(output_dir, 'metadata_filtered.csv'), index=False)
    
    # Split into train/val/test
    train_val, test = train_test_split(
        metadata_filtered, 
        test_size=0.15, 
        stratify=metadata_filtered['known_forest'],
        random_state=42
    )
    
    train, val = train_test_split(
        train_val, 
        test_size=0.15, 
        stratify=train_val['known_forest'],
        random_state=42
    )
    
    # Save splits
    train.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test_metadata.csv'), index=False)
    
    print(f"Dataset split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    # Prepare datasets
    X_train, y_train = prepare_dataset_without_landcover(
        train, 
        data_dir, 
        os.path.join(output_dir, 'image_datasets/train')
    )
    
    X_val, y_val = prepare_dataset_without_landcover(
        val, 
        data_dir, 
        os.path.join(output_dir, 'image_datasets/val')
    )
    
    X_test, y_test = prepare_dataset_without_landcover(
        test, 
        data_dir, 
        os.path.join(output_dir, 'image_datasets/test')
    )
    
    # Visualize samples
    visualize_sample_data(
        X_train, 
        y_train, 
        num_samples=3, 
        save_path=os.path.join(output_dir, 'train_samples.png')
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    # Example usage
    metadata_csv = "metadata_relabeled.csv"
    data_dir = "carbon_verification_data"
    output_dir = "processed_data_rgb_ndvi"
    
    process_and_split_data(metadata_csv, data_dir, output_dir)