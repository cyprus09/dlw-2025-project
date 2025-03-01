import os
import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

def relabel_forest_cover(metadata_csv, data_dir, output_csv, method='dynamic_world'):
    """
    Relabel forest cover based on actual imagery data instead of random values.
    
    Args:
        metadata_csv: Path to metadata CSV file
        data_dir: Directory containing the GeoTIFF files
        output_csv: Path to save the updated metadata
        method: Method to use for labeling ('dynamic_world', 'ndvi', 'combined')
    
    Returns:
        Updated metadata DataFrame
    """
    print(f"Relabeling forest cover using method: {method}")
    metadata = pd.read_csv(metadata_csv)
    valid_rows = []
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing images"):
        try:
            # Check if files exist
            rgb_path = os.path.join(data_dir, row['rgb_file'])
            ndvi_path = os.path.join(data_dir, row['ndvi_file'])
            lc_path = os.path.join(data_dir, row['landcover_file'])
            
            if not all(os.path.exists(p) for p in [rgb_path, ndvi_path, lc_path]):
                print(f"Skipping {row['id']}_{row['year']} - missing files")
                continue
            
            # Initialize with a default value
            is_forest = 0
            
            if method == 'dynamic_world' or method == 'combined':
                # Use Dynamic World land cover to determine forest cover
                with rasterio.open(lc_path) as src:
                    lc_data = src.read(1)  # Read first band
                    
                    # Check if the image has data
                    if np.count_nonzero(lc_data) < 100:  # Arbitrary threshold
                        print(f"Warning: Landcover image for {row['id']}_{row['year']} is mostly empty")
                        if method == 'combined':
                            # Fall back to NDVI if using combined method
                            use_ndvi = True
                        else:
                            continue
                    else:
                        use_ndvi = False
                        
                        # Dynamic World classes:
                        # 0: Water, 1: Trees, 2: Grass, 3: Flooded vegetation
                        # 4: Crops, 5: Shrub and scrub, 6: Built area
                        # 7: Bare ground, 8: Snow and ice, 9: Cloud
                        
                        # Classes 1 (Trees) and 5 (Shrub) are forest/woodland
                        forest_classes = [1, 5]
                        
                        # Calculate percentage of pixels that are forest
                        total_pixels = lc_data.size
                        forest_pixels = np.sum(np.isin(lc_data, forest_classes))
                        forest_percentage = (forest_pixels / total_pixels) * 100
                        
                        # Mark as forest if more than 30% is forest
                        is_forest = 1 if forest_percentage > 30 else 0
                        
                        # Store the forest percentage for reference
                        row['forest_percentage'] = forest_percentage
            
            if method == 'ndvi' or (method == 'combined' and use_ndvi):
                # Use NDVI to determine forest cover
                with rasterio.open(ndvi_path) as src:
                    ndvi_data = src.read(1)  # Read NDVI band
                    
                    # NDVI values:
                    # <0: No vegetation, 0-0.2: Sparse vegetation
                    # 0.2-0.4: Moderate vegetation, >0.4: Dense vegetation
                    
                    # Calculate mean NDVI (mask out no-data values)
                    valid_ndvi = ndvi_data[ndvi_data != src.nodata] if src.nodata else ndvi_data
                    mean_ndvi = np.mean(valid_ndvi)
                    
                    # Forests typically have NDVI > 0.5
                    is_forest = 1 if mean_ndvi > 0.5 else 0
                    
                    # Store the mean NDVI for reference
                    row['mean_ndvi'] = mean_ndvi
            
            # Update the known_forest value
            row['known_forest'] = is_forest
            valid_rows.append(row)
            
        except Exception as e:
            print(f"Error processing {row['id']}_{row['year']}: {e}")
            continue
    
    # Create updated DataFrame
    updated_metadata = pd.DataFrame(valid_rows)
    
    # Save updated metadata
    updated_metadata.to_csv(output_csv, index=False)
    print(f"Saved {len(updated_metadata)} entries to {output_csv}")
    
    # Print forest cover statistics
    forest_count = updated_metadata['known_forest'].sum()
    total_count = len(updated_metadata)
    print(f"Forest cover distribution: {forest_count} forest ({forest_count/total_count:.1%}), " 
          f"{total_count-forest_count} non-forest ({(total_count-forest_count)/total_count:.1%})")
    
    return updated_metadata


def visualize_examples(metadata, data_dir, num_examples=3):
    """
    Visualize sample forest and non-forest examples to verify labels.
    """
    forest_samples = metadata[metadata['known_forest'] == 1].sample(min(num_examples, sum(metadata['known_forest'])))
    nonforest_samples = metadata[metadata['known_forest'] == 0].sample(min(num_examples, sum(metadata['known_forest'] == 0)))
    
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 10))
    
    # Plot forest examples
    for i, (_, row) in enumerate(forest_samples.iterrows()):
        rgb_path = os.path.join(data_dir, row['rgb_file'])
        with rasterio.open(rgb_path) as src:
            # Read RGB bands and normalize for display
            rgb = src.read([1, 2, 3])  # Usually bands are in BGR order
            rgb = np.moveaxis(rgb, 0, 2)  # Change from (3,H,W) to (H,W,3)
            
            # Normalize to 0-1 range for display
            rgb_min, rgb_max = np.percentile(rgb, (2, 98))  # Use percentiles to handle outliers
            rgb_normalized = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
            
            axes[0, i].imshow(rgb_normalized)
            axes[0, i].set_title(f"Forest: {row['id']}_{row['year']}")
            axes[0, i].axis('off')
    
    # Plot non-forest examples
    for i, (_, row) in enumerate(nonforest_samples.iterrows()):
        rgb_path = os.path.join(data_dir, row['rgb_file'])
        with rasterio.open(rgb_path) as src:
            rgb = src.read([1, 2, 3])
            rgb = np.moveaxis(rgb, 0, 2)
            
            rgb_min, rgb_max = np.percentile(rgb, (2, 98))
            rgb_normalized = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
            
            axes[1, i].imshow(rgb_normalized)
            axes[1, i].set_title(f"Non-Forest: {row['id']}_{row['year']}")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('forest_label_examples.png')
    plt.close()
    print("Example visualizations saved to 'forest_label_examples.png'")


def prepare_dataset_for_unet(metadata, data_dir, output_dir, patch_size=64):
    """
    Prepare dataset for U-Net training by:
    1. Extracting patches from the images
    2. Creating feature stacks (RGB+NDVI+LC)
    3. Creating binary masks from known_forest
    
    Args:
        metadata: DataFrame with metadata
        data_dir: Directory containing GeoTIFF files
        output_dir: Directory to save processed data
        patch_size: Size of image patches to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = []  # Feature stacks
    y = []  # Labels
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Preparing U-Net dataset"):
        try:
            rgb_path = os.path.join(data_dir, row['rgb_file'])
            ndvi_path = os.path.join(data_dir, row['ndvi_file'])
            lc_path = os.path.join(data_dir, row['landcover_file'])
            
            # Read RGB data
            with rasterio.open(rgb_path) as src:
                rgb = src.read()
                transform = src.transform
                height, width = rgb.shape[1], rgb.shape[2]
            
            # Read NDVI data
            with rasterio.open(ndvi_path) as src:
                ndvi = src.read(1)
                
                # Reshape to match RGB dimensions if needed
                if ndvi.shape != (height, width):
                    from skimage.transform import resize
                    ndvi = resize(ndvi, (height, width), preserve_range=True)
            
            # Read land cover data
            with rasterio.open(lc_path) as src:
                lc = src.read(1)
                
                # Reshape to match RGB dimensions if needed
                if lc.shape != (height, width):
                    from skimage.transform import resize
                    lc = resize(lc, (height, width), preserve_range=True, order=0)
            
            # Normalize data
            rgb = rgb / 10000.0  # Typical scaling for reflectance values
            ndvi = (ndvi + 1) / 2.0  # Scale NDVI from [-1,1] to [0,1]
            
            # Create a one-hot encoding for landcover (simplify to 6 classes)
            # 0: Water, 1: Forest, 2: Grassland, 3: Cropland, 
            # 4: Built/Urban, 5: Barren/Other
            lc_simplified = np.zeros_like(lc)
            lc_simplified[lc == 0] = 0  # Water
            lc_simplified[np.isin(lc, [1, 5])] = 1  # Forest classes
            lc_simplified[np.isin(lc, [2, 3])] = 2  # Grassland
            lc_simplified[lc == 4] = 3  # Cropland
            lc_simplified[lc == 6] = 4  # Built
            lc_simplified[np.isin(lc, [7, 8, 9])] = 5  # Other
            
            # Stack features: RGB (3) + NDVI (1) + Landcover (1)
            features = np.vstack([rgb, ndvi[np.newaxis, :, :], lc_simplified[np.newaxis, :, :]])
            
            # Create binary mask based on known_forest
            # For a segmentation model, we'd ideally have pixel-level labels
            # Since we only have image-level labels, we'll create a synthetic mask
            # where forest images are assumed to be >50% forest
            is_forest = row['known_forest']
            
            if is_forest == 1:
                # If it's a forest image, create a mask highlighting forest areas
                # Use LC and NDVI to guide this
                forest_mask = np.zeros((height, width), dtype=np.float32)
                
                # Areas classified as forest (class 1) in land cover
                forest_mask[lc_simplified == 1] = 1.0
                
                # Areas with high NDVI but not classified as forest
                forest_mask[(ndvi > 0.6) & (lc_simplified != 1)] = 0.7
            else:
                # Non-forest image - create empty mask
                forest_mask = np.zeros((height, width), dtype=np.float32)
                
                # Small areas might still have trees
                forest_mask[lc_simplified == 1] = 0.3
            
            # Extract patches (simplistic approach - center crop)
            # In a full implementation, you'd extract multiple patches per image
            h_start = (height - patch_size) // 2
            w_start = (width - patch_size) // 2
            
            if h_start < 0 or w_start < 0:
                print(f"Warning: Image {row['id']}_{row['year']} is smaller than patch size")
                continue
                
            feature_patch = features[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
            mask_patch = forest_mask[h_start:h_start+patch_size, w_start:w_start+patch_size]
            
            # Append to lists
            X.append(feature_patch)
            y.append(mask_patch)
            
        except Exception as e:
            print(f"Error processing {row['id']}_{row['year']} for U-Net: {e}")
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


if __name__ == "__main__":
    # Set paths
    metadata_csv = "metadata.csv"
    data_dir = "carbon_verification_data"
    output_dir = "processed_data"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Relabel forest cover
    updated_metadata = relabel_forest_cover(
        metadata_csv, 
        data_dir, 
        os.path.join(output_dir, "metadata_relabeled.csv"),
        method='combined'  # Use both dynamic world and NDVI
    )
    
    # Step 2: Visualize examples to verify labels
    visualize_examples(updated_metadata, data_dir, num_examples=3)
    
    # Step 3: Split into train/val/test
    train_val, test = train_test_split(
        updated_metadata, 
        test_size=0.15, 
        stratify=updated_metadata['known_forest'],
        random_state=42
    )
    
    train, val = train_test_split(
        train_val, 
        test_size=0.15, 
        stratify=train_val['known_forest'],
        random_state=42
    )
    
    # Save splits
    train.to_csv(os.path.join(output_dir, "train_metadata.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val_metadata.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test_metadata.csv"), index=False)
    
    print(f"Dataset split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    # Step 4: Prepare dataset for U-Net training
    # We'll prepare each split separately
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    X_train, y_train = prepare_dataset_for_unet(train, data_dir, os.path.join(output_dir, 'train'))
    X_val, y_val = prepare_dataset_for_unet(val, data_dir, os.path.join(output_dir, 'val'))
    X_test, y_test = prepare_dataset_for_unet(test, data_dir, os.path.join(output_dir, 'test'))
    
    print("Dataset preparation complete!")