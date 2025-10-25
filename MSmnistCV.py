import pandas as pd
import numpy as np
import os
import sys
import subprocess
import time

# Check if sklearn and required packages are properly installed
try:
    from sklearn.cluster import MeanShift
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "scikit-learn==0.24.2", "numpy==1.20.3", "scipy==1.7.0"])
    from sklearn.cluster import MeanShift
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score

print("Loading MNIST dataset...")
# Load MNIST dataset
try:
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    # Convert string labels to integers
    y = y.astype(int)
except Exception as e:
    print(f"Error loading MNIST from sklearn: {e}")
    print("Trying alternative download method...")
    # Alternative approach if fetch_openml fails
    import urllib.request
    import gzip
    
    def load_mnist():
        # MNIST files
        files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }
        
        # Download files if not present
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        for name, file in files.items():
            if not os.path.exists(file):
                print(f"Downloading {file}...")
                urllib.request.urlretrieve(base_url + file, file)
        
        # Load training images
        with gzip.open(files['train_images'], 'rb') as f:
            train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        
        # Load training labels
        with gzip.open(files['train_labels'], 'rb') as f:
            train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
        # Load test images
        with gzip.open(files['test_images'], 'rb') as f:
            test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        
        # Load test labels
        with gzip.open(files['test_labels'], 'rb') as f:
            test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
        # Combine training and test sets
        X = np.vstack([train_images, test_images])
        y = np.hstack([train_labels, test_labels])
        
        return X, y
    
    X, y = load_mnist()

print("Preparing dataset with specific class distribution...")
# Select specific number of samples for each class
class_counts = {
    0: 7000,
    1: 6500,
    2: 6000,
    3: 5500,
    4: 5000,
    5: 4500,
    6: 4000,
    7: 3500,
    8: 3000,
    9: 2500
}

# Create the dataset with the specified distribution
X_selected = []
y_selected = []

for digit, count in class_counts.items():
    # Get indices of this digit
    indices = np.where(y == digit)[0]
    
    # Select the required number (or all if fewer available)
    selected_count = min(count, len(indices))
    selected_indices = indices[:selected_count]
    
    # Add to our dataset
    X_selected.append(X[selected_indices])
    y_selected.append(y[selected_indices])

# Combine all selected samples
X = np.vstack(X_selected)
y = np.hstack(y_selected)

# Normalize pixel values to [0, 1]
X = X / 255.0

print(f"Final dataset shape: {X.shape}, Labels shape: {y.shape}")
print("Class distribution:")
for digit in range(10):
    count = np.sum(y == digit)
    print(f"Digit {digit}: {count} samples")

# Function to map cluster labels to digit predictions
def map_clusters_to_digits(cluster_labels, y_true):
    # Find unique cluster labels
    unique_clusters = np.unique(cluster_labels)
    
    # For each cluster, find the majority digit
    cluster_to_digit = {}
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        if np.sum(mask) > 0:
            # Find majority digit in this cluster
            cluster_digits = y_true[mask]
            unique_digits, digit_counts = np.unique(cluster_digits, return_counts=True)
            majority_digit = unique_digits[np.argmax(digit_counts)]
            cluster_to_digit[cluster] = majority_digit
    
    # Map each point's cluster to its predicted digit
    y_pred = np.zeros_like(cluster_labels)
    for i, cluster in enumerate(cluster_labels):
        y_pred[i] = cluster_to_digit[cluster]
    
    return y_pred

# Implement our own k-fold to avoid dependency issues
def custom_kfold(X, y, n_splits=5, shuffle=True, random_state=None):
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    
    fold_size = n_samples // n_splits
    
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        yield train_indices, test_indices

# 5-fold cross-validation
accuracies = []

print("\n--- 5-Fold Cross-Validation with Mean Shift Clustering ---")

for fold, (train_idx, test_idx) in enumerate(custom_kfold(X, y, n_splits=5, shuffle=True, random_state=42)):
    fold_start_time = time.time()
    
    # Split data for this fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Print class distribution for this fold
    print(f"\nFold {fold+1} - Test set: {len(y_test)} samples")
    for digit in range(10):
        count = np.sum(y_test == digit)
        print(f"  Digit {digit}: {count} samples")
    
    # Calculate bandwidth based on training data
    # For MNIST, we'll use a smaller bandwidth due to the high dimensionality
    bandwidth = np.mean(np.std(X_train, axis=0)) * 0.2
    print(f"Using bandwidth: {bandwidth:.4f}")
    
    print("Running Mean Shift clustering...")
    clustering_start = time.time()
    
    # Use scikit-learn's MeanShift with bin_seeding for speed
    try:
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1, cluster_all=True)
        cluster_labels = meanshift.fit_predict(X_test)
        n_clusters = len(np.unique(cluster_labels))
        print(f"Found {n_clusters} clusters in {time.time() - clustering_start:.2f} seconds")
        
        # If we don't get close to 10 clusters, try adjusting bandwidth
        if n_clusters < 8 or n_clusters > 15:
            print(f"Adjusting bandwidth (got {n_clusters} clusters instead of ~10)")
            if n_clusters < 8:
                # Too few clusters, reduce bandwidth
                bandwidth *= 0.8
            else:
                # Too many clusters, increase bandwidth
                bandwidth *= 1.2
            
            print(f"Retrying with bandwidth: {bandwidth:.4f}")
            meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
            cluster_labels = meanshift.fit_predict(X_test)
            n_clusters = len(np.unique(cluster_labels))
            print(f"Found {n_clusters} clusters with adjusted bandwidth")
    
    except Exception as e:
        print(f"Error with scikit-learn's MeanShift: {e}")
        print("Mean Shift clustering failed for this fold. Skipping...")
        continue
    
    # Map clusters to digit predictions
    y_pred = map_clusters_to_digits(cluster_labels, y_test)
    
    # Calculate accuracy
    acc = np.mean(y_test == y_pred)
    accuracies.append(acc)
    
    # Count correct predictions by class
    print("Accuracy per digit:")
    for digit in range(10):
        mask = y_test == digit
        if np.sum(mask) > 0:
            digit_acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"  Digit {digit}: {digit_acc:.4f} ({np.sum(y_pred[mask] == y_test[mask])}/{np.sum(mask)})")
    
    print(f"Overall fold accuracy: {acc:.4f}")
    print(f"Fold {fold+1} completed in {time.time() - fold_start_time:.2f} seconds")

# Calculate mean and standard deviation of accuracies
if accuracies:
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print("\n--- Results ---")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_accuracy:.4f}")
else:
    print("\nNo successful folds completed. Please check your environment and try again.")
