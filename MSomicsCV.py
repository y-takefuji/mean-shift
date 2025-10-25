import pandas as pd
import numpy as np
import os
import sys
import subprocess

# Check if sklearn is properly installed, if not install it
try:
    from sklearn.cluster import MeanShift
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "scikit-learn==0.24.2", "numpy==1.20.3", "scipy==1.7.0"])
    from sklearn.cluster import MeanShift
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('vital.status', axis=1).values
y = df['vital.status'].values

# Function to map cluster labels to binary predictions
def map_clusters_to_binary(cluster_labels, y_true):
    # Find unique cluster labels
    unique_clusters = np.unique(cluster_labels)
    
    # For each cluster, find the majority class
    cluster_to_class = {}
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        if np.sum(mask) > 0:
            # Find majority class in this cluster
            cluster_classes = y_true[mask]
            unique_classes, class_counts = np.unique(cluster_classes, return_counts=True)
            majority_class = unique_classes[np.argmax(class_counts)]
            cluster_to_class[cluster] = majority_class
    
    # Map each point's cluster to its predicted class
    y_pred = np.zeros_like(cluster_labels)
    for i, cluster in enumerate(cluster_labels):
        y_pred[i] = cluster_to_class[cluster]
    
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
    # Split data for this fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Print class distribution for this fold
    n_positives = np.sum(y_test == 1)
    n_negatives = np.sum(y_test == 0)
    print(f"\nFold {fold+1} - Test set: {len(y_test)} samples "
          f"({n_positives} positives, {n_negatives} negatives)")
    
    # Calculate bandwidth based on training data variance
    bandwidth = np.mean(np.std(X_train, axis=0)) * 0.5
    
    # Use scikit-learn's MeanShift
    try:
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        cluster_labels = meanshift.fit_predict(X_test)
    except Exception as e:
        print(f"Error with scikit-learn's MeanShift: {e}")
        print("Falling back to simplified implementation...")
        
        # Simplified Mean Shift implementation
        def simplified_meanshift(X, bandwidth):
            n_samples = len(X)
            labels = np.zeros(n_samples, dtype=int)
            cluster_id = 0
            
            # For each unassigned point
            for i in range(n_samples):
                if labels[i] != 0:  # Already assigned
                    continue
                    
                # Start a new cluster
                cluster_id += 1
                labels[i] = cluster_id
                
                # Find all points within bandwidth
                for j in range(i+1, n_samples):
                    if labels[j] != 0:  # Already assigned
                        continue
                        
                    # Simple distance check
                    if np.linalg.norm(X[i] - X[j]) <= bandwidth:
                        labels[j] = cluster_id
            
            # Reindex from 0
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                labels[labels == label] = i
                
            return labels
        
        cluster_labels = simplified_meanshift(X_test, bandwidth)
    
    # Map clusters to binary predictions
    y_pred = map_clusters_to_binary(cluster_labels, y_test)
    
    # Calculate accuracy
    acc = np.mean(y_test == y_pred)
    accuracies.append(acc)
    
    # Count correct predictions by class
    true_pos = np.sum((y_test == 1) & (y_pred == 1))
    true_neg = np.sum((y_test == 0) & (y_pred == 0))
    
    print(f"Clusters found: {len(np.unique(cluster_labels))}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Correct predictions: {true_pos}/{n_positives} positives, {true_neg}/{n_negatives} negatives")

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("\n--- Results ---")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation: {std_accuracy:.4f}")

# Print overall class distribution
total_positives = np.sum(y == 1)
total_negatives = np.sum(y == 0)
print(f"\nOverall dataset: {len(y)} samples "
      f"({total_positives} positives, {total_negatives} negatives)")
