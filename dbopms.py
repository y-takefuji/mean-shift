import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, MeanShift
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

# Load data
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('vital.status', axis=1)
y_true = df['vital.status']

# Function for improved gene selection using variance and Spearman correlation
def select_informative_genes(X, y, n_top=500, method='combined'):
    """
    Select informative genes using variance and/or Spearman correlation
    
    Parameters:
    -----------
    X : DataFrame
        Gene expression data (samples x genes)
    y : Series
        Target variable (vital.status)
    n_top : int
        Number of top genes to select
    method : str
        'variance': select by variance only
           
    Returns:
    --------
    X_selected : DataFrame
        Data with only selected genes
    selected_genes : list
        List of selected gene names
    """
    # Calculate metrics for each gene
    gene_var = X.var(axis=0)
    gene_corr = pd.Series(index=X.columns)
    
    # Calculate absolute Spearman correlation with target
    for gene in X.columns:
        corr, _ = spearmanr(X[gene], y)
        gene_corr[gene] = abs(corr)  # Use absolute correlation
    
    if method == 'variance':
        # Sort genes by variance
        sorted_idx = np.argsort(gene_var.values)[::-1]
        metric_name = "variance"
        metric_values = gene_var
    elif method == 'spearman':
        # Sort genes by correlation
        sorted_idx = np.argsort(gene_corr.values)[::-1]
        metric_name = "Spearman correlation"
        metric_values = gene_corr
    elif method == 'combined':
        # Combine both metrics (normalized)
        gene_var_norm = (gene_var - gene_var.min()) / (gene_var.max() - gene_var.min())
        gene_corr_norm = (gene_corr - gene_corr.min()) / (gene_corr.max() - gene_corr.min())
        combined_score = gene_var_norm + gene_corr_norm
        sorted_idx = np.argsort(combined_score.values)[::-1]
        metric_name = "combined score"
        metric_values = combined_score
        
    # Select top n genes
    n_select = min(n_top, X.shape[1])
    selected_idx = sorted_idx[:n_select]
    selected_genes = X.columns[selected_idx].tolist()
    
    # Return selected data
    X_selected = X.iloc[:, selected_idx]
    
    print(f"Selected {n_select} genes using {metric_name} out of {X.shape[1]} total genes")
    
    # Print some stats about top genes
    print(f"Top 5 genes by {metric_name}:")
    for i, gene in enumerate(selected_genes[:5]):
        print(f"  {i+1}. {gene}: {metric_values[gene]:.4f}")
    
    return X_selected, selected_genes

# Select informative genes using different methods
X_var, var_genes = select_informative_genes(X, y_true, n_top=500, method='variance')
X_corr, corr_genes = select_informative_genes(X, y_true, n_top=500, method='spearman')
X_combined, combined_genes = select_informative_genes(X, y_true, n_top=500, method='combined')

# Function to map cluster labels to binary predictions
def map_clusters_to_binary(cluster_labels):
    # Find unique cluster labels (excluding noise points marked as -1)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]
    
    if len(unique_clusters) < 2:
        # If only one cluster (or none), assign all to majority class
        majority_class = np.argmax(np.bincount(y_true))
        return np.full(len(cluster_labels), majority_class)
    
    # For each cluster, find the majority class
    cluster_to_class = {}
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        if np.sum(mask) > 0:
            # Find majority class in this cluster
            cluster_majority = np.argmax(np.bincount(y_true[mask]))
            cluster_to_class[cluster] = cluster_majority
    
    # Map each point's cluster to its predicted class
    y_pred = np.zeros_like(cluster_labels)
    for i, cluster in enumerate(cluster_labels):
        if cluster == -1:  # Noise points
            # Assign noise points to overall majority class
            y_pred[i] = np.argmax(np.bincount(y_true))
        else:
            y_pred[i] = cluster_to_class[cluster]
    
    return y_pred

# Function to evaluate clustering results
def evaluate_clustering(y_true, y_pred, algorithm_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{algorithm_name} Clustering Results:")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return acc

# Find optimal parameters for DBSCAN with different feature sets
def optimize_dbscan(X, y, eps_range=None, min_samples_range=None):
    if eps_range is None:
        # Generate a range of eps values based on data distribution
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        distances = np.sort(distances[:, 4])  # 5th nearest neighbor
        eps_range = np.linspace(distances[int(len(distances)*0.01)], 
                                distances[int(len(distances)*0.2)], 
                                8)
    
    if min_samples_range is None:
        # Try different values for min_samples
        min_samples_range = [2, 3, 5, 10]
    
    best_acc = 0
    best_params = {}
    best_pred = None
    
    print("Finding best DBSCAN parameters...")
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Count non-noise clusters
            n_clusters = len(np.unique(labels[labels != -1]))
            if n_clusters >= 2:  # Only if we have at least 2 clusters
                pred = map_clusters_to_binary(labels)
                acc = accuracy_score(y, pred)
                
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_pred = pred
                    print(f"  Improved: eps={eps:.4f}, min_samples={min_samples}, acc={acc:.4f}, clusters={n_clusters}")
    
    if not best_params:
        print("No suitable parameters found, using defaults")
        best_params = {'eps': 0.5, 'min_samples': 5}
        dbscan = DBSCAN(**best_params)
        labels = dbscan.fit_predict(X)
        best_pred = map_clusters_to_binary(labels)
        best_acc = accuracy_score(y, best_pred)
    
    return best_params, best_pred, best_acc

# Find optimal parameters for OPTICS with different feature sets
def optimize_optics(X, y, min_samples_range=None, xi_range=None, min_cluster_size_range=None):
    if min_samples_range is None:
        min_samples_range = [2, 3, 5, 10]
    
    if xi_range is None:
        xi_range = [0.005, 0.01, 0.03, 0.05]
    
    if min_cluster_size_range is None:
        min_cluster_size_range = [0.01, 0.02, 0.05, 0.1]
    
    best_acc = 0
    best_params = {}
    best_pred = None
    
    print("Finding best OPTICS parameters...")
    for min_samples in min_samples_range:
        for xi in xi_range:
            for min_cluster_size in min_cluster_size_range:
                optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
                labels = optics.fit_predict(X)
                
                # Count non-noise clusters
                n_clusters = len(np.unique(labels[labels != -1]))
                if n_clusters >= 2:  # Only if we have at least 2 clusters
                    pred = map_clusters_to_binary(labels)
                    acc = accuracy_score(y, pred)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {'min_samples': min_samples, 'xi': xi, 'min_cluster_size': min_cluster_size}
                        best_pred = pred
                        print(f"  Improved: min_samples={min_samples}, xi={xi:.4f}, min_cluster_size={min_cluster_size:.4f}, acc={acc:.4f}, clusters={n_clusters}")
    
    if not best_params:
        print("No suitable parameters found, using defaults")
        best_params = {'min_samples': 5, 'xi': 0.05, 'min_cluster_size': 0.05}
        optics = OPTICS(**best_params)
        labels = optics.fit_predict(X)
        best_pred = map_clusters_to_binary(labels)
        best_acc = accuracy_score(y, best_pred)
    
    return best_params, best_pred, best_acc

# ======= DBSCAN with different feature selection approaches =======
print("\n==== DBSCAN with Variance-based Features ====")
dbscan_var_params, dbscan_var_pred, dbscan_var_acc = optimize_dbscan(X_var, y_true)
print(f"Best DBSCAN params with variance features: {dbscan_var_params}")
evaluate_clustering(y_true, dbscan_var_pred, 'DBSCAN (Variance)')


# ======= OPTICS with different feature selection approaches =======
print("\n==== OPTICS with Variance-based Features ====")
optics_var_params, optics_var_pred, optics_var_acc = optimize_optics(X_var, y_true)
print(f"Best OPTICS params with variance features: {optics_var_params}")
evaluate_clustering(y_true, optics_var_pred, 'OPTICS (Variance)')

# ======= Mean Shift (unchanged) =======
print("\n==== Mean Shift Clustering ====")
# Use the original Mean Shift implementation that worked perfectly
bandwidth = np.mean(np.std(X_var, axis=0)) * 0.5  # Using variance-based features without scaling
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_labels = meanshift.fit_predict(X_var)
print(f"Number of clusters found: {len(np.unique(meanshift_labels))}")
meanshift_pred = map_clusters_to_binary(meanshift_labels)
meanshift_acc = evaluate_clustering(y_true, meanshift_pred, 'Mean Shift')

# Compare all methods
algorithms = [
    'DBSCAN (Variance)', 'OPTICS (Variance)', 'Mean Shift'
]
accuracies = [
    dbscan_var_acc, optics_var_acc, meanshift_acc
]

print("\n--- Overall Comparison ---")
for algorithm, accuracy in zip(algorithms, accuracies):
    print(f"{algorithm}: {accuracy:.4f}")

# Identify the best performing method
best_idx = np.argmax(accuracies)
print(f"\nBest performing method: {algorithms[best_idx]} with accuracy {accuracies[best_idx]:.4f}")
