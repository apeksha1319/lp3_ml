Clustering is an unsupervised learning technique used to group similar data points together based on their features. Here, we’ll go through the steps of implementing K-Means Clustering on a sample dataset (sales_data_sample.csv) and use the Elbow Method to determine the optimal number of clusters. We’ll also discuss hierarchical clustering briefly.

Theory Behind K-Means Clustering

K-Means is a centroid-based clustering algorithm that works as follows:

1. Initialize: Define the number of clusters  and randomly select  points as initial centroids.


2. Assign: Assign each data point to the nearest centroid, forming  clusters.


3. Update: Calculate the mean of each cluster and move the centroids to these means.


4. Repeat: Steps 2 and 3 are repeated until convergence (when centroids don’t change or minimal improvement occurs).



Determining Optimal Number of Clusters: The Elbow Method

The Elbow Method helps determine the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS) for different values of . WCSS measures the variance within each cluster:

\text{WCSS} = \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2

Theory Behind Hierarchical Clustering

Hierarchical Clustering builds a hierarchy of clusters, typically represented by a dendrogram:

1. Agglomerative (bottom-up): Start with each point as its own cluster and iteratively merge the closest clusters until only one remains.


2. Divisive (top-down): Start with one cluster and iteratively split it until each point is in its own cluster.



Distance Metrics for Hierarchical Clustering:

Single Linkage: Minimum distance between points across clusters.

Complete Linkage: Maximum distance between points across clusters.

Average Linkage: Average distance between all points across clusters.


Practical Implementation Using K-Means and the Elbow Method

Let’s go through a Python implementation for K-Means clustering with the Elbow Method on sales_data_sample.csv. We'll assume this dataset contains sales records with relevant numerical features (e.g., sales amounts, quantities).

Required Libraries

To implement this, we’ll use pandas to handle the data, KMeans from sklearn.cluster for K-Means clustering, and matplotlib for plotting.

Code Implementation

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('sales_data_sample.csv')

# Select relevant numerical features for clustering (assume they are available in the dataset)
# Here, we use only numerical features. Modify this selection as needed based on the dataset.
features = data[['SALES', 'QUANTITYORDERED']]  # Replace with actual numerical column names

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # List to store the Within-Cluster Sum of Squares for each k

# Try k values from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS for the given k

# Plot the Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

Explanation of the Code

1. Data Loading and Selection: We load the dataset and select relevant numerical features for clustering.


2. Elbow Method: For each  from 1 to 10, we fit the K-Means model, store the WCSS (using kmeans.inertia_), and plot the WCSS values.


3. Elbow Point: The "elbow" point in the plot indicates the optimal number of clusters, where the WCSS starts to decrease more slowly.



After finding the optimal , we can apply K-Means clustering to the dataset.

Applying K-Means Clustering with the Optimal 

# Based on the elbow method, assume the optimal k is determined to be 3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)  # Assign each point to a cluster

# Plot the clusters
plt.scatter(data['SALES'], data['QUANTITYORDERED'], c=data['Cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Sales')
plt.ylabel('Quantity Ordered')
plt.colorbar(label='Cluster')
plt.show()

This code will plot the clusters with colors representing each cluster, helping to visualize the segmentation created by K-Means.

Hierarchical Clustering

For comparison, hierarchical clustering can also be applied to the dataset. Here's a brief code snippet to perform hierarchical clustering and plot a dendrogram:

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Perform hierarchical clustering
Z = linkage(features, method='ward')  # 'ward' minimizes variance within clusters

# Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
dendrogram(Z)
plt.show()

In the dendrogram, the vertical line at which clusters merge provides an idea of where to cut to form clusters.

Interpreting Results

Optimal Number of Clusters: For K-Means, the Elbow Method suggests the optimal . For hierarchical clustering, you can cut the dendrogram at a specific height to get clusters.

Cluster Analysis: Each cluster represents a group of data points with similar sales and quantity patterns, providing valuable insights for business strategies.


Practical Applications

Customer Segmentation: In sales data, clustering can help segment customers or products based on purchase patterns.

Inventory Management: Identifying clusters in sales and quantity data helps in categorizing fast-moving and slow-moving items.


This approach demonstrates how to use K-Means clustering to analyze sales data, determine the number of clusters, and visualize the results.