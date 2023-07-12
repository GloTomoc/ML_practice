import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    st.title("Seed Clustering")
    
    # Generate random seed data
    np.random.seed(123)
    seeds = np.random.rand(100, 2)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(seeds)
    
    # Get the cluster labels
    labels = kmeans.labels_
    
    # Visualize the clusters
    st.subheader("Cluster Visualization")
    plot_clusters(seeds, labels)
    
def plot_clusters(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Cluster Visualization")
    st.pyplot(plt)

if __name__ == '__main__':
    main()
