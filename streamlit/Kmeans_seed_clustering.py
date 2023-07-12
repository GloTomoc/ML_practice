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
    
    # Sidebar buttons
    st.sidebar.subheader("Actions")
    if st.sidebar.button("Show Data"):
        st.dataframe(seeds, width=400)
    if st.sidebar.button("Show Cluster Labels"):
        st.write(labels)
    
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
