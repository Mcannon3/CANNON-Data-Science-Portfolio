# Import all the necessary modules/packages/libraries necessary to run this app

# This imports the streamlit library and shortens the name to make the coding a little faster
import streamlit as st
# This imports pandas for data manipulation and shortens it to pd to make the coding a little faster
import pandas as pd
# This imports numpy to help with numerical operations and shortens it to np to make the coding a little faster
import numpy as np
# This imports unsupervised clustering models from the sklearn library
from sklearn.cluster import KMeans, AgglomerativeClustering
# This imports Principal Component Analysis from the sklearn library
from sklearn.decomposition import PCA
# This imports the silhouette score which helps evaluate how well clusters are formed
from sklearn.metrics import silhouette_score
# This will help us create hierarchical clustering visuals
from scipy.cluster.hierarchy import dendrogram, linkage
# This imports matplot library for plotting charts and making graphs and shortens it to plt for faster coding
import matplotlib.pyplot as plt
# This imports seaborn to help with data visualizations and shortens it to sns for faster coding
import seaborn as sns

# This sets the title of the web app on the side bar - I'm calling it 'Interactive Unsupervised Machine Learning Experience'
st.sidebar.title("Interactive Unsupervised Machine Learning Experience")

# This sets the title of the web app - I'm calling it 'Interactive Unsupervised Machine Learning Experience'
st.title("Interactive Unsupervised Machine Learning Experience")

# This uses a markdown comment to give the user some instructions and what they can expect from the app
st.markdown("""
            Upload your own CSV dataset and experiment with different supervised learning models!
            You will be able to tweak model hyperparameters and observe how these affect model trainig and performance.
            """)


# Creates a sidebar title with the instruction to upload a dataset
st.sidebar.title("1. Upload Your Dataset")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Read uploaded CSV into a DataFrame using session state
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Save to session state so that as the hyperparameters are changed the models will update
    st.session_state['df'] = df  
elif 'df' in st.session_state:
    df = st.session_state['df']
else:
    # Will not allow the user to continue without a dataset being uploaded
    st.warning("Please upload a dataset to continue.")
    st.stop()

# This will give users a chance to preview their dataset and make sure they uploaded the correct one
st.subheader("Preview Your Dataset")
st.dataframe(df.head())

# This makes sure that we only select numerical columns for modeling because clustering and PCA both work with numeric data)
numeric_df = df.select_dtypes(include=np.number)

# This gives the user the option to choose which unsupervised model to run - K-Means, Hierarchical Clustering, or PCA
model_choice = st.sidebar.selectbox("Choose Unsupervised Model", ["K-Means", "Hierarchical Clustering", "PCA"])

# K-Means Clustering

if model_choice == "K-Means":
    # This lets the user choose the number of clusters
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)

    # This will initialize and fit the K-Means model
    kmeans = KMeans(n_clusters=k, random_state=42)
    # This assigns cluster labels to each sample
    clusters = kmeans.fit_predict(numeric_df)  
    # This shows a scatter plot of the first two features
    st.write("Cluster Scatter Plot (First 2 Features)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=numeric_df.iloc[:, 0], y=numeric_df.iloc[:, 1], hue=clusters, palette="tab10", ax=ax)
    st.pyplot(fig)

    # This will display the silhouette score to help the user evaluate clustering quality
    st.write("Silhouette Score:", silhouette_score(numeric_df, clusters))

    # Elbow plot: Shows inertia for different k values to help choose the best k
    st.write("Elbow Method")
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(numeric_df)
        distortions.append(km.inertia_)  

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), distortions, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (WCSS)')
    ax.set_title('Elbow Plot for Optimal k')
    st.pyplot(fig)

# Hierarchical Clustering 

elif model_choice == "Hierarchical Clustering":
    # This allows the user to select a linkage method - either ward, complete, average, or single
    method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])

    # This fits the hierarchical clustering model
    hc = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = hc.fit_predict(numeric_df)

    # Then we'll display a dendrogram to visualize the hierarchy
    st.write("Dendrogram")
    linked = linkage(numeric_df, method=method)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, ax=ax)
    st.pyplot(fig)

    # Creates a scatter plot with hierarchical cluster labels
    st.write("### Cluster Scatter Plot (First 2 Features)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=numeric_df.iloc[:, 0], y=numeric_df.iloc[:, 1], hue=labels, palette="tab10", ax=ax)
    st.pyplot(fig)

# Principal Component Analysis (PCA) 

elif model_choice == "PCA":
    # This allows the user to select the number of PCA components
    n_components = st.sidebar.slider("Number of Components", 2, min(5, numeric_df.shape[1]), 2)

    # This fits the PCA model and transforms the data
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(numeric_df)

    # This creates a scatter plot of first two principal components and displays it
    st.write("PCA Scatter Plot (First 2 Components)")
    fig, ax = plt.subplots()
    ax.scatter(components[:, 0], components[:, 1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection")
    st.pyplot(fig)

    # This will display the explained variance for each component
    st.write("### Explained Variance Ratio")
    st.write(pca.explained_variance_ratio_)

