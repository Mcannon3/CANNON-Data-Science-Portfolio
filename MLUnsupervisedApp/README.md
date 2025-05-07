# Cannon Unsupervised Machine Learning Streamlit App
This project is an interactive Streamlit app that allows users to upload a dataset and explore unsupervised machine learning techniques
like K-Means Clustering, Hierarchical Clustering, and Principal Component Analysis (PCA). This app helps users to visualize and better
understand patterns in unlabeled data.

## Project Overview
With this app users can:
- Upload their own CSV dataset
- Choose between 3 different unsupervised models
- Tune model-specific hyperparameters
- Visualize results through scatter plots, dendrograms, and explained variance ratios

## Instructions

### To run the app locally
1. Clone the GitHub Repository
2. Create and Activate a Virtual Environment
3. Install Dependencies using the requirements.txt file and the command 'pip install -r requirements.txt'
4. Run the streamlit app using the command 'streamlit run StreamlitApp.py'

The app should automatically open the app in your default web browser.

### To run the app using the deployed version click this link
https://cannon-data-science-portfolio-eplzvmmyptdbrr773686mm.streamlit.app/

## Example Usage
1. Launch the App using either of the two methods listed in the Instructions section
2. Within the app upload a CSV file
3. Select Target and Features
4. Choose a Model (K-Means Clustering, Hierarchical Clustering, Principal Component Analysis)
5. Tune Hyperparameters (Number of clusters, Linkage Method, Number of Principal Components)
6. Review Model Performance

## Visual Examples
![Example Output from my ML Streamlit App](https://github.com/Mcannon3/CANNON-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/App%20Overview.png)
![Example Output from my ML Streamlit App](https://github.com/Mcannon3/CANNON-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/PCA%20Projection.png)
![Example Output from my ML Streamlit App](https://github.com/Mcannon3/CANNON-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/Dendrogram.png)
![Example Output from my ML Streamlit App](https://github.com/Mcannon3/CANNON-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/Elbow%20Plot.png)
