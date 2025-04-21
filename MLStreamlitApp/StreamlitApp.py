# Import all the necessary modules/packages/libraries necessary to run this app

# This imports the streamlit library and shortens the name to make the coding a little faster
import streamlit as st
# This imports pandas for data manipulation and shortens it to pd to make the coding a little faster
import pandas as pd
# From scikit-learn this imports the specific function train_test_split
from sklearn.model_selection import train_test_split # type: ignore
# From scikit-learn this import Label Encoder
from sklearn.preprocessing import LabelEncoder # type: ignore
# This imports the specific metric functions we want to use 
# (accuracy_score, classification_report, roc_auc_score, and RocCurveDisplay) instead of the entire model
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay # type: ignore
# This imports a specific model class - for Logisitic Regression
from sklearn.linear_model import LogisticRegression # type: ignore
# This imports a specific model class - for Decision Trees
from sklearn.tree import DecisionTreeClassifier # type: ignore
# This imports a specific model class - for KNN models
from sklearn.neighbors import KNeighborsClassifier # type: ignore
# This imports matplotlib's plotting module as plt for faster coding
import matplotlib.pyplot as plt


# This sets the title of the web app on the side bar - I'm calling it 'Interactive Machine Learning Experience'
st.sidebar.title("Interactive Machine Learning Experience")

# This sets the title of the web app - I'm calling it 'Interactive Machine Learning Experience'
st.title("Interactive Machine Learning Experience")

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

# This gives further instruction to the user for the next part
st.markdown("""
            Now you will select your target (what you are trying to predict) as well as your features 
            (what you will use to predict your target). You must choose one target and at least one feature to proceed
            """)

# Titles the section as the Target and Feature Selection step
st.sidebar.title("2. Select Target and Features")

# This creates a dropdown menu so the user can select the target column
target_col = st.sidebar.selectbox("Target (what you want to predict)", df.columns)

# This creates a dropdown menu so the user can multiselect features - they may not select the 
# target as a feature if it has already been chosen 
feature_cols = st.sidebar.multiselect("Features (these are used to predict the target)",
                                      [col for col in df.columns if col != target_col])

# This ensures that the user selects at least one feature otherwise it raises an error warning prompting
# them to select at least one feature column
if not feature_cols:
    st.warning("You must select at least one feature column")
    st.stop()

# This creates arrays for the features and target for further analysis
X = df[feature_cols]
y = df[target_col]

# This step makes sure that if the user selected target is categorical it will encode it into 
# numbers for easier analysis
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Now we are going to split the data
# We will use test_train_split with 80% train and 20% test and random state 42 to ensure we can reproduce these results 
# (I also googled why we use random state = 42 and learned it's a Hitchhiker's Guide to the Galaxy reference so you learn something new every day)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.markdown("""
            Now you will choose which Model you would like to represent your data as.
            You may choose between Logistic Regression, Decision Tree, and K-Nearest Neighbors.
            You will also be able to tune the hyperparameters within each model
            """)

# This titles the section as Choose Model and Tune Hyperparameters
st.sidebar.title("3. Choose Model and Tune Hyperparameters")

# This creates a dropdown menu to select which model the user would like to choose
# They have the options of Logistic Regression, Decision Tree, and K-Nearest Neighbors
model_name = st.sidebar.selectbox("Choose a model",
                                  ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

# Depending on which model was chosen different hyperparameter selection sliders will appear
st.subheader("Hyperparameters Explanations:")
st.markdown("""
            If you choose Logistic Regression you will be prompted to select the regularization strength (C).
            This factor helps make sure the model is not overfitted.
            """)
st.markdown("""
            If you choose Decision Tree you will be prompted to select the max depth of the Decision Tree model.
            This controls the max depth or levels that the tree can grow to - it will help avoid overfitting.
            """)
st.markdown("""
            If you choose K-Nearest Neighbors you will be prompted to select the number of neighbors (k) that are
            considered when predicting the target. This will help to avoid overfitting the model.
            """)
# If Logistic Regression is chosen the app will prompt the user to select the regularization strength (C) - this helps make sure the model is not overfitted 
if model_name == "Logistic Regression":
    # Creates a slider to set the regularization strength of the model
    C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
    # Creates the Logistic Regression model
    model = LogisticRegression(C=C, max_iter=1000)
# If Decision Tree is chosen the app will prompt the user to select the max depth of the Decision Tree model
# This controls the max depth the tree can grow to - can help the user avoid overfitting 
elif model_name == "Decision Tree":
    # Creates a sliding bar to choose the max depth of the Decision Tree
    max_depth = st.sidebar.slider("Max Tree Depth", 1, 20, 5)
    # Creates the Decision Tree model
    model = DecisionTreeClassifier(max_depth=max_depth)
# If K-Nearest Neighbors is chosen the app will prompt the user to choose the number of neighbors (k)
# The number of neighbors (k) controls how many neighbors are considered when predicting the target - also helps to control with overfitting
elif model_name == "K-Nearest Neighbors":
    # Creates a sliding bar to choose the number of neighbors for the model
    k = st.sidebar.slider("Number of Neighbors (k)", 1, 20,5)
    # Creates the K-Nearest Neighbors model
    model = KNeighborsClassifier(n_neighbors=k)


# Train the model
# This fits the model to the training data
model.fit(X_train, y_train)
# This predicts on the test data
y_pred = model.predict(X_test)


# Model Performance Metrics 

st.subheader("Model Performance Metrics")

st.markdown("""
            Accuracy: This shows the percentage of correct predictions made by the model
            (generally a higher accuracy score is better)
            """)
st.markdown("""
            Classification Report: This breaks down the performance for each class and shows 
            the precision (how many selected items were relevant in creating the model), 
            Recall (how many relevant items were selected), and F1-Score 
            (which shows the balance between precision and recall)
            """)
st.markdown("""
            ROC Curve (Only for binary problems): This visual shows how well the model distiguishes 
            between two classes at different thresholds. In genral, the closer the curve is to the
             top-left corner of the graph, the better the model is.
            """)

# This displays the accuracy score
st.write("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))

# This displays the classification report (details explained in markdown comments)
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# ROC Curve
# This ensures that only binary models display an ROC curve - otherwise there would be an error message displayed
if len(set(y)) == 2:
    st.subheader("ROC Curve")

    # This graph shows how well the model separates classes at different thresholds
    try:
        y_prob = model.predict_proba(X_test)[:,1]

        roc_display = RocCurveDisplay.from_predictions(y_test, y_prob)
        st.pyplot(plt.gcf())
    # This error will display if the ROC curve is not able to be shown for the given model
    except:
        st.warning("ROC Curve is not supported for this model")




