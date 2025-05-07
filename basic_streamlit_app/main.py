# import necessary libraries
import streamlit as st
import pandas as pd

# This displays the App title
st.title("Penguin Data Explorer")

# This gives the user some background into what the app does
st.write("This interactive app will load data from the Palmer Archipelago" \
"penguins and allows you to filter by species and display informational statistics.")

# This loads the dataset in from the github url and stores it as a dataframe
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

# This displays a preview of the dataset for the user
st.subheader("Data Preview")
st.dataframe(df.head())

# This provides a sidebar filter 
st.sidebar.header("Filter Options")

# This creates the species filter
species_list = df['species'].unique()
selected_species = st.sidebar.multiselect("Select Species:", species_list, default=species_list)

# This creates the sex filter
sex_list = df['sex'].unique()
selected_sex = st.sidebar.multiselect("Select Sex:", sex_list, default=sex_list)

# This filters the dataset based on the user's selections
filtered_df = df[
    (df['species'].isin(selected_species)) &
    (df['sex'].isin(selected_sex))
]

# This displays the filtered data for the user
st.subheader(f"Filtered Data ({len(filtered_df)} records)")
st.dataframe(filtered_df)

# This shows the summary statistics for the user
st.subheader("Summary Statistics")
st.write(filtered_df.describe())

