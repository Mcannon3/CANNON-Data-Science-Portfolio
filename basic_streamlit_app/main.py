import streamlit as st
import pandas as pd

st.title("Basic Streamlit App")

st.write("This app displays sample data and allows you to filter it by species.")

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)


st.write("Sample of the data:")
st.dataframe(df.head())

species = st.selectbox("Select species to filter by:", df['species'].unique())
filtered_data = df[df['species'] == species]

st.write(f"Data filtered by {species}:")
st.dataframe(filtered_data)
