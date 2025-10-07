import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('../../data/processed/macroeconomic.csv')

# Streamlit title and description
st.title("Macroeconomic Data Dashboard")
st.write("Explore macroeconomic indicators like GDP Growth, Inflation, and Unemployment across countries.")

# Sidebar for user interaction
country_selection = st.sidebar.selectbox('Select Country', df['Country Name'].unique())
year_range = st.sidebar.slider(
    'Select Year Range', 
    min_value=int(df['Year'].min()), 
    max_value=int(df['Year'].max()), 
    value=(int(df['Year'].min()), int(df['Year'].max()))
)

# Filter data based on selection
filtered_df = df[(df['Country Name'] == country_selection) & 
                 (df['Year'] >= year_range[0]) & 
                 (df['Year'] <= year_range[1])]

# Show data in table format
st.subheader(f"Data for {country_selection} ({year_range[0]} - {year_range[1]})")
st.dataframe(filtered_df)

# GDP Growth plot
gdp_fig = px.line(filtered_df, x='Year', y='GDP_Growth', title=f"GDP Growth for {country_selection}")
gdp_fig.update_layout(xaxis_title='Year', yaxis_title='GDP Growth (%)')
st.plotly_chart(gdp_fig)

# Inflation plot
inflation_fig = px.line(filtered_df, x='Year', y='Inflation', title=f"Inflation for {country_selection}")
inflation_fig.update_layout(xaxis_title='Year', yaxis_title='Inflation Rate (%)')
st.plotly_chart(inflation_fig)

# Unemployment plot
unemployment_fig = px.line(filtered_df, x='Year', y='Unemployment', title=f"Unemployment for {country_selection}")
unemployment_fig.update_layout(xaxis_title='Year', yaxis_title='Unemployment Rate (%)')
st.plotly_chart(unemployment_fig)

# Show a summary of the data
st.subheader("Summary Statistics")
st.write(filtered_df.describe())
