import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Student Performance Dashboard", page_icon=":mortar_board:", layout="wide")

st.title("Student Performance Prediction Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Validate required columns
    required_columns = {'hours_studied', 'final_grade', 'extra_curricular'}
    if not required_columns.issubset(df.columns):
        st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
    else:
        # Data preprocessing
        df['extra_curricular'] = df['extra_curricular'].map({'Yes': 1, 'No': 0})
        
        # Sidebar for model parameters
        st.sidebar.header("Model Parameters")
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.1)
        random_state = st.sidebar.number_input("Random State", 0, 100, 42)

        # Model Training
        X = df.drop(columns=['student_id', 'final_grade'])
        y = df['final_grade']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Model Metrics", "Visualizations", "Data Preview"])

        with tab1:
            st.subheader("Model Evaluation Metrics")
            col1, col2 = st.columns(2)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            col1.metric("Mean Squared Error", f"{mse:.2f}")
            col2.metric("R-squared", f"{r2:.2f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.coef_
            })
            st.bar_chart(importance_df.set_index('Feature'))

        with tab2:
            st.subheader("Data Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6,4))
                sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax1)
                st.pyplot(fig1)
                
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6,4))
                sns.boxplot(data=df, x='extra_curricular', y='final_grade', ax=ax2)
                st.pyplot(fig2)
                
            # Scatter plot
            fig3, ax3 = plt.subplots(figsize=(8,4))
            sns.scatterplot(data=df, x='hours_studied', y='final_grade', hue='extra_curricular')
            st.pyplot(fig3)

        with tab3:
            st.subheader("Sample Data Preview")
            st.dataframe(df)
            
            # Download button for processed data
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download processed data as CSV",
                data=csv,
                file_name='processed_data.csv',
                mime='text/csv',
            )
else:
    # Sample Data as fallback
    data = {'student_id': [1, 2, 3, 4, 5],
            'hours_studied': [2, 5, 3, 7, 4], 
            'extra_curricular': ['Yes', 'No', 'Yes', 'Yes', 'No'],
            'final_grade': [60, 80, 70, 90, 75]}
    df = pd.DataFrame(data)
    
    # Data preprocessing
    df['extra_curricular'] = df['extra_curricular'].map({'Yes': 1, 'No': 0})

    # Sidebar for model parameters
    st.sidebar.header("Model Parameters")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.1)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)

    # Model Training
    X = df.drop(columns=['student_id', 'final_grade'])
    y = df['final_grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Model Metrics", "Visualizations", "Data Preview"])

    with tab1:
        st.subheader("Model Evaluation Metrics")
        col1, col2 = st.columns(2)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        col1.metric("Mean Squared Error", f"{mse:.2f}")
        col2.metric("R-squared", f"{r2:.2f}")
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.coef_
        })
        st.bar_chart(importance_df.set_index('Feature'))

    with tab2:
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax1)
            st.pyplot(fig1)
            
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.boxplot(data=df, x='extra_curricular', y='final_grade', ax=ax2)
            st.pyplot(fig2)
            
        # Scatter plot
        fig3, ax3 = plt.subplots(figsize=(8,4))
        sns.scatterplot(data=df, x='hours_studied', y='final_grade', hue='extra_curricular')
        st.pyplot(fig3)

    with tab3:
        st.subheader("Sample Data Preview")
        st.dataframe(df)
        
        # Download button for processed data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download processed data as CSV",
            data=csv,
            file_name='processed_data.csv',
            mime='text/csv',
        )
