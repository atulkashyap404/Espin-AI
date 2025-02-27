import streamlit as st
import pandas as pd
import plotly.express as px

# 📌 Page Title
st.title("📊 E-SPIN Advanced Data Visualization")

# 📂 File Upload Section
st.subheader("📂 Upload a Data File (CSV, Excel, TSV)")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])

# Initialize session state for data
if "chart_data" not in st.session_state:
    st.session_state.chart_data = None

# ✅ Function to Load Data from Various File Formats
def load_data(file):
    """Reads data from CSV, Excel (XLSX), or TSV formats and returns a DataFrame."""
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file, engine="openpyxl")
        elif file.name.endswith(".tsv"):
            return pd.read_csv(file, sep="\t")
        else:
            st.error("❌ Unsupported file format! Please upload CSV, XLSX, or TSV.")
            return None
    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")
        return None

# ✅ Process Uploaded File
if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None and df.shape[1] >= 2:
        st.session_state.chart_data = df
    else:
        st.error("⚠️ The file must contain at least two columns.")

# ✅ Display Data and Charts
if st.session_state.chart_data is not None:
    st.subheader("🔍 Data Preview")
    st.dataframe(st.session_state.chart_data)

    # Select X and Y Columns
    x_column = st.selectbox("Select X-axis column", st.session_state.chart_data.columns)
    y_column = st.selectbox("Select Y-axis column", st.session_state.chart_data.columns)

    # 📈 Line Chart
    st.subheader("📈 Line Chart")
    st.line_chart(st.session_state.chart_data, x=x_column, y=y_column)

    # 📊 Bar Chart
    st.subheader("📊 Bar Chart")
    st.bar_chart(st.session_state.chart_data, x=x_column, y=y_column)

    # 📉 Area Chart
    st.subheader("📉 Area Chart")
    st.area_chart(st.session_state.chart_data, x=x_column, y=y_column)

    # 🔵 Scatter Plot
    st.subheader("🔵 Scatter Plot")
    fig_scatter = px.scatter(st.session_state.chart_data, x=x_column, y=y_column, title="Scatter Plot")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 📏 Histogram
    st.subheader("📏 Histogram")
    fig_hist = px.histogram(st.session_state.chart_data, x=x_column, title="Histogram")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 🥧 Pie Chart (Only if X column has fewer than 20 unique values)
    if st.session_state.chart_data[x_column].nunique() < 20:
        st.subheader("🥧 Pie Chart")
        fig_pie = px.pie(st.session_state.chart_data, names=x_column, values=y_column, title="Pie Chart")
        st.plotly_chart(fig_pie, use_container_width=True)

    # 📦 Box Plot
    st.subheader("📦 Box Plot")
    fig_box = px.box(st.session_state.chart_data, x=x_column, y=y_column, title="Box Plot")
    st.plotly_chart(fig_box, use_container_width=True)

    # 🔄 Reset Button
    if st.button("🔄 Reset Data"):
        st.session_state.chart_data = None
        st.rerun()
else:
    st.info("📂 Please upload a CSV, Excel, or TSV file to visualize data.")
