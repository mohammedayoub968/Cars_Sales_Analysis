# 🚗 Car Sales Analysis Dashboard

This is my first Data Analysis project, where I explored and visualized a dataset of 100,000+ vehicles using Python, Pandas, Matplotlib, Seaborn, and Streamlit.

The project aims to analyze the car market trends by brand, model, year, condition, and more — and present the findings through an interactive dashboard.

---

## 📊 Dataset

- **Source**: [Kaggle - Car Dataset](https://www.kaggle.com/datasets/zain280/car-dataset)
- **Size**: 100,000+ rows
- **Columns**:
  - `ID`: Unique identifier
  - `Brand`: Manufacturer (e.g., Ford, Lexus)
  - `Model`: Model name
  - `Year`: Year of manufacture
  - `Color`: Vehicle color
  - `Mileage`: Distance traveled (miles)
  - `Price`: Price in USD
  - `Condition`: New or Used

---

## 📌 Objectives

- Clean and explore the dataset
- Analyze key performance indicators (KPIs)
- Identify trends in price, mileage, and sales by year
- Compare performance across brands and models
- Build a fully interactive report using Streamlit

---

## 📈 Key Insights

- Total cars: `100,000`
- Total revenue: `$4.25 Billion`
- Average revenue per car: `$42,000`
- Most sold year: `2017`
- Most used color by year
- Average car age: `19 years`
- Price per mile: `~$3.15`

---

## 🧪 Tools & Technologies

- Python 3
- Pandas
- Matplotlib
- Seaborn
- Streamlit

---

## 🚀 Live Demo

👉 Try the live interactive dashboard here:  
[🔗 Streamlit App](https://carssalesanalysis-hxhxnyaahgrhtyjckxtsdd.streamlit.app/)

---

## 💻 Run Locally

To run this project on your machine:

```bash
# 1. Clone the repo
git clone https://github.com/mohammedayoub968/Cars_Sales_Analysis.git
cd Cars_Sales_Analysis

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate.bat   # For Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run streamlit_app.py
