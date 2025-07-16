import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


cars = pd.read_csv('Car_Data.csv')

st.title("Car Market Data Analysis Report")
st.markdown("""
This report analyzes car market data including pricing, mileage, condition, brand performance, and trends over time.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Total Cars", len(cars))
col2.metric("Total Revenue", f"${cars['Price'].sum():,.0f}")
col3.metric("Average Price", f"${cars['Price'].mean():,.2f}")



# Step 1: Select Brand
brands = cars["Brand"].unique()
selected_brand = st.selectbox("Select a Brand", brands)

# Step 2: Select Condition
conditions = ["All"] + cars["Condition"].dropna().unique().tolist()
selected_condition = st.selectbox("Select Condition", conditions, key="condition_filter")

# Filter brand data
brand_data = cars[cars["Brand"] == selected_brand]

# Apply condition filter if not "All"
if selected_condition != "All":
    brand_data = brand_data[brand_data["Condition"] == selected_condition]

# Stats per model after filtering
model_stats = brand_data.groupby('Model')['Price'].agg(
    Cars_Sold='count',
    Total_Revenue='sum',
    Avg_Revenue='mean'
).reset_index().sort_values(by='Cars_Sold', ascending=False)

# Step 3: General KPIs for the Brand
st.subheader(f"📊 General KPIs for {selected_brand} ({selected_condition})")
total_cars_brand = brand_data.shape[0]
total_revenue_brand = brand_data["Price"].sum()
num_models_in_brand = brand_data["Model"].nunique()
avg_revenue_per_car = brand_data["Price"].mean()

col1, col2 = st.columns(2)
col1.metric("Total Cars Sold", total_cars_brand)
col2.metric("Total Revenue", f"${total_revenue_brand:,.0f}")

col3, col4 = st.columns(2)
col3.metric("Number of Models", num_models_in_brand)
col4.metric("Avg Revenue per Car", f"${avg_revenue_per_car:,.0f}")

# Step 4: Select a Model inside that Brand
st.subheader(f"📦 Cars Sold per Model in {selected_brand}")

models = brand_data["Model"].unique()
selected_model = st.selectbox(f"Select a Model in {selected_brand}", models)

# Filter model data
model_data = brand_data[brand_data["Model"] == selected_model]
total_cars_model = model_data.shape[0]
total_revenue_model = model_data["Price"].sum()
avg_revenue_model = model_data["Price"].mean()

# Step 5: KPIs for Model
st.subheader(f"🔍 KPIs for Model: {selected_model}")
col5, col6 = st.columns(2)
col5.metric("Cars Sold", total_cars_model)
col6.metric("Total Revenue", f"${total_revenue_model:,.0f}")

col7, _ = st.columns(2)
col7.metric("Avg Revenue", f"${avg_revenue_model:,.0f}")


st.subheader(f"📦 Cars Sold per Model in {selected_brand}")

fig1 = px.bar(
    model_stats,
    x='Model',
    y='Cars_Sold',
    text='Cars_Sold',
    title=f"Number of Cars Sold by Model in {selected_brand}",
    labels={'Cars_Sold': 'Cars Sold'},
    color='Cars_Sold',
    color_continuous_scale='Blues',
)

fig1.update_traces(texttemplate='%{text}', textposition='outside')
fig1.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig1, use_container_width=True)

st.subheader(f"💰 Total Revenue per Model in {selected_brand}")

fig2 = px.bar(
    model_stats,
    x='Model',
    y='Total_Revenue',
    text='Total_Revenue',
    title=f"Total Revenue by Model in {selected_brand}",
    labels={'Total_Revenue': 'Revenue (USD)'},
    color='Total_Revenue',
    color_continuous_scale='Greens',
)

fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig2.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig2, use_container_width=True)


# حساب عدد السيارات والإيرادات في كل سنة
number_of_cars_in_year = cars.groupby('Year').agg(
    Number_of_sold_cars=('Price', 'count'),
    Average_Revenue=('Price', 'mean'),
    Total_Revenue=('Price', 'sum')
).reset_index()

# 🔽 فلتر بالسنة
st.subheader("📅 Yearly Sales Insights")
selected_year = st.selectbox("Select a Year", number_of_cars_in_year['Year'].sort_values())

# استخراج البيانات الخاصة بالسنة المختارة
year_stats = number_of_cars_in_year[number_of_cars_in_year['Year'] == selected_year].iloc[0]

# عرض الـ KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Number of Cars Sold", year_stats['Number_of_sold_cars'])
col2.metric("Total Revenue", f"${year_stats['Total_Revenue']:,.0f}")
col3.metric("Average Revenue", f"${year_stats['Average_Revenue']:,.2f}")

st.subheader("📈 Number of Cars Sold Over Years")

fig3 = px.line(
    number_of_cars_in_year,
    x="Year",
    y="Number_of_sold_cars",
    title="Cars Sold by Year",
    markers=True,
    labels={"Number_of_sold_cars": "Cars Sold"},
)

fig3.update_traces(mode='lines+markers', marker=dict(size=8))
st.plotly_chart(fig3, use_container_width=True)


st.subheader("💹Total Revenue Over Years")

fig4 = px.line(
    number_of_cars_in_year,
    x="Year",
    y="Total_Revenue",
    title="Total Revenue by Year",
    markers=True,
    labels={"Total_Revenue": "Revenue (USD)"},
)

fig4.update_traces(mode='lines+markers', marker=dict(size=8))
st.plotly_chart(fig4, use_container_width=True)

number_of_cars_in_year = cars.groupby(['Year' , 'Color' ,  'Condition']).agg(
    Number_of_sold_cars=('Price', 'count'),
    Average_Revenue=('Price', 'mean'),
    Total_Revenue=('Price', 'sum')
).reset_index()


st.subheader("🧪 Filtered Sales by Year, Color, and Condition")

# 1. فلترة بالسنة (مع خيار All)
years = sorted(number_of_cars_in_year["Year"].unique().tolist())
years.insert(0, "All")
selected_year = st.selectbox("Select a Year", years, key="year_filter")

# 2. فلترة باللون (مع خيار All)
colors = number_of_cars_in_year["Color"].dropna().unique().tolist()
colors.insert(0, "All")
selected_color = st.selectbox("Select a Color", colors, key="color_filter")

# 3. فلترة بالحالة (مع خيار All)
conditions = number_of_cars_in_year["Condition"].dropna().unique().tolist()
conditions.insert(0, "All")
selected_condition = st.selectbox("Select a Condition", conditions, key="condition_filter_kpi")

# ---------------------
# ✅ 4. فلترة الداتا بناءً على كل الفلاتر
# ---------------------
filtered_data = number_of_cars_in_year.copy()

if selected_year != "All":
    filtered_data = filtered_data[filtered_data["Year"] == selected_year]

if selected_color != "All":
    filtered_data = filtered_data[filtered_data["Color"] == selected_color]

if selected_condition != "All":
    filtered_data = filtered_data[filtered_data["Condition"] == selected_condition]

# ---------------------
# ✅ 5. KPIs بعد الفلترة
# ---------------------
total_cars = filtered_data["Number_of_sold_cars"].sum()
total_revenue = filtered_data["Total_Revenue"].sum()
average_revenue = filtered_data["Average_Revenue"].mean()

st.subheader("📊 Filtered Results")
col1, col2, col3 = st.columns(3)
col1.metric("Total Cars Sold", total_cars)
col2.metric("Total Revenue", f"${total_revenue:,.0f}")
col3.metric("Average Revenue", f"${average_revenue:,.2f}")

# ---------------------
# ✅ 6. رسومات تفاعلية بعد الفلترة
# ---------------------

# نجمع القيم حسب السنة علشان نرسمها
grouped_chart_data = filtered_data.groupby("Year").agg({
    "Number_of_sold_cars": "sum",
    "Total_Revenue": "sum"
}).reset_index()

# 📈 عدد السيارات المباعة حسب السنة
st.subheader("📈 Total Cars Sold per Year (Filtered)")

fig_sold = px.line(
    grouped_chart_data,
    x="Year",
    y="Number_of_sold_cars",
    title="Total Cars Sold per Year",
    markers=True,
    labels={"Number_of_sold_cars": "Cars Sold"},
)
fig_sold.update_traces(mode='lines+markers', marker=dict(size=8))
fig_sold.update_layout(hovermode="x unified")
st.plotly_chart(fig_sold, use_container_width=True)

st.subheader("💹 Total Revenue per Year (Filtered)")

fig_rev = px.line(
    grouped_chart_data,
    x="Year",
    y="Total_Revenue",
    title="Total Revenue per Year",
    markers=True,
    labels={"Total_Revenue": "Revenue (USD)"},
)
fig_rev.update_traces(mode='lines+markers', marker=dict(size=8))
fig_rev.update_layout(hovermode="x unified")
st.plotly_chart(fig_rev, use_container_width=True)

st.subheader("🏆 Top Brands by Total Revenue")
top_n = st.slider("Select number of top brands to display", min_value=1, max_value=20, value=5)

top_brands = cars.groupby('Brand')['Price'].sum().sort_values(ascending=False).head(top_n).reset_index()
top_brands.rename(columns={'Price': 'Total Revenue'}, inplace=True)

top_brands['Formatted Revenue'] = top_brands['Total Revenue'].apply(lambda x: f"${x:,.0f}")

st.subheader(f"🏆 Top {top_n} Brands by Total Revenue")
st.dataframe(top_brands[['Brand', 'Formatted Revenue']])

fig = px.bar(
    top_brands,
    x='Brand',
    y='Total Revenue',
    title=f"Top {top_n} Brands by Total Revenue",
    color='Total Revenue',
    color_continuous_scale='Tealgrn'
)

fig.update_traces(
    hovertemplate='Brand: %{x}<br>Total Revenue: $%{y:,.0f}<extra></extra>'
)

st.plotly_chart(fig, use_container_width=True)

current_year = 2025
cars['Cars_Age'] = current_year - cars['Year']
average_car_age = cars['Cars_Age'].mean()
st.subheader("📆 Car Age Analysis")
st.metric("Average Car Age", f"{average_car_age:.1f} years")


most_popular_color_per_year = cars.groupby('Year')['Color'] \
                                  .agg(lambda x: x.value_counts().idxmax()) \
                                  .reset_index(name='Most_Popular_Color')
st.subheader("🎨 Most Popular Car Color per Year")
st.dataframe(most_popular_color_per_year)
fig_color = px.bar(
    most_popular_color_per_year,
    x='Year',
    y='Most_Popular_Color',
    title='Most Popular Car Color per Year',
    labels={'Most_Popular_Color': 'Color'},
    color='Most_Popular_Color'
)

fig_color.update_layout(
    yaxis_title='Most Popular Color',
    xaxis_title='Year',
    showlegend=False
)

st.plotly_chart(fig_color, use_container_width=True)


cars['Price_per_mile_Ratio'] = cars['Price'] / cars['Mileage']
cars = cars[cars['Mileage'] != 0]
average_price_per_mile = cars['Price_per_mile_Ratio'].mean()

st.subheader("⚖️ Price per Mile Analysis")
st.metric("Average Price per Mile", f"${average_price_per_mile:,.2f}")

best_sell = (
    cars.groupby(['Brand', 'Model'])
    .size()
    .reset_index(name='Sales')
    .sort_values(['Brand', 'Sales'], ascending=[True, False])
    .groupby('Brand')
    .first()
    .reset_index()
)

st.subheader("🏆 Best Selling Model per Brand (Filtered by Condition)")

# فلتر الحالة
conditions = cars["Condition"].unique().tolist()
conditions.insert(0, "All")
selected_condition = st.selectbox("Select Condition", conditions, key="best_model_condition")

# تصفية الداتا حسب الحالة
filtered_cars = cars.copy()
if selected_condition != "All":
    filtered_cars = filtered_cars[filtered_cars["Condition"] == selected_condition]

# حساب أفضل موديل مبيعًا في كل براند (حسب الحالة)
best_sell_filtered = (
    filtered_cars.groupby(['Brand', 'Model'])
    .size()
    .reset_index(name='Sales')
    .sort_values(['Brand', 'Sales'], ascending=[True, False])
    .groupby('Brand')
    .first()
    .reset_index()
)

# عرض جدول النتائج
st.dataframe(best_sell_filtered)

# رسم تفاعلي
import plotly.express as px

fig_best = px.bar(
    best_sell_filtered,
    x='Brand',
    y='Sales',
    color='Model',
    title=f"Best Selling Model per Brand ({selected_condition})",
    labels={'Sales': 'Units Sold'},
    hover_data=['Model']
)

fig_best.update_layout(xaxis_title="Brand", yaxis_title="Units Sold")
st.plotly_chart(fig_best, use_container_width=True)
