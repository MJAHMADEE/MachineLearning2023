# -*- coding: utf-8 -*-
"""AI4021 - HW1 - 2 - Python Tutorial 2 - Sol.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QdKPWHbEIQsdHd3Dyettr6pbI5YgoajJ

**Problem 1:** Create a NumPy array of 20 random integers between 1 and 100. Calculate the mean, median, and standard deviation of the array.
"""

import numpy as np

# Create a NumPy array of 20 random integers
arr = np.random.randint(1, 100, 20)

# Calculate mean, median, and standard deviation
mean = np.mean(arr)
median = np.median(arr)
std_dev = np.std(arr)

print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)

"""**Problem 2:** Create a Pandas DataFrame with three columns - 'Name', 'Age', and 'City' - and at least five rows of data. Perform a basic data exploration by displaying the first 3 rows, the last 2 rows, and a summary of statistics for the 'Age' column."""

import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 22, 35, 28],
        'City': ['New York', 'San Francisco', 'Chicago', 'Los Angeles', 'Seattle']}

df = pd.DataFrame(data)

print("First 3 rows:")
print(df.head(3))

print("\nLast 2 rows:")
print(df.tail(2))

print("\nSummary of 'Age' column:")
print(df['Age'].describe())

"""**Problem 3:** Create a bar chart using Matplotlib to visualize the following data: Monthly sales for a store (January to May). Use appropriate labels and titles for the chart."""

import matplotlib.pyplot as plt

months = ['January', 'February', 'March', 'April', 'May']
sales = [5000, 6000, 7500, 8200, 6900]

plt.bar(months, sales)
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.title('Monthly Sales for Store')
plt.show()

"""**Problem 4:** Create two NumPy arrays, 'arr1' and 'arr2', both of size 5x5 with random integer values. Perform element-wise addition, subtraction, multiplication, and division between the two arrays."""

import numpy as np

arr1 = np.random.randint(1, 10, (5, 5))
arr2 = np.random.randint(1, 10, (5, 5))

addition = arr1 + arr2
subtraction = arr1 - arr2
multiplication = arr1 * arr2
division = arr1 / arr2

print("Addition:\n", addition)
print("Subtraction:\n", subtraction)
print("Multiplication:\n", multiplication)
print("Division:\n", division)

"""**Problem 5:** Given a Pandas DataFrame with information about employees, filter and display only the employees who are older than 30 years."""

import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 28]}

df = pd.DataFrame(data)

filtered_df = df[df['Age'] > 30]
print(filtered_df)

"""**Problem 6:** Create a scatter plot using Matplotlib to visualize a dataset of 100 random (x, y) coordinates. Label the axes appropriately and give the plot a title."""

import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100)
y = np.random.rand(100)

plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Random Scatter Plot')
plt.show()

"""**Problem 7:** Create two NumPy matrices, 'matrix1' (2x3) and 'matrix2' (3x4), with random values. Perform matrix multiplication between the two matrices."""

import numpy as np

matrix1 = np.random.rand(2, 3)
matrix2 = np.random.rand(3, 4)

result = np.dot(matrix1, matrix2)
print("Matrix Multiplication Result:\n", result)

"""**Problem 8:** Given a Pandas DataFrame containing sales data, calculate and display the total sales for each product category."""

import pandas as pd

data = {'Product': ['A', 'B', 'A', 'C', 'B', 'C'],
        'Sales': [100, 150, 200, 50, 120, 80]}

df = pd.DataFrame(data)

total_sales = df.groupby('Product')['Sales'].sum()
print(total_sales)

"""**Problem 9:** Create a line chart using Matplotlib to visualize the population growth of a city over 10 years. Label the axes and provide a title."""

import matplotlib.pyplot as plt

years = range(2010, 2021)
population = [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

plt.plot(years, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('City Population Growth Over 10 Years')
plt.show()

"""Problem: Given a Pandas DataFrame containing information about students, add a new column 'Grade' based on the following criteria:
* If 'Score' >= 90, Grade is 'A'
* If 'Score' >= 80, Grade is 'B'
* If 'Score' >= 70, Grade is 'C'
* If 'Score' < 70, Grade is 'D'
"""

import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Score': [95, 85, 72, 60, 78]}

df = pd.DataFrame(data)

df['Grade'] = pd.cut(df['Score'], [0, 69, 79, 89, 100], labels=['D', 'C', 'B', 'A'])
print(df)

"""**Problem 11:** You are given a dataset containing monthly sales data for three products (A, B, and C) over a two-year period. Your task is to perform various data analysis tasks using NumPy, Pandas, and Matplotlib.


*   Dataset:

```
import pandas as pd

data = {
    'Month': pd.date_range(start='2021-01-01', periods=24, freq='M'),
    'Product A Sales': [500, 480, 600, 750, 900, 850, 920, 1100, 1300, 1350, 1500, 1450, 1550, 1600, 1650, 1600, 1500, 1400, 1600, 1700, 1800, 1750, 1850, 1900],
    'Product B Sales': [300, 320, 400, 450, 500, 580, 700, 750, 820, 900, 950, 980, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1500, 1600, 1550, 1700],
    'Product C Sales': [200, 210, 250, 280, 320, 350, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720]
}

sales_df = pd.DataFrame(data)
```
*  Tasks:
1. Calculate and display the total sales for each product over the two-year period.

2. Calculate and display the average monthly sales for each product.

3. Find the month with the highest sales for each product, and display the product and the sales value for that month.

4. Calculate and display the percentage change in sales for each product from January to December in the second year (2022).

5. Create a line chart using Matplotlib to visualize the monthly sales data for each product over the two-year period. Label the axes and provide a title for the chart.

6. Calculate the correlation between Product A and Product B sales. Is there a strong correlation between these two products? Display the correlation coefficient.

"""

import pandas as pd

data = {
    'Month': pd.date_range(start='2021-01-01', periods=24, freq='M'),
    'Product A Sales': [500, 480, 600, 750, 900, 850, 920, 1100, 1300, 1350, 1500, 1450, 1550, 1600, 1650, 1600, 1500, 1400, 1600, 1700, 1800, 1750, 1850, 1900],
    'Product B Sales': [300, 320, 400, 450, 500, 580, 700, 750, 820, 900, 950, 980, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1500, 1600, 1550, 1700],
    'Product C Sales': [200, 210, 250, 280, 320, 350, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720]
}

sales_df = pd.DataFrame(data)

# Task 1
total_sales = sales_df[['Product A Sales', 'Product B Sales', 'Product C Sales']].sum()
print("Total Sales for Each Product:")
print(total_sales)

# Task 2
average_monthly_sales = sales_df[['Product A Sales', 'Product B Sales', 'Product C Sales']].mean()
print("\nAverage Monthly Sales for Each Product:")
print(average_monthly_sales)

# Task 3
max_sales_month_A = sales_df[sales_df['Product A Sales'] == sales_df['Product A Sales'].max()]['Month'].values[0]
max_sales_month_B = sales_df[sales_df['Product B Sales'] == sales_df['Product B Sales'].max()]['Month'].values[0]
max_sales_month_C = sales_df[sales_df['Product C Sales'] == sales_df['Product C Sales'].max()]['Month'].values[0]
print(f"\nMonth with Highest Sales for Product A: {max_sales_month_A}")
print(f"Month with Highest Sales for Product B: {max_sales_month_B}")
print(f"Month with Highest Sales for Product C: {max_sales_month_C}")

# Task 4
sales_January = sales_df.iloc[0][1:]
sales_December = sales_df.iloc[11][1:]
percentage_change = ((sales_December - sales_January) / sales_January) * 100
print("\nPercentage Change in Sales from January to December (2022):")
print(percentage_change)

# Task 5
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(sales_df['Month'], sales_df['Product A Sales'], label='Product A')
plt.plot(sales_df['Month'], sales_df['Product B Sales'], label='Product B')
plt.plot(sales_df['Month'], sales_df['Product C Sales'], label='Product C')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Sales Data for Products A, B, and C')
plt.legend()
plt.grid(True)
plt.show()

# Task 6
correlation_AB = sales_df['Product A Sales'].corr(sales_df['Product B Sales'])
print("\nCorrelation between Product A and Product B Sales:")
print(f"Correlation Coefficient: {correlation_AB}")