import matplotlib.pyplot as plt
import numpy as np

# Line Plot
x = [1, 2, 3, 4, 5]
y = [5, 7, 4, 6, 8]

plt.figure(figsize=(6, 4))
plt.plot(x, y, label='Line', color='blue', marker='o')
plt.title('Line Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Chart
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]

plt.figure(figsize=(6, 4))
plt.bar(categories, values, color='green')
plt.title('Bar Chart')
plt.xlabel('Category')
plt.ylabel('Values')
plt.tight_layout()
plt.show()

# Histogram
data = [22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27]

plt.figure(figsize=(6, 4))
plt.hist(data, bins=5, color='purple', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter Plot
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 100 * np.random.rand(50)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.title('Scatter Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.colorbar(label='Color Scale')
plt.tight_layout()
plt.show()

# Pie Chart
labels = ['Python', 'C++', 'Java', 'Ruby']
sizes = [45, 30, 15, 10]
explode = (0.1, 0, 0, 0)  # explode first slice

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.title('Pie Chart')
plt.tight_layout()
plt.show()

# Box Plot
data1 = np.random.normal(100, 10, 200)
data2 = np.random.normal(90, 20, 200)

plt.figure(figsize=(6, 4))
plt.boxplot([data1, data2], labels=['Group 1', 'Group 2'])
plt.title('Box Plot')
plt.ylabel('Values')
plt.tight_layout()
plt.show()
