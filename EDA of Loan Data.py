#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace filename with your actual path)
df = pd.read_csv("LoanApprovalPrediction (1).csv")

# Display first few rows
df.head()


# # Data Summary
# 

# In[31]:


print("Shape of the data: ",df.shape  )


# In[37]:


print("column names:\n ",df.columns  )


# In[21]:


df.info() 


# In[45]:


summary = pd.DataFrame({
    '  Column': df.columns,
    '  Data Type': df.dtypes.values,
    'Non-Null Count': df.notnull().sum().values,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
})
print(summary)


# In[51]:


from IPython.display import display

summary = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes.values,
    'Non-Null Count': df.notnull().sum().values,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
})

# Sort by missing % (optional, makes it easier to focus on columns that need cleaning)
summary = summary.sort_values(by='Missing %', ascending=False).reset_index(drop=True)

# Display as a styled table
display(summary.style
        .background_gradient(subset=['Missing %'], cmap='Reds')
        .format({'Missing %': '{:.2f}%'})
        .set_caption("🔍 Column Overview with Missing Value Info")
        .hide_index()
        .set_properties(**{'text-align': 'left'})
       )


# In[22]:


df.describe()      


# In[55]:


df.describe(include='object')


# In[56]:


df['Gender'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    shadow=True,
    colors=['skyblue', 'lightcoral'],
    figsize=(5, 5),
    title='Gender Distribution'
)
plt.ylabel('')  # Hide y-label
plt.show()


# In[57]:


df['Loan_Status'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    shadow=True,
    colors=['lightgreen', 'salmon'],
    figsize=(5, 5),
    title='Loan Approval Status'
)
plt.ylabel('')
plt.show()


# In[58]:


df['Property_Area'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    shadow=True,
    colors=['gold', 'lightblue', 'lightpink'],
    figsize=(5, 5),
    title='Property Area Distribution'
)
plt.ylabel('')
plt.show()


# In[59]:


df['Education'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    shadow=True,
    colors=['orchid', 'palegreen'],
    figsize=(5, 5),
    title='Education Level'
)
plt.ylabel('')
plt.show()


# In[23]:


df.isnull().sum()


# # Univariate Analysis

# In[24]:


categorical_cols = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']

for col in categorical_cols:
    print(df[col].value_counts())
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()


# ## Numerical data

# In[61]:


numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

for col in numerical_cols:
    sns.histplot(df[col].dropna(), kde=True,bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[68]:


numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

for col in numerical_cols:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:





# # Bivariate Analysis 

# ## Categorical Bivariate

# In[26]:


for col in categorical_cols:
    sns.countplot(x=col, hue='Loan_Status', data=df)
    plt.title(f'Loan Status by {col}')
    plt.xticks(rotation=45)
    plt.show()


# ## Numerical bivariate

# In[27]:


for col in numerical_cols:
    sns.boxplot(x='Loan_Status', y=col, data=df)
    plt.title(f'{col} by Loan Status')
    plt.show()


# In[28]:


sns.catplot(x="Credit_History", hue="Loan_Status", col="Education",
            data=df, kind="count", height=4, aspect=.7)


# In[29]:


corr = df[numerical_cols + ['Credit_History']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[69]:


summary_table = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.values,
    'Missing Values': df.isnull().sum().values,
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values,
    'Unique Values': df.nunique().values,
    'Sample Value': [df[col].dropna().unique()[0] if df[col].dropna().nunique() > 0 else None for col in df.columns]
})

summary_table = summary_table.sort_values(by='Missing %', ascending=False).reset_index(drop=True)

display(
    summary_table.style
    .background_gradient(subset=['Missing %'], cmap='Reds')
    .format({'Missing %': '{:.2f}%'})
    .set_caption("📊 Dataset Summary")
    .hide_index()
    .set_properties(**{'text-align': 'left'})
)


# In[ ]:




