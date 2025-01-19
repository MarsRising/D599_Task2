#!/usr/bin/env python
# coding: utf-8

# In[109]:


#Our Libraries
import pandas as pd # used for handling the dataset
import numpy as np #used for handling numbers
import scipy.stats as stats #parametric and nonparametric statistical tests
import statsmodels.api as sm #statistical models
import matplotlib.pyplot as plt #checking for outliers
import seaborn as sns #checking for outliers
from sklearn.impute import SimpleImputer #handles missing data
from scipy.stats import chi2_contingency, spearmanr
from sklearn.linear_model import LinearRegression

print("Complete")


# In[110]:


#create dataframe
df = pd.read_excel(r'C:\Users\tyler\OneDrive - SNHU\WGU\Data Preparation and Exploration\Health Insurance Dataset.xlsx')
print("df complete")
df.info()


# In[111]:


#extra details at bottom of the dataset that are not important
df = df.drop(columns=['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17'])
df.info()


# In[112]:


#drop empty rows
df=df.dropna()
df.info()


# In[113]:


#For Univariate and Bivariate, I will use categorical: Sex and Smoker Continuous: Age and Charges
#First the Univariate of Age/Descriptive statistics
age_stats = df['age'].describe()
print("Age Descriptive Statistics:")
print(age_stats)
#mode
mode_age=df['age'].mode()
print('Mode', mode_age)
age_skewness = df['age'].skew()
print(f"Skewness: {age_skewness}")
#visual for age
plt.subplot(1, 2, 1)
sns.histplot(df['age'], kde=True, bins=30, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
#boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=df['age'], color='green')
plt.title('Age Box Plot')
plt.xlabel('Age')


# In[114]:


#Univariate/ Descriptive statistics of charges
charges_stats = df['charges'].describe()
print("\nCharges Descriptive Statistics:")
print(charges_stats)
#mode
mode_charges=df['charges'].mode()
print('Mode', mode_charges)
charges_skewness = df['charges'].skew()
print(f"Skewness: {charges_skewness}")
#histogram
plt.subplot(1, 2, 1)
sns.histplot(df['charges'], kde=True, bins=30, color='orange')
plt.title('Charges Distribution')
plt.xlabel('Charges')
plt.ylabel('Frequency')
#Bocplot
plt.subplot(1, 2, 2)
sns.boxplot(x=df['charges'], color='red')
plt.title('Charges Box Plot')
plt.xlabel('Charges')


# In[115]:


#Bivariate Analysis
#Correlation age and charges
correlation, _ = stats.pearsonr(df['age'], df['charges'])
print("\nCorrelation between Age and Charges: {:.2f}".format(correlation))
#scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='charges', data=df, color='purple')
plt.title('Scatter Plot: Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()


# In[116]:


#spearman's correlation due to outliers
spearman_corr, _ = spearmanr(df['age'], df['charges'])
print("\nSpearman's rank correlation between Age and Charges: {:.2f}".format(spearman_corr))


# In[117]:


#Now for our Categorical variables Sex and Smoker
#sex descriptive statistics
#counts
sex_counts = df['sex'].value_counts()
print("\nSex Distribution:")
print(sex_counts)
#proportions
sex_proportions = df['sex'].value_counts(normalize=True)
print(sex_proportions)
#descriptive
sex_summary = df['sex'].describe()
print(sex_summary)
#visualization
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', data=df, hue='sex', palette='Set2', legend=False)
plt.title('Distribution of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


# In[118]:


#smoker descriptive statistics
smoker_counts = df['smoker'].value_counts()
print("\nSmoker Distribution:")
print(smoker_counts)
#proportions
smoker_proportions = df['smoker'].value_counts(normalize=True)
print(smoker_proportions)
#descriptive
smoker_summary = df['smoker'].describe()
print(smoker_summary)
#visualization
plt.figure(figsize=(6, 4))
sns.countplot(x='smoker', data=df, hue='smoker', palette='Set2', legend=False)
plt.title('Distribution of Smoker')
plt.xlabel('Smoker')
plt.ylabel('Count')
plt.show()


# In[119]:


#Bivariate of sex and smoker
#cross-tabulation
contingency_table = pd.crosstab(df['sex'], df['smoker'])
print("\nContingency Table between Sex and Smoker:")
print(contingency_table)
#chi-square test for independence
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print("\nChi-square Test for Independence between 'sex' and 'smoker':")
print(f"Chi2 Statistic: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies Table:\n{expected}")
#heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Heatmap: Sex vs Smoker')
plt.ylabel('Sex')
plt.xlabel('Smoker')
plt.show()


# In[120]:


#Linear regression model for age and charges
X = df['age']
y = df['charges']
#intercept constant
X = sm.add_constant(X)
#fit model
model = sm.OLS(y, X).fit()
print(model.summary())


# In[105]:


#want to see what the p_value actually is
p_value = model.pvalues['age']
print(p_value)


# In[ ]:




