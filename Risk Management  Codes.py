#!/usr/bin/env python
# coding: utf-8

# In[37]:


# Importing required packages for this chapter
from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pylab as plt
import statsmodels.api as sm
from scipy.linalg import eigh
from dmba import classificationSummary
from dmba import AIC_score
from dmba import backward_elimination, forward_selection, stepwise_selection

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dmba import regressionSummary

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#loading the file
Aviation_df = pd.read_csv('Capstone_dataset_2010_2019_FINAL_SET.csv')
Aviation_df.head(15)


# In[4]:


Aviation_df.tail(15)


# In[5]:


Aviation_df.columns


# In[6]:


#descriptive statistics summary
Aviation_df.describe()


# In[7]:


Aviation_df.shape


# In[8]:


# get the number of missing data points per column
missing_values_count = Aviation_df.isnull().sum()
# look at the # of missing points in the first ten columns
missing_values_count[0:28]


# In[9]:


# how many total missing values do we have?
total_cells = np.product(Aviation_df.shape)
total_missing = missing_values_count.sum()
# percent of data that is missing
(total_missing/total_cells) * 100


# In[10]:


Aviation_df.dropna()


# In[12]:


# remove all columns with at least one missing value
columns_with_na_dropped = Aviation_df.dropna(axis=1)
columns_with_na_dropped.tail(15)


# In[13]:


columns_with_na_dropped.head(15)


# In[14]:


# just how much data did we lose?
print("Columns in original dataset: %d \n" % Aviation_df.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])


# In[15]:


# subplots
Aviation_df.plot(subplots = True)
plt.show()


# In[17]:


# get a small subset of the aviation dataset
subset_Aviation_df = Aviation_df.loc[:, 'DATE':'WN_LateAircraft_Delay'].head(15)
subset_Aviation_df


# In[18]:


# get a small subset of the aviation dataset
subset_Aviation_df = Aviation_df.loc[:, 'DATE':'WN_LateAircraft_Delay'].tail(15)
subset_Aviation_df


# In[19]:


# replace all NA's with 0
subset_Aviation_df.fillna(0)


# In[20]:


# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset = subset_Aviation_df.fillna(method = 'bfill', axis=0).fillna(0)


# In[21]:


subset.shape


# In[22]:


# subplots
subset.plot(subplots = True)
plt.show()


# In[23]:


#scatter plot between 'DAL_close' and 'Price_Gal'
plt.scatter(subset['DAL_close'], subset['DATE'])
plt.xlabel('DAL_close')
plt.ylabel('DATE')
plt.title('Scatter Plot between DATE and UA_Bird_Strike')
plt.show()


# In[55]:


#scatter plot between 'LUV_close' and 'Price_Gal'
plt.scatter(subset['LUV_close'], subset['DATE'])
plt.xlabel('LUV_close')
plt.ylabel('DATE')
plt.title('Scatter Plot between DATE and LUV_close')
plt.show()


# In[56]:


#scatter plot between 'UAL_close' and 'Price_Gal'
plt.scatter(subset['UAL_close'], subset['DATE'])
plt.xlabel('UAL_close')
plt.ylabel('DATE')
plt.title('Scatter Plot between DATE and UAL_close')
plt.show()


# In[32]:


# Plotting all data 
subset1 = subset.loc[:,[ "DATE", "DL_Bird_Strike", "DL_Weather_Delay", "DL_Taxi_Out", "DL_Taxi_In", "DL_Arrival_Delay", "DL_Carrier_Delay", "DL_LateAircraft_Delay"]]
subset.plot()
# it is confusing


# In[33]:


# subplots
subset1.plot(subplots = True)
plt.show()


# REGRESSION ANALYSIS

# In[38]:


# Define the independent variables
dependent_variables = ['DAL_close']

# Define the list of dependent variables
independent_variables = ['DL_Bird_Strike', 
                       'DL_Weather_Delay', 
                       'DL_Departure_Delay', 
                       'DL_Taxi_Out', 
                       'DL_Taxi_In', 
                       'DL_Arrival_Delay', 
                       'DL_Carrier_Delay', 
                       'DL_LateAircraft_Delay']

# Iterate through each dependent variable and create a regression model
for dependent_variable in dependent_variables:
    y = subset[dependent_variable]
    X = subset[independent_variables]
    X = sm.add_constant(X)  # Add a constant term (intercept)

    model = sm.OLS(y, X).fit()
    print(f"Dependent Variable: {dependent_variable}")
    print(model.summary())
    print("\n")


# DElTA AIRLINE 
# 
# In this regression analysis, the dependent variable is labeled as "DAL_close." This variable represents the closing price of a financial instrument, presumably the stock price of Delta Air Lines (DAL) or a related financial product. The regression results provide information on how this dependent variable (DAL_close) is influenced by various independent variables.
# 
# Here's a breakdown of the key information from the regression analysis:
# 
# R-squared (R²): The R-squared value is a measure of how well the independent variables explain the variation in the dependent variable. In this case, the R-squared value is 0.961, indicating that approximately 96.1% of the variance in DAL_close can be explained by the independent variables included in the model. This suggests a strong relationship between the dependent and independent variables.
# 
# Adjusted R-squared: The adjusted R-squared adjusts the R-squared value for the number of independent variables in the model. It helps account for the possibility of overfitting. Here, the adjusted R-squared is 0.910, which is still relatively high and indicates a good fit.
# 
# F-statistic: The F-statistic is used to test the overall significance of the regression model. A higher F-statistic suggests a better overall fit of the model. In this case, the F-statistic is 18.60, and the associated probability (Prob (F-statistic)) is 0.00107, indicating that the overall model is statistically significant.
# 
# Coefficients: The table provides coefficients for each independent variable (including the intercept constant). These coefficients represent the estimated impact of each independent variable on the dependent variable (DAL_close). The "std err" values are standard errors associated with each coefficient.
# 
# P-values (P>|t|): The p-values associated with each coefficient indicate whether the independent variable is statistically significant in explaining the variation in the dependent variable. In this table, some variables have p-values less than 0.05 (e.g., DL_Weather_Delay, DL_Taxi_Out, DL_Arrival_Delay, DL_LateAircraft_Delay), which suggests they are statistically significant predictors of DAL_close.
# 
# Other Statistics: The table also includes additional statistics like the Omnibus test, Durbin-Watson statistic, Jarque-Bera test, skewness, kurtosis, and condition number, which can provide insights into the quality and assumptions of the regression model.
# 
# Overall, this regression analysis suggests that the closing price of Delta Air Lines (DAL_close) is strongly influenced by the included independent variables. However, it's important to consider the context and purpose of the analysis and to interpret the coefficients and statistical significance of each variable carefully when making financial decisions or drawing conclusions about the stock's performance.

# In[39]:


#UNITED AIRLINE

# Define the independent variables
dependent_variables = ['UAL_close']

# Define the list of dependent variables
independent_variables = [ 'UA_Bird_Strike',
                          'UA_Weather_Delay', 
                          'UA_Departure_Delay',
                          'UA_Taxi_Out', 
                          'UA_Taxi_In', 
                          'UA_Arrival_Delay', 
                          'UA_Carrier_Delay',
                          'UA_LateAircraft_Delay']

# Iterate through each dependent variable and create a regression model
for dependent_variable in dependent_variables:
    y = subset[dependent_variable]
    X = subset[independent_variables]
    X = sm.add_constant(X)  # Add a constant term (intercept)

    model = sm.OLS(y, X).fit()
    print(f"Dependent Variable: {dependent_variable}")
    print(model.summary())
    print("\n")


# THE RESULT 
# 
# In this regression analysis, the dependent variable is labeled as "UAL_close," representing the closing price of a stock price of United Airlines (UAL) Let's interpret the key findings from the regression results:
# 
# R-squared (R²): The R-squared value is 0.617, indicating that approximately 61.7% of the variance in UAL_close can be explained by the independent variables included in the model. This suggests that the model accounts for a moderate portion of the variability in the stock price.
# 
# Adjusted R-squared: The adjusted R-squared is 0.106, which is significantly lower than the R-squared value. This suggests that after adjusting for the number of independent variables in the model, the explanatory power decreases substantially. It might indicate overfitting or the inclusion of non-significant variables.
# 
# F-statistic: The F-statistic is 1.208, and the associated probability (Prob (F-statistic)) is 0.421. The F-statistic tests the overall significance of the regression model. In this case, the F-statistic is relatively low, and the probability is high, indicating that the overall model is not statistically significant.
# 
# Coefficients: The table provides coefficients for each independent variable (including the intercept constant). These coefficients represent the estimated impact of each independent variable on the dependent variable (UAL_close). The "std err" values are standard errors associated with each coefficient.
# 
# P-values (P>|t|): The p-values associated with each coefficient indicate whether the independent variable is statistically significant in explaining the variation in the dependent variable. In this table, most variables have p-values greater than 0.05, which suggests that they are not statistically significant predictors of UAL_close. The exceptions are UA_Departure_Delay and UA_Arrival_Delay, which have p-values below 0.05, indicating statistical significance.
# 
# Other Statistics: The table includes additional statistics like the Omnibus test, Durbin-Watson statistic, Jarque-Bera test, skewness, kurtosis, and condition number. These statistics can provide insights into the quality and assumptions of the regression model.
# 
# Overall, the regression analysis for UAL_close suggests that the model does not have a strong overall fit, as indicated by the low F-statistic and high p-values for most independent variables. It's important to carefully interpret the results and consider whether the model is adequately capturing the factors that influence the stock price. Additionally, further analysis and potentially different independent variables may be needed to improve the model's predictive power for UAL_close.

# In[41]:


#SOUTHWEST AIRLINE

# Define the independent variables
dependent_variables = ['LUV_close']

# Define the list of dependent variables
independent_variables = ['WN_Bird_Strike',
                         'WN_Weather_Delay',
                         'WN_Departure_Delay',
                         'WN_Taxi_Out',
                         'WN_Taxi_In',
                         'WN_Arrival_Delay',
                         'WN_Carrier_Delay',
                         'WN_LateAircraft_Delay']

# Iterate through each dependent variable and create a regression model
for dependent_variable in dependent_variables:
    y = subset[dependent_variable]
    X = subset[independent_variables]
    X = sm.add_constant(X)  # Add a constant term (intercept)

    model = sm.OLS(y, X).fit()
    print(f"Dependent Variable: {dependent_variable}")
    print(model.summary())
    print("\n")


# THE RESULT 
# 
# In this regression analysis, the dependent variable is labeled as "LUV_close," representing the closing price of the stock price of Southwest Airlines (LUV) or a related financial product. Let's interpret the key findings from the regression results:
# 
# R-squared (R²): The R-squared value is 0.848, indicating that approximately 84.8% of the variance in LUV_close can be explained by the independent variables included in the model. This suggests that the model accounts for a significant portion of the variability in the stock price.
# 
# Adjusted R-squared: The adjusted R-squared is 0.645, which is lower than the R-squared value. This indicates that after adjusting for the number of independent variables in the model, the explanatory power decreases. It suggests that some independent variables may not be statistically significant.
# 
# F-statistic: The F-statistic is 4.173, and the associated probability (Prob (F-statistic)) is 0.0493. The F-statistic tests the overall significance of the regression model. In this case, the F-statistic is relatively high, and the probability is below the 0.05 significance level, indicating that the overall model is statistically significant.
# 
# Coefficients: The table provides coefficients for each independent variable (including the intercept constant). These coefficients represent the estimated impact of each independent variable on the dependent variable (LUV_close). The "std err" values are standard errors associated with each coefficient.
# 
# P-values (P>|t|): The p-values associated with each coefficient indicate whether the independent variable is statistically significant in explaining the variation in the dependent variable. In this table, some variables have p-values less than 0.05 (e.g., WN_Taxi_In, WN_Arrival_Delay, WN_Carrier_Delay, WN_LateAircraft_Delay), suggesting that they are statistically significant predictors of LUV_close.
# 
# Other Statistics: The table includes additional statistics like the Omnibus test, Durbin-Watson statistic, Jarque-Bera test, skewness, kurtosis, and condition number. These statistics can provide insights into the quality and assumptions of the regression model.
# 
# Overall, the regression analysis for LUV_close suggests that the model has a reasonably good overall fit, as indicated by the high R-squared value, significant F-statistic, and some statistically significant independent variables. However, it's important to note that not all independent variables are statistically significant, as indicated by the p-values. Careful consideration should be given to the variables included in the model, and further analysis may be needed to refine and improve the model's predictive power for LUV_close.

# # Stepwise Regression 

# In[61]:


def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature} with p-value {best_pval}')
        # Backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # Use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval}')
        if not changed:
            break
    return included
# Specifying your dependent variable and independent variables
X = subset[['DAL_close']] 
y = subset['Price_Gal']

selected_features = stepwise_selection(X, y)
print("Selected Features:", selected_features)

# Build the final model with the selected features
final_model = sm.OLS(y, sm.add_constant(subset[selected_features])).fit()
print(final_model.summary())

# Calculate AIC for the final model
aic = final_model.aic
print("AIC for the Final Model:", aic)


# In[62]:


def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature} with p-value {best_pval}')
        # Backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # Use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval}')
        if not changed:
            break
    return included
# Specifying your dependent variable and independent variables
X = subset[['UAL_close']] 
y = subset['Price_Gal']

selected_features = stepwise_selection(X, y)
print("Selected Features:", selected_features)

# Build the final model with the selected features
final_model = sm.OLS(y, sm.add_constant(subset[selected_features])).fit()
print(final_model.summary())

# Calculate AIC for the final model
aic = final_model.aic
print("AIC for the Final Model:", aic)


# PRINCIPAL COMPONENT ANALYSIS 

# In[ ]:





# In[46]:


#DELTA AIRLINE
# Define the independent variables
dependent_variables = ['DAL_close']

# Define the dependent variables
independent_variables = ['DL_Bird_Strike', 
                       'DL_Weather_Delay', 
                       'DL_Departure_Delay', 
                       'DL_Taxi_Out', 
                       'DL_Taxi_In', 
                       'DL_Arrival_Delay', 
                       'DL_Carrier_Delay', 
                       'DL_LateAircraft_Delay']

# Extract the data for PCA
X = subset[independent_variables]
Y = subset[dependent_variables]

# Standardize the data (mean=0, variance=1) for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(X_scaled)

# Access the explained variance ratios to see how much variance each component explains
explained_variance_ratios = pca.explained_variance_ratio_

# Create a DataFrame to store the results
pca_results = pd.DataFrame({
    'Principal Component': [f'PC{i + 1}' for i in range(len(explained_variance_ratios))],
    'Explained Variance Ratio': explained_variance_ratios,
    'Cumulative Variance Ratio': np.cumsum(explained_variance_ratios)
})

# Print the PCA results
print("PCA Results:")
print(pca_results)


# THE RESULT 
# 
# By PC1  we 've explained 65.23% of the total variance.
# By PC2, you've explained 79.60% of the total variance (PC1 + PC2).
# By PC3, you've explained 91.12% of the total variance (PC1 + PC2 + PC3).
# By PC4, you've explained 98.29% of the total variance (PC1 + PC2 + PC3 + PC4).
# By PC5, you've explained 99.23% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5).
# By PC6, you've explained 99.83% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6).
# By PC7, you've explained 99.98% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7).
# By PC8, you've explained 100% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8).
# Interpretation:
# 
# These results suggest that the first few principal components (PC1 to PC4) capture the vast majority of the variance in your dataset. PC1 alone accounts for about 65% of the total variance, and by PC4, you've captured over 98% of the total variance.
# PC5, PC6, PC7, and PC8 each explain a relatively small amount of additional variance, and they are not contributing significantly to the overall explanation of variance in the data.
# In practical terms, this means that you can potentially reduce the dimensionality of your dataset by retaining only the first four principal components (PC1 to PC4) while still retaining most of the relevant information in your data. This can simplify your analysis and potentially improve model training or visualization while preserving the essential patterns in your data.

# In[47]:


#UNITED AIRLINE
# Define the independent variables
dependent_variables = ['UAL_close']

# Define the list of dependent variables
independent_variables = [ 'UA_Bird_Strike',
                          'UA_Weather_Delay', 
                          'UA_Departure_Delay',
                          'UA_Taxi_Out', 
                          'UA_Taxi_In', 
                          'UA_Arrival_Delay', 
                          'UA_Carrier_Delay',
                          'UA_LateAircraft_Delay']

# Extract the data for PCA
X = subset[independent_variables]
Y = subset[dependent_variables]

# Standardize the data (mean=0, variance=1) for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(X_scaled)

# Access the explained variance ratios to see how much variance each component explains
explained_variance_ratios = pca.explained_variance_ratio_

# Create a DataFrame to store the results
pca_results = pd.DataFrame({
    'Principal Component': [f'PC{i + 1}' for i in range(len(explained_variance_ratios))],
    'Explained Variance Ratio': explained_variance_ratios,
    'Cumulative Variance Ratio': np.cumsum(explained_variance_ratios)
})

# Print the PCA results
print("PCA Results:")
print(pca_results)


# THE RESULT 
# 
# By PC1 alone, you've explained 65.92% of the total variance.
# By PC2, you've explained 79.80% of the total variance (PC1 + PC2).
# By PC3, you've explained 90.75% of the total variance (PC1 + PC2 + PC3).
# By PC4, you've explained 95.53% of the total variance (PC1 + PC2 + PC3 + PC4).
# By PC5, you've explained 98.59% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5).
# By PC6, you've explained 99.66% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6).
# By PC7, you've explained 99.87% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7).
# By PC8, you've explained 100% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8).
# 
# Interpretation:
# These results suggest that the first few principal components (PC1 to PC3) capture the majority of the variance in your dataset. PC1 alone accounts for about 65.92% of the total variance, and by PC3, you've captured over 90% of the total variance.
# PC4, PC5, PC6, PC7, and PC8 each explain a relatively small amount of additional variance, and they are not contributing significantly to the overall explanation of variance in the data.
# In practical terms, this means that you can potentially reduce the dimensionality of your dataset by retaining only the first three principal components (PC1 to PC3) while still retaining most of the relevant information in your data. This can simplify your analysis and potentially improve model training or visualization while preserving the essential patterns in your data.

# In[49]:


#SOUTHWWEST AIRLINE
# Define the independent variables
dependent_variables = ['LUV_close']

# Define the list of dependent variables
independent_variables = ['WN_Bird_Strike',
                         'WN_Weather_Delay',
                         'WN_Departure_Delay',
                         'WN_Taxi_Out',
                         'WN_Taxi_In',
                         'WN_Arrival_Delay',
                         'WN_Carrier_Delay',
                         'WN_LateAircraft_Delay']

# Extract the data for PCA
X = subset[independent_variables]
Y = subset[dependent_variables]

# Standardize the data (mean=0, variance=1) for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(X_scaled)

# Access the explained variance ratios to see how much variance each component explains
explained_variance_ratios = pca.explained_variance_ratio_

# Create a DataFrame to store the results
pca_results = pd.DataFrame({
    'Principal Component': [f'PC{i + 1}' for i in range(len(explained_variance_ratios))],
    'Explained Variance Ratio': explained_variance_ratios,
    'Cumulative Variance Ratio': np.cumsum(explained_variance_ratios)
})

# Print the PCA results
print("PCA Results:")
print(pca_results)


# THE RESULT
# 
# By PC1 alone, you've explained 69.30% of the total variance.
# By PC2, you've explained 83.59% of the total variance (PC1 + PC2).
# By PC3, you've explained 91.75% of the total variance (PC1 + PC2 + PC3).
# By PC4, you've explained 98.39% of the total variance (PC1 + PC2 + PC3 + PC4).
# By PC5, you've explained 99.54% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5).
# By PC6, you've explained 99.87% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6).
# By PC7, you've explained 99.97% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7).
# By PC8, you've explained 100% of the total variance (PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8).
# Interpretation:
# 
# These results indicate that the first few principal components (PC1 to PC4) capture the vast majority of the variance in your dataset. PC1 alone accounts for about 69.30% of the total variance, and by PC4, you've captured over 98% of the total variance.
# PC5, PC6, PC7, and PC8 each explain a relatively small amount of additional variance and are not contributing significantly to the overall explanation of variance in the data.
# In practical terms, this means that you can potentially reduce the dimensionality of your dataset by retaining only the first four principal components (PC1 to PC4) while still retaining most of the relevant information in your data. This can simplify your analysis and potentially improve model training or visualization while preserving the essential patterns in your data.

# In[ ]:





# In[50]:


#Create the line graph
plt.figure(figsize=(10, 6))

# Plot the data for the independent variables
plt.plot(subset['DATE'], subset['DAL_close'], label='DAL_close', marker='o', linestyle='-')
plt.plot(subset['DATE'], subset['LUV_close'], label='LUV_close', marker='s', linestyle='--')
plt.plot(subset['DATE'], subset['UAL_close'], label='UAL_close', marker='^', linestyle='-.')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices vs. Date')

# Add a legend
plt.legend()

# Create a second y-axis for 'Price_Gal'
ax2 = plt.twinx()  # Create a secondary y-axis
ax2.plot(subset['DATE'], subset['Price_Gal'], label='Price_Gal', color='purple', linestyle='-.')
ax2.set_ylabel('Gasoline Price', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Show the graph
plt.grid(True)
plt.show()





# In[51]:


# Create the line graph
plt.figure(figsize=(10, 6))

# Plot the data for the independent variables
plt.plot(subset['DATE'], subset['DAL_close'], label='DAL_close', marker='o', linestyle='-')
plt.plot(subset['DATE'], subset['LUV_close'], label='LUV_close', marker='s', linestyle='--')
plt.plot(subset['DATE'], subset['UAL_close'], label='UAL_close', marker='^', linestyle='-.')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices vs. Date')

# Add a legend
plt.legend()

# Show the graph
plt.grid(True)
plt.show()


# In[52]:


# Create the line graph
plt.figure(figsize=(10, 6))

# Plot the data for the dependent variable 'Price_Gal'
plt.plot(subset['Price_Gal'], subset['DAL_close'], label='DAL_close', marker='o', linestyle='-')
plt.plot(subset['Price_Gal'], subset['LUV_close'], label='LUV_close', marker='s', linestyle='--')
plt.plot(subset['Price_Gal'], subset['UAL_close'], label='UAL_close', marker='^', linestyle='-.')

# Set labels and title
plt.xlabel('Gasoline Price (Price_Gal)')
plt.ylabel('Stock Prices')
plt.title('Stock Prices vs. Gasoline Price')

# Add a legend
plt.legend()

# Show the graph
plt.grid(True)
plt.show()


# In[53]:


# Create the line graph
plt.figure(figsize=(10, 6))

# Plot the data for the dependent variable 'Price_Gal'
plt.plot(subset['Price_Gal'], subset['DAL_close'], label='DAL_close', marker='o', linestyle='-')

# Set labels and title
plt.xlabel('Gasoline Price (Price_Gal)')
plt.ylabel('Stock Prices')
plt.title('Stock Prices vs. Gasoline Price')

# Add a legend
plt.legend()

# Show the graph
plt.grid(True)
plt.show()


# In[ ]:




