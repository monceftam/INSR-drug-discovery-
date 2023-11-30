#!/usr/bin/env python
# coding: utf-8

# In[341]:


# Basic array and data structure handling
import numpy as np
import pandas as pd

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Statistical tests
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import kruskal

# Machine Learning models and tools
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

# Model evaluation and selection tools
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

# Data preprocessing tools
from sklearn.preprocessing import StandardScaler

# Advanced statistical modeling
import statsmodels.api as sm

# External data source (e.g., for drug discovery)
from chembl_webresource_client.new_client import new_client


# In[342]:


#initialize new client
target = new_client.target


# In[343]:


# Step 1: Search for targets related to 'INSR' using the target search method.
target_query = target.search("INSR")

# Step 2: Convert the query results into a pandas DataFrame for easier data manipulation.
targets = pd.DataFrame.from_dict(target_query)

# Step 3: Display specific columns of the DataFrame to get an overview of the search results.
print("Targets related to 'INSR':")
print(targets[['target_chembl_id', 'organism', 'target_type', 'pref_name']])


# In[412]:


# Select the fourth target's ChEMBL ID from the DataFrame.
# The index [3] is used because Python indexing starts at 0, so this refers to the fourth item.
selected_target = targets.target_chembl_id[3]


# In[413]:


# Step 1: Access the activity service from the new_client, which is likely a ChEMBL web resource client.
activity = new_client.activity

# Step 2: Filter the activities to only include those related to the selected target.
# Further, it filters the activities by the standard type "IC50", which is a common measure in pharmacology.
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

# Step 3: Convert the filtered results into a pandas DataFrame.
# This step makes the data easier to manipulate and analyze in Python.
df = pd.DataFrame.from_dict(res)

# Step 4: Display the first three rows of the DataFrame.
# This gives a quick overview of the data structure and the first few entries of the retrieved data.
print(df.head(3))


# In[414]:


# Save the DataFrame 'df' to a CSV file named 'GLP.csv'.
# The 'index=False' parameter is used to prevent writing row indices into the CSV file.
df.to_csv('GLP.csv', index=False)


# In[415]:


# Filter the DataFrame 'df' to include only rows where 'standard_value' is not missing.
# The result is assigned to a new DataFrame 'df2'.
# This is useful for data cleaning, ensuring that analyses or models built on this data don't include missing values in 'standard_value'.
df2 = df[df.standard_value.notna()]


# In[416]:


# Step 1: Convert the 'standard_value' column in 'df2' to numeric type.
# 'errors='coerce'' converts non-numeric values to NaN, which helps in handling unexpected data formats.
df2['standard_value'] = pd.to_numeric(df2['standard_value'], errors='coerce')

# Step 2: Remove any rows in 'df2' where 'standard_value' is NaN.
# This further cleans the data by ensuring that all entries in 'standard_value' are valid numeric values.
df2.dropna(subset=['standard_value'], inplace=True)

# Step 3: Calculate the pIC50 values for each row in 'df2'.
# pIC50 is a common logarithmic scale used in drug discovery to express inhibitor concentration.
# It's calculated as the negative logarithm (base 10) of the IC50 value (in molar units).
# The multiplication by 10**-9 converts the standard value from nM to M.
df2['pIC50'] = -np.log10(df2['standard_value'] * (10**-9))


# In[417]:


# Fit a Gaussian mixture model to the pIC50 values.
# The model is set to have two components, assuming two distinct groups in the data (active and inactive).
gmm = GaussianMixture(n_components=2, random_state=0).fit(df2['pIC50'].values.reshape(-1, 1))

# Retrieve the means of the two fitted Gaussian distributions.
# These represent the central tendency of each group in the data.
means = gmm.means_.flatten()

# Sort the means to identify the 'active' (higher mean) and 'inactive' (lower mean) groups.
sorted_indices = np.argsort(means)
active_mean = means[sorted_indices[1]]   # Higher mean
inactive_mean = means[sorted_indices[0]] # Lower mean

# Define cutoffs for 'active' and 'inactive' based on the means.
# The 'active' cutoff is set to the mean of the 'active' distribution, and similarly for 'inactive'.
active_cutoff = active_mean
inactive_cutoff = inactive_mean

# Function to categorize compounds as 'active', 'inactive', or 'intermediate' based on pIC50 values.
def categorize_compound(pIC50, active_cutoff, inactive_cutoff):
    if pIC50 >= active_cutoff:
        return 'active'
    elif pIC50 < inactive_cutoff:
        return 'inactive'
    else:
        return 'intermediate'

# Apply the categorization function to each pIC50 value in the DataFrame.
df2['bioactivity_class'] = df2['pIC50'].apply(categorize_compound, args=(active_cutoff, inactive_cutoff))

# Visualize the distribution of pIC50 values.
# Histogram is plotted with vertical lines indicating the active and inactive cutoffs.
sns.histplot(df2['pIC50'], bins=30, kde=False)
plt.axvline(x=active_cutoff, color='green', linestyle='--', label=f'Active Cutoff: {active_cutoff:.2f}')
plt.axvline(x=inactive_cutoff, color='red', linestyle='--', label=f'Inactive Cutoff: {inactive_cutoff:.2f}')
plt.xlabel('pIC50')
plt.ylabel('Frequency')
plt.title('Distribution of pIC50 values with Cutoffs')
plt.legend()
plt.show()


# In[418]:


# Print the head of the updated DataFrame
print(df2[['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'bioactivity_class']].head())


# In[419]:


# Step 1: Filter the DataFrame 'df2' to obtain only those compounds classified as 'active'.
# This creates a new DataFrame 'active_compounds' containing only the rows where 'bioactivity_class' is 'active'.
active_compounds = df2[df2['bioactivity_class'] == 'active']

# Step 2: Similarly, filter 'df2' to get only the 'inactive' compounds.
# 'inactive_compounds' is a new DataFrame with rows where 'bioactivity_class' is 'inactive'.
inactive_compounds = df2[df2['bioactivity_class'] == 'inactive']


# In[420]:


# Perform the Shapiro-Wilk test for normality on the 'pIC50' column of the DataFrame 'df2'.
# The Shapiro-Wilk test assesses the null hypothesis that the data is drawn from a normal distribution.
# 'stat' will store the test statistic, and 'p_value' will store the p-value of the test.
stat, p_value = shapiro(df2['pIC50'])

# Print the results of the Shapiro-Wilk test.
# A p-value less than 0.05 typically suggests that the data is not normally distributed.
print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p_value))


# In[421]:


# Perform the Kruskal-Wallis H test, a non-parametric method for testing whether samples originate from the same distribution.
# It's used here for comparing three groups: 'active', 'intermediate', and 'inactive' based on their 'pIC50' values.
stat, p_value = kruskal(df2[df2['bioactivity_class'] == 'active']['pIC50'],
                        df2[df2['bioactivity_class'] == 'intermediate']['pIC50'],
                        df2[df2['bioactivity_class'] == 'inactive']['pIC50'])

# Print the results of the Kruskal-Wallis H test.
# A p-value less than 0.05 typically indicates a statistically significant difference between the groups.
print('Kruskal-Wallis Test: Statistics=%.3f, p=%.3f' % (stat, p_value))


# In[422]:


# Create a new binary column 'active_binary' in the DataFrame 'df2'.
# This column assigns a value based on whether each compound's 'pIC50' value meets or exceeds the previously defined 'active_cutoff'.
# The expression (df2['pIC50'] >= active_cutoff) creates a boolean series (True/False) depending on the condition.
# The '.astype(int)' method converts this boolean series to integers (1 for True, 0 for False).
# This results in 'active_binary' being 1 for compounds with 'pIC50' values equal to or higher than 'active_cutoff', and 0 otherwise.
df2['active_binary'] = (df2['pIC50'] >= active_cutoff).astype(int)


# In[423]:


# Perform the Mann-Whitney U Test on the 'pIC50' values between active and inactive compounds.
# This non-parametric test is used to compare differences between two independent groups.
# It assesses whether the distribution of 'pIC50' values in one group is shifted with respect to the other group.
u_statistic, p_value = mannwhitneyu(active_compounds['pIC50'], inactive_compounds['pIC50'])

# Print the results of the Mann-Whitney U Test.
# The 'u_statistic' gives the value of the U statistic, and 'p_value' is the significance level.
# A low p-value (usually < 0.05) suggests a significant difference in the distribution of 'pIC50' values between the two groups.
print("Mann-Whitney U Test for pIC50: U-Statistic =", u_statistic, ", P-value =", p_value)


# In[424]:


# Plot the frequency distribution of 'pIC50' values for active compounds.
# seaborn's 'histplot' function is used to create a histogram.
# 'bins=30' specifies that the data should be divided into 30 bins for the histogram.
# 'kde=False' indicates that a Kernel Density Estimate plot should not be overlaid.
sns.histplot(active_compounds['pIC50'], bins=30, kde=False)

# Label the x-axis as 'pIC50'.
plt.xlabel('pIC50')

# Label the y-axis as 'Frequency'.
plt.ylabel('Frequency')

# Add a title to the plot: 'Distribution of pIC50 Values for Active Compounds'.
plt.title('Distribution of pIC50 Values for Active Compounds')

# Display the plot.
plt.show()


# In[425]:


# Create a box plot to compare the distribution of 'pIC50' values across different bioactivity classes.
# seaborn's 'boxplot' function is used for this purpose.
# The 'x' parameter represents the categories (here, 'bioactivity_class'), and 'y' represents the numeric values to be plotted (here, 'pIC50').
# 'data=df2' specifies the DataFrame where these columns are located.
sns.boxplot(x='bioactivity_class', y='pIC50', data=df2)

# Label the x-axis as 'Bioactivity Class'.
# This represents the different classes of bioactivity in the dataset.
plt.xlabel('Bioactivity Class')

# Label the y-axis as 'pIC50'.
# This represents the pIC50 values, a measure of the potency of a compound.
plt.ylabel('pIC50')

# Add a title to the plot: 'Box Plot of pIC50 by Bioactivity Class'.
# This title provides a clear description of what the plot is depicting.
plt.title('Box Plot of pIC50 by Bioactivity Class')

# Display the plot.
plt.show()


# In[426]:


# Preparing the data for Logistic Regression.

# Set 'X' as the predictor variable(s).
# Here, we're using only 'pIC50' from the DataFrame 'df2'.
# The double square brackets around 'pIC50' ensure that 'X' is a DataFrame, which is often required for scikit-learn models.
X = df2[['pIC50']]  # Predictor variable

# Create the target variable 'y'.
# The 'bioactivity_class' column is transformed into a binary format: 1 for 'active' and 0 for all other classes.
# This is done using the 'apply' method with a lambda function.
# In logistic regression, the target variable needs to be binary (or categorical for multinomial regression).
y = df2['bioactivity_class'].apply(lambda x: 1 if x == 'active' else 0)  # Target variable



# In[427]:


# Splitting the dataset into training and temporary sets.
# This is the first of two splits, where 60% of the data is allocated for training.
# 'train_test_split' from sklearn.model_selection is used for this purpose.
# 'X' is the DataFrame of predictor variables, and 'y' is the Series of the target variable.
# 'train_size=0.6' specifies that 60% of the data should go to the training set.
# 'random_state=42' ensures reproducibility, meaning the split will be the same each time the code is run.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=42)


# In[428]:


# Splitting the remaining data into validation and test sets.
# This is the second split, where the remaining 40% of the data (X_temp, y_temp) is divided equally into validation and test sets.
# 'train_test_split' is used again for this purpose.
# 'test_size=0.5' ensures that half of the remaining data goes to the test set, and half to the validation set.
# 'random_state=42' is set for reproducibility, ensuring the split is consistent each time the code is run.
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[429]:


# Creating and training a Logistic Regression model.
# 'LogisticRegression()' creates an instance of the LogisticRegression class from scikit-learn.
model = LogisticRegression()

# Fit the model to the training data.
# 'model.fit()' trains the model on 'X_train' and 'y_train'.
# 'X_train' contains the predictor variables for the training set, and 'y_train' contains the corresponding target variable.
model.fit(X_train, y_train)


# In[430]:


# Predicting probabilities on the validation set using the trained Logistic Regression model.
# 'model.predict_proba()' is used to get the probabilities that each instance in the validation set belongs to each class.
# The output is a 2D array where each row corresponds to an instance and each column to a class.
# The first column is the probability of the class being '0', and the second column is for class '1'.
# By using '[:, 1]', we select only the probabilities for class '1' (which, in our case, represents 'active').
# These probabilities are stored in 'y_valid_pred_probs'.
y_valid_pred_probs = model.predict_proba(X_valid)[:, 1]


# In[431]:


# Predicting classes on the validation set using the trained Logistic Regression model.
# The 'model.predict()' function is used to predict the class labels for each instance in the validation set.
# Unlike 'predict_proba', which provides probabilities, 'predict' returns the actual class predictions.
# These predictions are based on the probability threshold (usually 0.5) where probabilities above the threshold are classified as '1', and below as '0'.
# The predicted class labels are stored in 'y_valid_pred'.
y_valid_pred = model.predict(X_valid)


# In[432]:


# Evaluate the Logistic Regression model on the validation set.

# Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the validation set.
# 'roc_auc_score' compares the true binary labels 'y_valid' with the predicted probabilities 'y_valid_pred_probs'.
# AUC is a performance measurement for classification problems at various thresholds settings.
valid_auc = roc_auc_score(y_valid, y_valid_pred_probs)
print(f"Validation AUC: {valid_auc}")

# Print a classification report for the validation set.
# 'classification_report' provides key metrics like precision, recall, f1-score for each class, and support (the number of true instances for each label).
print("Validation Classification Report:")
print(classification_report(y_valid, y_valid_pred))

# Print a confusion matrix for the validation set.
# 'confusion_matrix' shows the number of correct and incorrect predictions broken down by each class.
# This matrix compares the true labels 'y_valid' against the predicted labels 'y_valid_pred'.
print("Validation Confusion Matrix:")
print(confusion_matrix(y_valid, y_valid_pred))


# In[433]:


# Predicting probabilities on the test set using the trained Logistic Regression model.
# 'model.predict_proba()' is used to obtain the probabilities that each instance in the test set belongs to each class.
# The output is a 2D array with each row corresponding to an instance and each column to a class.
# The first column is the probability of the class being '0', and the second column is for class '1'.
# By using '[:, 1]', we select only the probabilities for class '1' (which represents 'active' in this case).
# These probabilities are stored in 'y_test_pred_probs'.
y_test_pred_probs = model.predict_proba(X_test)[:, 1]

# Predicting class labels on the test set.
# 'model.predict()' is used to predict the class labels for each instance in the test set.
# Unlike 'predict_proba', which gives probabilities, 'predict' returns the actual class predictions.
# The predictions are based on a default probability threshold (usually 0.5) - above the threshold classified as '1', below as '0'.
# The predicted class labels are stored in 'y_test_pred'.
y_test_pred = model.predict(X_test)


# In[434]:


# Evaluate the Logistic Regression model on the test set.

# Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC) for the test set.
# The 'roc_auc_score' function compares the true binary labels 'y_test' with the predicted probabilities 'y_test_pred_probs'.
# The ROC AUC is a performance measure for classification problems at various threshold settings.
test_auc = roc_auc_score(y_test, y_test_pred_probs)
print(f"Test AUC: {test_auc}")

# Print a classification report for the test set.
# The 'classification_report' function provides key metrics like precision, recall, and f1-score for each class, as well as support (the number of true instances for each label).
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

# Print a confusion matrix for the test set.
# The 'confusion_matrix' function shows the number of correct and incorrect predictions broken down by class.
# It compares the true labels 'y_test' against the predicted labels 'y_test_pred'.
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


# In[435]:


# Plotting the distribution of 'pIC50' values for different bioactivity classes.

# Create a new figure for plotting with a specified size (10 inches wide by 6 inches tall).
plt.figure(figsize=(10, 6))

# Use seaborn's 'histplot' to create a histogram.
# The data is taken from the DataFrame 'df2', with 'pIC50' values on the x-axis.
# 'hue' parameter is used to differentiate the data points by 'bioactivity_class', which will give different colors to active and inactive classes.
# 'element='step'' changes the bars in the histogram to step-like structures.
# 'stat='density'' normalizes the histogram so that the area under the histogram sums to 1, making it a density plot.
# 'common_norm=False' ensures that the density is calculated for each class independently, providing a clearer comparison between classes.
sns.histplot(df2, x='pIC50', hue='bioactivity_class', element='step', stat='density', common_norm=False)

# Set the labels for x and y-axes and the title of the plot.
plt.xlabel('pIC50')
plt.ylabel('Density')
plt.title('Density Distribution of pIC50 Values by Bioactivity Class')

# Display the plot.
plt.show()


# In[436]:


# Cross-Validation of the Logistic Regression Model.

# Create a KFold object for splitting the data.
# 'KFold' is used for cross-validation, splitting the dataset into 'n_splits=5' different folds.
# 'random_state=42' is set for reproducibility of the splits.
# 'shuffle=True' ensures the data is shuffled before being split into batches, which is crucial for unbiased cross-validation.
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# Perform cross-validation.
# 'cross_val_score' computes the accuracy of the model for each fold of cross-validation.
# 'model' is the previously defined Logistic Regression model.
# 'X' and 'y' are the feature matrix and target vector, respectively.
# 'cv=kf' specifies that the KFold object 'kf' should be used for cross-validation.
# 'scoring='accuracy'' indicates that the model's accuracy is used as the scoring metric.
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Print the individual cross-validation scores and the average score.
# These scores represent the accuracy of the model for each fold.
print("Cross-Validation Scores for Logistic Regression:", cv_scores)
print("Average CV Score for Logistic Regression:", np.mean(cv_scores))


# In[437]:


# Saving the DataFrame of high potency compounds to a CSV file.
# 'index=False' is specified so that the DataFrame's index (row numbers) is not included in the CSV file, resulting in cleaner data.
very_high_potency_df.to_csv('insr_very_high_potency_compounds.csv', index=False)


# In[438]:


# Sorting the DataFrame 'very_high_potency_df' by 'pIC50' values in descending order.

# The '.sort_values()' method is used to sort the DataFrame.
# 'by='pIC50'' specifies that the sorting is to be done based on the 'pIC50' column.
# 'ascending=False' sorts the values in descending order, meaning the highest 'pIC50' values will be at the top of the DataFrame.
# The sorted DataFrame is assigned to 'sorted_very_high_potency_df'.
sorted_very_high_potency_df = very_high_potency_df.sort_values(by='pIC50', ascending=False)



# In[439]:


# Selecting the top ten most potent compounds from the sorted DataFrame.

# The '.head(10)' method is used to select the first ten rows from 'sorted_very_high_potency_df'.
# Since 'sorted_very_high_potency_df' is sorted by 'pIC50' in descending order, these first ten rows correspond to the ten most potent compounds.
# The selected rows are stored in a new DataFrame 'top_ten_compounds'.
top_ten_compounds = sorted_very_high_potency_df.head(10)


# In[440]:


# Printing the CHEMBL IDs and 'pIC50' values of the top ten most potent compounds.

# The 'print' function is used to display a message introducing the output.
print("Top 10 Potential Molecules:")

# The top ten compounds are already stored in 'top_ten_compounds'.
# By using [['molecule_chembl_id', 'pIC50']], we select only these two columns to display.
# 'molecule_chembl_id' likely represents the unique identifier for each compound in the CHEMBL database.
# 'pIC50' represents the potency of each compound.
# The resulting DataFrame slice is then printed, showing the identifiers and potencies of the top ten compounds.
print(top_ten_compounds[['molecule_chembl_id', 'pIC50']])


# In[ ]:





# In[ ]:





# In[ ]:




