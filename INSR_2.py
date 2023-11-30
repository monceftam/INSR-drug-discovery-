#!/usr/bin/env python
# coding: utf-8

# In[145]:


# Basic data handling and operations
import pandas as pd
import numpy as np

# Visualization libraries for data exploration and presenting results
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis and hypothesis testing
from scipy.stats import mannwhitneyu
import statsmodels.api as sm

# Machine Learning model and evaluation tools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cheminformatics tools for handling and analyzing chemical data
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


# In[146]:


# Load the dataset from a CSV file.
# The pandas function 'pd.read_csv()' is used to read the CSV file 'insr_very_high_potency_compounds.csv'.
# The contents of the file are stored in a DataFrame 'df'.
# This file is assumed to contain data on compounds with very high potency, possibly with details like CHEMBL IDs and pIC50 values.
df = pd.read_csv('insr_very_high_potency_compounds.csv')


# In[147]:


# Define a function 'lipinski_rule' to evaluate compounds based on Lipinski's Rule of Five.
# Lipinski's Rule of Five is a set of guidelines for determining druglikeness of compounds.
# The function takes a compound's SMILES (Simplified Molecular Input Line Entry System) notation as input.
def lipinski_rule(smiles):
    # Convert the SMILES notation to a molecular object using RDKit.
    mol = Chem.MolFromSmiles(smiles)

    # If the molecule is valid, calculate its properties and check if they comply with Lipinski's rules.
    if mol:
        mw = Descriptors.MolWt(mol)       # Molecular weight
        logp = Descriptors.MolLogP(mol)   # LogP value (octanol-water partition coefficient)
        hbd = Lipinski.NumHDonors(mol)    # Number of hydrogen bond donors
        hba = Lipinski.NumHAcceptors(mol) # Number of hydrogen bond acceptors

        # Return True if the molecule adheres to all four of Lipinski's rules:
        # Molecular weight <= 500, LogP <= 5, <= 5 hydrogen bond donors, <= 10 hydrogen bond acceptors.
        return (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
    else:
        # Return False if the molecule is not valid (e.g., if the SMILES string couldn't be parsed).
        return False


# In[148]:


# Apply Lipinski's rule to each compound in the DataFrame.

# The 'apply' method is used to apply the 'lipinski_rule' function to each entry in the 'canonical_smiles' column of 'df'.
# 'canonical_smiles' presumably contains the SMILES (Simplified Molecular Input Line Entry System) representation of each compound.
# The 'lipinski_rule' function evaluates whether each compound adheres to Lipinski's Rule of Five.
# The results (True or False) are stored in a new column 'Lipinski_Rule_Passed' in the DataFrame.
# True indicates that the compound passes Lipinski's Rule of Five, and False indicates it does not.
df['Lipinski_Rule_Passed'] = df['canonical_smiles'].apply(lipinski_rule)


# In[149]:


# Calculate the molecular weight descriptor for each compound in the DataFrame.

# The 'apply' method is used to apply a function to each entry in the 'canonical_smiles' column of 'df'.
# 'canonical_smiles' column contains the SMILES (Simplified Molecular Input Line Entry System) representation of each compound.
# The lambda function within 'apply' converts each SMILES string to a molecular object using RDKit's 'Chem.MolFromSmiles'.
# Then, RDKit's 'Descriptors.MolWt' function is used to calculate the molecular weight of each compound.
# If the SMILES string is not valid (i.e., 'Chem.MolFromSmiles(x)' returns None), the lambda function returns None.
# The result of this calculation is stored in a new column in the DataFrame 'df'. The name of this new column is given by the variable 'descriptor'.
df[descriptor] = df['canonical_smiles'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None)


# In[150]:


# Splitting the dataset into two groups based on Lipinski's rule compliance.

# Create a subset 'pass_lipinski' containing the descriptor values for compounds that passed Lipinski's rule.
# 'df[df['Lipinski_Rule_Passed']]' filters the DataFrame to include only rows where 'Lipinski_Rule_Passed' is True.
# '[descriptor]' selects the column specified by the variable 'descriptor', which contains the molecular descriptor values.
# '.dropna()' removes any rows where the descriptor value is missing (NaN).
pass_lipinski = df[df['Lipinski_Rule_Passed']][descriptor].dropna()

# Similarly, create a subset 'fail_lipinski' containing the descriptor values for compounds that failed Lipinski's rule.
# 'df[~df['Lipinski_Rule_Passed']]' filters the DataFrame to include only rows where 'Lipinski_Rule_Passed' is False (indicated by '~').
# Again, '[descriptor]' selects the same molecular descriptor column, and '.dropna()' removes any rows with missing values.
fail_lipinski = df[~df['Lipinski_Rule_Passed']][descriptor].dropna()


# In[151]:


# Counting the number of instances for each category in the 'Lipinski_Rule_Passed' column.

# The 'value_counts()' method is used on the 'Lipinski_Rule_Passed' column of the DataFrame 'df'.
# This method counts the number of occurrences of each unique value in the column.
# In this case, it counts how many compounds passed and failed Lipinski's Rule of Five.
# The results are stored in 'pass_fail_counts', which will have the count for True (passed) and False (failed) values.
pass_fail_counts = df['Lipinski_Rule_Passed'].value_counts()


# In[152]:


# Creating a bar plot
sns.barplot(x=pass_fail_counts.index, y=pass_fail_counts.values)


# In[153]:


# Display the plot
plt.show()


# In[154]:


# Perform the Mann-Whitney U test
u_statistic, p_value = mannwhitneyu(pass_lipinski, fail_lipinski)


# In[155]:


# Print the U-Statistic and P-Value
print(f"Mann-Whitney U Test for {descriptor}: U-Statistic = {u_statistic}, P-value = {p_value}")
interpretation = 'Different distribution (p < 0.05)' if p_value < 0.05 else 'Same distribution (p >= 0.05)'
print(f"Interpretation: {interpretation}")


# In[156]:


print(df.columns)


# In[157]:


# Calculate additional descriptors for the DataFrame
df['LogP'] = df['canonical_smiles'].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None)
df['HBD'] = df['canonical_smiles'].apply(lambda x: Lipinski.NumHDonors(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None)
df['HBA'] = df['canonical_smiles'].apply(lambda x: Lipinski.NumHAcceptors(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None)


# In[158]:


# Independent variables
X = df[['MW', 'LogP', 'HBD', 'HBA']]

# Dependent variable (binary: 1 for pass, 0 for fail)
y = df['Lipinski_Rule_Passed'].astype(int)


# In[159]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[160]:


# Create a logistic regression object
logistic_regressor = LogisticRegression()


# In[161]:


# Train the model using the training sets
logistic_regressor.fit(X_train, y_train)


# In[162]:


# Make predictions using the testing set
y_pred = logistic_regressor.predict(X_test)


# In[163]:


# Generating the classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[164]:


# Define independent variables (X) and dependent variable (y)
X = df[['MW', 'LogP', 'HBD', 'HBA']]
y = df['Lipinski_Rule_Passed'].astype(int)


# In[165]:


# Create a logistic regression model with balanced class weights
logistic_regressor = LogisticRegression(class_weight='balanced')


# In[166]:


# Define stratified k-fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[167]:


# Define different scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']


# In[168]:


from sklearn.model_selection import cross_val_score

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model using the training set
logistic_regressor.fit(X_train, y_train)

# Perform cross-validation and evaluate the model
cv_results = cross_validate(logistic_regressor, X_train, y_train, cv=stratified_kfold, scoring=scoring)

# Output the results from cross-validation
print(f"Cross-validation scores:\n")
for metric_name, scores in cv_results.items():
    if metric_name.startswith('test_'):
        metric_name_clean = metric_name.replace('test_', '')
        print(f"{metric_name_clean.capitalize()} scores: {scores}")
        print(f"Mean {metric_name_clean.capitalize()}: {scores.mean():.2f}")
        print(f"Standard Deviation {metric_name_clean.capitalize()}: {scores.std():.2f}\n")

# Predict on the testing set
y_pred = logistic_regressor.predict(X_test)

# Generate a classification report and confusion matrix for the testing set
print(f"Classification report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")


# In[169]:


passed_compounds = df[df['Lipinski_Rule_Passed'] == True]


# In[170]:


# Selecting the columns of interest
top_passed_compounds = passed_compounds[['molecule_chembl_id', 'canonical_smiles', 'action_type', 'ligand_efficiency']]


# In[171]:


# Display the top 10 compounds
print(top_passed_compounds.head(10))


# In[172]:


# Save the top 10 compounds to a CSV file
top_passed_compounds.head(50).to_csv('the_ten_compounds.csv', index=False)

