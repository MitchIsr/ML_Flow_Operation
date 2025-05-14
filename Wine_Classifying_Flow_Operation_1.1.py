#!/usr/bin/env python
# coding: utf-8

# # Assignment2 - Supervised Learning flow

# # Part 1(a) Student details:
# * Please write the First-Name, First letter of Last-Name and last 4 digits of the i.d. for each student. 

# ## Part 1(b) - Chat-GPT/other AI-agent/other assistance used:
# * If you changed the prompt until you got a satisfying answer, please add all versions
# * don't delete "pre" tags, so new-line is supported
# * double click the following markdown cell to change
# * press shift+enter to view
# * Add information:

# # Submitted by Michael I 5936

# In[1]:


# stundent details example: John S. 9812
#                       student details 1: Michael I 5936
# (if exists)           student details 2: 
# (if exists)           student details 3: 
# (if exists&premitted) student details 4: 


# #### Add information in this Markdown cell (double click to change, shift-enter to view)
# <pre>   
# AI agent name:ChatGPT
# Goal:helping to plot correlations based on heat map
# Propmpt1: https://chatgpt.com/share/682372cd-19bc-800f-b4a6-598de2e4a285
#     
# Propmpt2:
#     
# Propmpt3: 
# 
# 
# AI agent name 2:Grok
# Goal:finding Cross Validation grid search and k-fold syntax using my model 
# Propmpt1:   https://grok.com/chat/c1b0c8de-b8bc-425d-9904-46dde6cee4b5  how to create Cross validation grid search and
#             split data to k-fold validation groups
#     
# Propmpt2:   https://grok.com/share/bGVnYWN5_ca8ad480-0e14-4ff5-b6ba-ee89d7150dd1 how to create Confusion matrix for my test                 predictions 
#     
# Propmpt3:
# Other assistanse:    
# </pre>

# ## Part 1(c) - Learning Problem and dataset explaination.
# * Please explain in one paragraph
# * don't delete "pre" tags, so new-line is supported
# * double click the following markdown cell to change
# * press shift+enter to view
# * Add explaining text:

# #### Add information in this Markdown cell (double click to change, shift-enter to view)
# <pre>
# Wine Dataset , Wine classification into 3 different classes 
# Model methods: K-NN 
# Scaling Methods: Z-score
# Feature Engineering: PCA for KNN
# Hyper Parameters Search using grid search and k-fold combinations splitting train data
# and choosing best method of distance metric, PCA components, K parameters
# using f1 cross validation method.
# </pre>

# ## Part 2 - Initial Preparations 
# You could add as many code cells as needed

# In[39]:


import string
import re
import math
import statistics
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



# In[ ]:





# In[40]:


def load_dataset(X_file,Y_file, category_col_name):
    X_train = pd.read_csv(X_file)
    X_test = pd.read_csv(Y_file)
    Y_train = X_train[category_col_name]
    X_train = X_train.drop(columns=[category_col_name])
    Y_test = X_test[category_col_name]
    X_test = X_test.drop(columns=[category_col_name])
    return X_train, Y_train, X_test, Y_test


# In[41]:


X_file = 'wine_train.csv'
Y_file = 'wine_test.csv'
category_col_name = 'target'
x_train, y_train , x_test, y_test = load_dataset(X_file,Y_file,category_col_name)
x_train.head()
x_test.head()
y_train.head()
x_train.describe(include='all')
x_train.dtypes


# In[42]:


corr_matrix = x_train.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Wine Features')
plt.show()


# In[56]:


x_train_vis = x_train.copy(deep=True)
x_train_vis['target'] = y_train
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Flavanoids vs Total Phenols
sns.scatterplot(data=x_train_vis, x='flavanoids', y='total_phenols',hue='target', ax=axs[0, 0])
axs[0, 0].set_title('Flavanoids vs Total Phenols')

# 2. Flavanoids vs OD280/OD315_of_diluted_wines
sns.scatterplot(data=x_train_vis, x='flavanoids', y='od280/od315_of_diluted_wines',hue='target', ax=axs[0, 1])
axs[0, 1].set_title('Flavanoids vs OD280/OD315 of Diluted Wines')

# 3. Alcohol vs Proline
sns.scatterplot(data=x_train_vis, x='alcohol', y='proline',hue='target', ax=axs[1, 0])
axs[1, 0].set_title('Alcohol vs Proline')

# 4. Color Intensity vs Hue
sns.scatterplot(data=x_train_vis, x='malic_acid', y='hue',hue='target', ax=axs[1, 1])
axs[1, 1].set_title('Malic acid vs Hue')

plt.tight_layout()
plt.show()


# In[44]:


def fit_train(train_set, category_col_name, y_train):
    train_set_Copy = train_set.copy(deep=True)
    mean = train_set.mean()
    std = train_set.std()
    train_set_Copy = (train_set - mean) / std
    return train_set_Copy, mean, std

def fit_test(test_set, y_test, train_mean, train_std):
    test_set_Copy = test_set.copy(deep=True)
    test_set_Copy = (test_set - train_mean) / train_std
    return test_set_Copy


# In[66]:


X_train_normalized, mean , std = fit_train(x_train,category_col_name,y_train)
X_train_normalized.head()

x_train_vis1 = X_train_normalized.copy(deep=True)
x_train_vis1['target'] = y_train

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Flavanoids vs Total Phenols
sns.scatterplot(data=x_train_vis1, x='flavanoids', y='total_phenols',hue='target', ax=axs[0, 0])
axs[0, 0].set_title('Flavanoids vs Total Phenols')

# 2. Flavanoids vs OD280/OD315_of_diluted_wines
sns.scatterplot(data=x_train_vis1, x='flavanoids', y='od280/od315_of_diluted_wines',hue='target', ax=axs[0, 1])
axs[0, 1].set_title('Flavanoids vs OD280/OD315 of Diluted Wines')

# 3. Alcohol vs Proline
sns.scatterplot(data=x_train_vis1, x='alcohol', y='proline',hue='target', ax=axs[1, 0])
axs[1, 0].set_title('Alcohol vs Proline')

# 4. Color Intensity vs Hue
sns.scatterplot(data=x_train_vis1, x='malic_acid', y='hue',hue='target', ax=axs[1, 1])
axs[1, 1].set_title('Malic acid vs Hue')

plt.tight_layout()
plt.show()


# In[65]:


x_train_vis2 = X_train_normalized.copy(deep=True)
pca_vis = PCA(n_components=2)  
x_vis_pca = pca_vis.fit_transform(x_train_vis2)
x_train_vis2 = pd.DataFrame(x_vis_pca, columns=['pca1', 'pca2'])
x_train_vis2['target'] = y_train

    
    
plt.figure(figsize=(7, 5))
sns.scatterplot(data=x_train_vis2, x='pca1', y='pca2', hue='target')
plt.title('pca1 vs pca2')
plt.xlabel('pca1')
plt.ylabel('pca2')
plt.show()


# ## Part 3 - Experiments
# You could add as many code cells as needed

# In[47]:


def Calc_Distance(X_train_feature,X_test_feature,Method):
    if Method == 'Euclidean':
        return np.sqrt(np.sum((X_train_feature - X_test_feature) ** 2))
    if Method == 'Minikowski':
        p=3
        return np.power(np.sum(np.abs(X_train_feature - X_test_feature) ** p), 1/p)
    if Method == 'Manhattan':
        return np.sum(np.abs(X_train_feature - X_test_feature))


# In[48]:


def load_df_distance(X_train_normalized,X_test_normalized,y_train):
    df_euclidean = pd.DataFrame({'distance':np.zeros(len(y_train)),'y_train':y_train.values})
    return df_euclidean , 'distance'


# In[49]:


def ClassElement(y_train):
    return sorted(np.unique(y_train))


# In[50]:


def predict(df_euclidean_distance,col_name,element_lst,k):
    df_sorted = df_euclidean_distance.sort_values(by=col_name, ascending=True).head(k)
    # Count occurrences of each class in top k neighbors
    class_counts = pd.Series(0, index=element_lst, dtype=float)

    # Update class_counts with actual counts from top k neighbors
    actual_counts = df_sorted['y_train'].value_counts()
    for cls in actual_counts.index:
        if cls in class_counts.index:
            class_counts[cls] = actual_counts[cls]


    return class_counts.idxmax()


# In[51]:


def KNN(X_train_normalized,X_test_normalized,y_train,k,distance_method,pca_n):
    pca = PCA(n_components=pca_n)  # Keep 95% of variance
    x_train_pca = pca.fit_transform(X_train_normalized)
    x_test_pca = pca.transform(X_test_normalized)
    X_train_normalized = pd.DataFrame(x_train_pca)
    X_test_normalized = pd.DataFrame(x_test_pca)
    df_distance , df_e_col = load_df_distance(X_train_normalized,X_test_normalized,y_train)
    X_Predict = X_test_normalized.copy(deep=True)
    X_Predict['y_predict'] = np.nan
    element_lst = ClassElement(y_train)
    x_testsize = X_test_normalized.shape[0]
    df_distancesize = df_distance.shape[0]
    for j in range(x_testsize):
        test_instance = X_test_normalized.iloc[j]
        for i in range(df_distancesize):
            train_instance = X_train_normalized.iloc[i]
            df_distance.loc[i,df_e_col] = Calc_Distance(train_instance,test_instance,distance_method)
        X_Predict.loc[j,'y_predict'] = predict(df_distance,df_e_col,element_lst,k)
    return X_Predict


# In[54]:


Distance_Hyper_Parameters = {'Euclidean','Minikowski','Manhattan'}
K_Hyper_Parameters = {3, 5,7, 9}
PCI_Components = {2 , 0.95, 0.9}
results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for pci_comp in PCI_Components:
    for dist_param in Distance_Hyper_Parameters:
        for k in K_Hyper_Parameters:
            f1_scores = []
            for train_idx, val_idx in kf.split(X_train_normalized):
                # Split data into train and validation folds
                X_train_fold = X_train_normalized.iloc[train_idx]
                y_train_fold = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                X_val_fold = X_train_normalized.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
                
                # Run KNN on the fold
                X_Predict = KNN(X_train_fold,
                    X_val_fold,
                    y_train_fold,
                    k,
                    dist_param,
                    pca_n=pci_comp
                )
                
                # Extract predictions
                y_pred = X_Predict['y_predict']
                
                # Compute macro-average F1 score
                f1 = f1_score(y_val_fold, y_pred, average='macro')
                f1_scores.append(f1)
            
            # Compute mean F1 score across folds
            mean_f1 = np.mean(f1_scores)
            
            # Store results
            results.append({
                'PCA_Components': pci_comp,
                'Distance_Method': dist_param,
                'K': k,
                'Mean_Macro_F1': mean_f1
            })
            
          # Print progress
            print(f"PCA={pci_comp}, Distance={dist_param}, K={k}, Mean Macro F1={mean_f1:.4f}")

# Create summary table
results_df = pd.DataFrame(results)

# Sort by Mean_Macro_F1 in descending order
results_df = results_df.sort_values(by='Mean_Macro_F1', ascending=False)

# Display summary table
print("\nSummary Table of Hyperparameter Performance (5-Fold Cross-Validation):")
print(results_df)

# Identify best hyperparameters
best_params = results_df.iloc[0]
print("\nBest Hyperparameters:")
print(f"PCA Components: {best_params['PCA_Components']}")
print(f"Distance Method: {best_params['Distance_Method']}")
print(f"K: {best_params['K']}")
print(f"Mean Macro F1 Score: {best_params['Mean_Macro_F1']:.4f}")

# Save summary table to CSV (optional)
results_df.to_csv('knn_grid_search_results.csv', index=False)


# # Part 4 - Training 
# Use the best combination of feature engineering, model (algorithm and hyperparameters) from the experiment part (part 3)
# DF-engineering

# In[20]:


X_test_normalized = fit_test(x_test,y_test,mean,std)
Method = 'Manhattan'
pca_n = 0.95
k=3


# In[ ]:





# ## Part 5 - Apply on test and show model performance estimation

# In[29]:


X_test_Predict = KNN(X_train_normalized, X_test_normalized, y_train, k , Method, pca_n=pca_n)
X_test_Predict.head()
y_pred = X_test_Predict['y_predict']
y_unique_lst = ClassElement(y_train)
comparison = pd.DataFrame({'Predicted': y_pred,'Actual': y_test,'Correct': y_pred == y_test})
comparison


# In[38]:


macro_f1 = f1_score(y_test, y_pred, average='macro')  # Mean Macro F1 Score
accuracy = accuracy_score(y_test,y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix,  index=[f"True {unq}" for unq in y_unique_lst], columns=[f"Pred {unq}" for unq in y_unique_lst])
print(f"Mean Macro F1 Score: {macro_f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
conf_matrix_df


# In[ ]:




