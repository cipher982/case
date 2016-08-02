
# coding: utf-8

# # Choosing the Ideal Job Type for an Applicant
# 
# ### Using clustering analysis to find the right fit
# 
# In the previous project, an attempt was made to create a best-fit model to predict the future success of a worker brought on by a recruiter. In this project, we will see if we can predict the best type of work for a new recruit, to help guarantee success in the future.

# ### 1. Pull in and pre-process data

# In[2]:

# Import libraries
import numpy as np
import pandas as pd


# In[3]:

# Read in the worker data

xls_file = pd.ExcelFile("Origami_Data.xlsx", encoding = 'utf-8')
worker_data = xls_file.parse('Client Information')
print "worker data read successfully!"


# In[4]:

n_office = np.shape(worker_data[worker_data['OFFICE/MANUAL']=='OFFICE'])[0]
n_manual = np.shape(worker_data[worker_data['OFFICE/MANUAL']=='MANUAL'])[0]

print "Number of workers in the office field: {}".format(n_office)
print "Number of workers in the manual labor field: {}".format(n_manual)


# #### Clean up values
# First make sure to clean up non-consistent data in columns state and gender

# In[12]:

# Make all state data shorthand and include gender only with M or F
worker_data['State'] = map(lambda x: x.lower(), worker_data['State'])
worker_data = worker_data.replace({'alabama':'al','florida':'fl','georgia':'ga','south carolina':'sc','louisiana':'la'}, regex=True)
states = ['al','fl','ga','la','sc']
worker_data = worker_data.loc[worker_data['State'].isin(states)]
gender = ['M','F']
worker_data = worker_data.loc[worker_data['Gender'].isin(gender)]

# Remove NaN
worker_data2 = worker_data.dropna(axis = 0, how = 'any', subset = ['Employed In Past 6 Months','Gender','Age','State','Education Level'])


# In[13]:

# Extract feature (X) and target (y) columns, and removing ID and Comments columns
feature_cols = ['Employed In Past 6 Months','Age','Gender','State','Education Level','OFFICE/MANUAL']
target_col = ['Placement Successful']


x_all = worker_data2[feature_cols]
y_all = worker_data2[target_col]



# #### Preprocess feature columns
# 
# It turns out there are a few non-numeric columns that need to be converted! One of them is simply `yes`/`no`, e.g. `'Employed In Past 6 Months'`. This can be reasonably converted into `1`/`0` (binary) values.
# 
# Other columns, like `State` and `Education Level`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `AL`, `GA`, `FL`, etc.), and assign a `1` to one of them and `0` to all others.
# 
# These generated columns are called _dummy variables_, and so we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to create these columns.

# In[14]:

# Convert the target feature Y/N -Placement Successful- to 1/0
y_all = y_all.replace({'Y':1, 'N':0})

def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['Yes', 'No'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            if col == 'OFFICE/MANUAL':
                pass
            else:
                col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'state' => 'state_AL', 'state_GA'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

x_all = preprocess_features(x_all)

print "Processed feature columns ({}):-\n{}".format(len(x_all.columns), list(x_all.columns))


# In[15]:

import sklearn.cross_validation as cv

x_office = x_all[x_all['OFFICE/MANUAL'] == 'OFFICE']
y_office = y_all[x_all['OFFICE/MANUAL'] == 'OFFICE']
x_office = x_office.drop('OFFICE/MANUAL', 1)


x_manual = x_all[x_all['OFFICE/MANUAL'] == 'MANUAL']
y_manual = y_all[x_all['OFFICE/MANUAL'] == 'MANUAL']
x_manual = x_manual.drop('OFFICE/MANUAL', 1)


# ### 3. Training and Evaluating Models
# 
# #### Running multiple trials and finding the mean accuracy of classifier
# 
# As the dataset is relatively small and the results can vary from run to run. We will average 50 trials to find the most likely probability to our future predictions, assuming we do not overfit too much.

# #### Create success model for office job

# In[16]:

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 
office_scores = []

for x in range(50):
    x_office_train, x_office_test, y_office_train, y_office_test = cv.train_test_split(x_office, y_office.values.ravel(), test_size=.3)
    clf = RandomForestClassifier(n_estimators = 30)
    clf = clf.fit(x_office_train,y_office_train)
    office_scores.append(clf.score(x_office_test, y_office_test))

print "The average score for this classifier over 50 trials is {:.2%}".format(np.mean(office_scores))


# #### Create success model for manual labor job

# In[17]:

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 
manual_scores = []

for x in range(50):
    x_manual_train, x_manual_test, y_manual_train, y_manual_test = cv.train_test_split(x_manual, y_manual.values.ravel(), test_size=.3)
    clf = RandomForestClassifier(n_estimators = 30)
    clf = clf.fit(x_manual_train,y_manual_train)
    manual_scores.append(clf.score(x_manual_test, y_manual_test))

print "The average score for this classifier over 50 trials is {:.2%}".format(np.mean(manual_scores))


# ## With this Random Forest Classifier model, we are able to predict the success rates of an applicant in each field of work with ~78% accuracy according to the dataset provided.
