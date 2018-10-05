
# coding: utf-8

# **Machine Learning Checklist**
# 1. Frame the problem and look at the big picture.
# 2. Get the data.
# 3. Explore the data to gain insights.
# 4. Prepare the data to better expose the underlying data patterns to Machine Learning algorithms.
# 5. Explore many different models and short-list the best ones.
# 6. Fine-tune your models and combine them into a great solution.
# 7. Present your solution.
# 8. Launch, monitor, and maintain your system.

# **1. Frame the problem and look at the big picture**
# - Goal: Predict the survival of the passengers in test set.
# - Objective: Creation of a supervised binary classifier (Survived: 1, Did Not Survive: 0)
# - Measurement of Performance: Create a confusion matrix and compute Accuracy, Recall, Precision and f-score.
# - List the assumptions.
# 
# **2. Get the data**
# - Completed
# 
# **3. Explore the data to get insights**
# - Study each attribute and its characteristics:
#     
#     * Name
#     
#     * Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
#     
#     * % of missing values
#     
#     * Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
#     
#     * Possibly useful for the task?
#     
#     * Type of distribution (Gaussian, uniform, logarithmic, etc.)
# 
# - For supervised learning tasks, identify the target attribute(s).
# - Visualize the data.
# - Study the correlations between attributes.
# - Study how you would solve the problem manually.
# - Identify the promising transformations you may want to apply.
# - Identify extra data that would be useful (go back to “Get the Data” on page 498).
# - Document what you have learned.
# 
# **4. Prepare the data to better expose the underlying data patterns to Machine Learning algorithms**
# - Work on copies of the data (keep the original dataset intact).
# 
# - Write functions for all data transformations you apply, for five reasons:
# 
#     * So you can easily prepare the data the next time you get a fresh dataset
#     
#     * So you can apply these transformations in future projects
#     
#     * To clean and prepare the test set
#     
#     * To clean and prepare new data instances once your solution is live
#     
#     * To make it easy to treat your preparation choices as hyperparameters
# 
# - Data cleaning:
# 
#     * Fix or remove outliers (optional).
#     
#     * Fill in missing values (e.g., with zero, mean, median...) or drop their rows (or columns).
#     
# - Feature selection (optional):
# 
#     * Drop the attributes that provide no useful information for the task.
# 
# - Feature engineering, where appropriate: 
# 
#     * Discretize continuous features.
# 
#     * Decompose features (e.g., categorical, date/time, etc.).
# 
#     * Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.).
# 
#     * Aggregate features into promising new features.
#     
# - Feature scaling: standardize or normalize features.
# 
# **5. Explore many different models and short-list the best ones**
# 
# **6. Fine-tune your models and combine them into a great solution**
# 
# **7. Present your solution**
# 
# **8. Launch, monitor, and maintain your system**

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
#%gui

from sklearn.model_selection import train_test_split


# In[2]:


import os
os.getcwd()


# **Data Loading**

# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# **Data Overview**

# In[4]:


train.info()


# In[5]:


train.isnull().sum()


# In[6]:


train.head()


# In[7]:


test.info()


# In[8]:


test.isnull().sum()


# **I) Exploring PassengerId**
# 1. Check if number all PassengerIds are unique
# 2. Checklist:
#     - Name: PassengerId
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Categorical/Index
# 
#     - % of missing values: 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): No
# 
#     - Possibly useful for the task?: Yes. Unique identifier for a passenger

# In[9]:


# Shape of train: (Rows, Columns)
print("Shape of train dataset is: Rows %s, Columns %s" %(train.shape))
# R Command: len(unique(x))
print("Number of unique PassengerIds is: %s" %(len(train['PassengerId'].unique())))

# Since, number of rows and number of unique PassengerIds are the same, it means there are no duplicates.


# **II) Survived**:
# 1. Checklist:
#     - Name: Survived
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Categorical, Binary, Survived: 1, Did not survive: 0
# 
#     - % of missing values: 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): No
# 
#     - Possibly useful for the task?: Yes. It is the label/output.

# In[10]:


# value_counts() : Equivalent R Command: table()
print(train['Survived'].value_counts())
print(train['Survived'].value_counts(normalize=True))

# 61% or 549 out of 891 dies. 38% or 342 out of 891 survived.
# based on this, survival rate is 38%.


# In[11]:


# R Command: colnames()
train.columns


# **III) Pclass**
# 1. Checklist:
#     - Name: Pclass
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Categorical with 3 levels: 1,2,3
# 
#     - % of missing values: 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): No
# 
#     - Possibly useful for the task?: Yes. Passengers of different Pclass have different survival rates.

# In[12]:


print('NAs in Pclass: %s' %(train['Pclass'].isnull().sum()))
print(train.groupby('Pclass')['Survived'].value_counts())
print((train.groupby('Pclass')['Survived'].value_counts(normalize=True)))

# Pclass 1 has highest survival rate at 62.96%.


# **IV) Name**
# 1. Checklist:
#     - Name: Name
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): String, May be used as an identifier
# 
#     - % of missing values: 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): White spaces, Non-ASCII
# 
#     - Possibly useful for the task?: Yes. Title (Mr, Miss, Mrs, Master etc) can be extracted for additional information such as social status which may be correlated to higher survival rates.

# In[13]:


print('NAs in Name: %s' %(train['Name'].isnull().sum()))
print(train['Name'])
# This will be explored further later.


# **V) Sex**
# 1. Checklist:
# 
#     - Name: Sex
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): String as of now (male, female), Needs to be converted into Categorical with 2 levels: Male - 0, Female - 1
# 
#     - % of missing values: 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): No
# 
#     - Possibly useful for the task?: Yes. Females have significanlty higher survival rates overall.

# In[14]:


print('NAs in Sex: %s' %(train['Sex'].isnull().sum()))
print(train['Sex'].value_counts())
print(train.groupby('Sex')['Survived'].value_counts())
print(train.groupby('Sex')['Survived'].value_counts(normalize=True))

# There are lesser number of females overall but their survival rate is higher.
# Analysis of survival of females by Pclass is performed as below.


# **VI) Pclass and Sex**
# 1. Checklist:
# 
#     - Name: Pclass and Sex
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Categorical
# 
#     - % of missing values: 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): No
# 
#     - Possibly useful for the task?: Yes. Females have significanlty higher survival rates overall in Pclass 1 and Pclass 2.

# In[15]:


print(train['Sex'].value_counts())
print(train.groupby(['Pclass', 'Sex'])['Survived'].value_counts())
print(train.groupby(['Pclass', 'Sex'])['Survived'].value_counts(normalize=True))
# Pclass 1: 96.8% females have survived whereas only 63.1% males survived.
# Pclass 2: 92.1% females have survived whereas 84.2% males have surived.
# Pclass 3: 50% females have survived whereas 86.4% males have survived.
# This clearly indicates that female with Pclass 1 and Pclass 2 have very high probability of survival.
# This will be useful feature later.


# **VII) Age**
# 1. Checklist:
# 
#     - Name: Age
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Continuous, Float, Needs to be converted into Categorical with pd.cut i.e. into Age Bands (0-15, 15-30, 30-45, 45-60, 60-75, >75), Range is from 0.42 to 80.0.
# 
#     - % of missing values: 177, Will need imputation as this is quite high.
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): No outliers, NAs present in significant number.
# 
#     - Possibly useful for the task?: Yes, maybe.

# In[16]:


# Age is a continuous variable. It will be explored as part of data visualization.
print("NAs in Age: %s" %(train['Age'].isnull().sum()))
#print(np.nanmedian(train['Age']))
print(max(train['Age']))


# **VIII) SibSp and Parch**
# 1. Checklist:
# 
#     - Name: SibSp, Parch
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Integer/Number, Can be considered Categorical
# 
#     - % of missing values: 0, 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): 
# 
#     - Possibly useful for the task?: Yes, maybe.

# In[17]:


print("NAs in SibSp: %s" %(train['SibSp'].isnull().sum()))
print("NAs in Parch: %s" %(train['Parch'].isnull().sum()))
print(train['SibSp'].value_counts())
print(train.groupby('SibSp')['Survived'].value_counts())
print(train.groupby('SibSp')['Survived'].value_counts(normalize=True))
print(train.groupby(['SibSp','Sex'])['Survived'].value_counts())
print(train.groupby(['SibSp','Sex'])['Survived'].value_counts(normalize=True))
# Overall, if you have 1 sibling/spouse, you have highest chance of survival.
# If you are a female with 0,1,2 sibling/spouse, you have more than 70% chance of survival.


# In[18]:


# Parch
print(train.groupby('Parch')['Survived'].value_counts())
print(train.groupby('Parch')['Survived'].value_counts(normalize=True))

#SibSp and Parch can be added/combined to form a single column of family.


# **IX) Ticket**
# 1. Checklist:
# 
#     - Name: Ticket
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Integer/Number, Can be considered Categorical
# 
#     - % of missing values: 0, 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): 
# 
#     - Possibly useful for the task?: No

# In[19]:


print(train['Ticket'].isnull().sum()) # No NA present
print(train[['Ticket', 'Pclass','Survived']])
# Ticket number does not appear to be providing any significant information and will be dropped.


# **X) Fare**
# 1. Checklist:
# 
#     - Name: Fare
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Float, Continuous, Needs to be converted into categorical or Fare Bands
# 
#     - % of missing values: 0
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): 
# 
#     - Possibly useful for the task?: Yes, maybe. Correlation with Pclass needs to be studied.

# In[20]:


print('NAs in Fare: %s' %(train['Fare'].isnull().sum()))
print(train[['Fare', 'Pclass', 'Survived']])
# Further exploration as part of data visualization where correlation with Pclass will also be determined.


# **XI) Cabin**
# 1. Checklist:
# 
#     - Name: Cabin
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Integer/Number, Can be considered Categorical
# 
#     - % of missing values: 687
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): 
# 
#     - Possibly useful for the task?: No. Needs to be dropped.

# In[21]:


print(train['Cabin'].isnull().sum()) # 687 NAs out of 891 rows => Will be dropped
print(train['Cabin'])


# **XI) Embarked**
# 1. Checklist:
# 
#     - Name: Embarked
# 
#     - Type (categorical, int/float, bounded/unbounded, text, structured, etc.): Categorical with 3 levels (C,Q,S)
# 
#     - % of missing values: 2
# 
#     - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.): NA
# 
#     - Possibly useful for the task?: Yes, maybe.

# In[22]:


print(train['Embarked'].isnull().sum()) # 2 NAs
print(train.groupby('Embarked')['Survived'].value_counts(normalize=False))
print(train.groupby('Embarked')['Survived'].value_counts(normalize=True))
print(train.groupby(['Embarked', 'Pclass'])['Survived'].value_counts(normalize=False))
print(train.groupby(['Embarked', 'Pclass'])['Survived'].value_counts(normalize=True))
print(train[['Embarked','Pclass','Survived']])
# Most people embarked from S. Highest urvival: Embarked C and Pclass 1


# **Data Visualization**
# 1. Distribution to decied upon feature engineering, feature scaling and transformation to be used.
# 2. Correlation between attributes to decide upon feature selection

# In[23]:


train.columns
#[u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age',
#       u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked']


# **I) PassengerId**
# - It acts as the index and unique identifier. Visulization is not necessary.

# **II) Survived**
# 1. Overall Survival Rates: Barplot for levels 0 and 1
# 2. Survival Rates by Sex: Barplot for male and female
# 3. Survival Rates by Pclass: Barplot for 1,2,3
# 4. Survival Rates by Sex for different Pclass: Barplot for male and female on Facet Grid with Pclass 

# In[31]:


fig = plt.figure()


# In[32]:


fig1 = fig.add_subplot(2,2,1)
fig2 = fig.add_subplot(2,2,2)
fig3 = fig.add_subplot(2,2,3)


# In[33]:


plt.plot([1.5, 3.5, -2, 1.6])


# In[34]:


plt.ion()
fig1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)


# In[35]:


plt.show()


# **XXXXXXXXXXXXXXXXXXXXXXXXX LATER XXXXXXXXXXXXXXXXXXXXXXXXXXXX**

# # Should be done much later. train and test need to be preprocessed first.
# **Creating X_train, X_test and y_train, y_test using train_test_split from sklearn**
# This split is made from the train dataset itself.
# The model is trained on X_train, y_train and tested on X_test and y_test.
# 

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),
                                                    train['Survived'], test_size=0.20,
                                                    random_state=42, shuffle=True)


# In[33]:


X_train.tail()


# In[32]:


y_train.tail()


# In[34]:


X_test.head()


# In[35]:


y_test.head()


# In[87]:


test.head()


# ### Copied from other notebook

# There are multiple approcahes to approcahing this problem.
# In general, the following steps are performed in creating a framework to operate on the given dataset.
# 1. Determine the relevant packages/libraries involved
# 2. Determine the nature of dataset - tabular/textual
# 3. Nature of operation to be performed: Data Analysis vs Machine Learning/Modelling
# 
# **Loading the relevant packages as below:**
# In Python, numpy, pandas, matplotlib and seaborn are the four packages that are needed for most of the data analysis and exploration purposes.
# These four can be considered to be uploaded by default epecially in cases where data is in tabular format and can be easily represented as  dataframe.
# 
# **sklearn** is a comprehensive library used for Machine Learning.
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# ### Part I: A Gentle Introduction
# **A Bird's Eye View: Exploration to obtain an overview**
# 

# In[ ]:


# info ===> R command: str()
print(train.info())
# This shows that the training dataset has 12 columns.
# Total number of rows and columns as well as datatypes of each columns
# is also provided here.
# This data is also provided in the metadata given on Kaggle.
# Survived is the label/output/column which will be predicted for test data.

# Printing out the rows and columns using .shape() ===> R command: dim()
print("\nTraining Dataset: Rows %s, Columns %s" %(train.shape[0], train.shape[1]))

# Related observations will be made below, after train.head() has been explored.


# In[ ]:


# info ===> R command: str()
print(test.info())
# This shows that the test dataset has 11 columns.
# Survived is the missing column which will be predicted for test data.

# Printing out the rows and columns using .shape() ===> R command: dim()
print("\nTest Dataset: Rows %s, Columns %s" %(test.shape[0], test.shape[1]))


# In[ ]:


# Exploring few rows of training dataset
# .head() ===> R Command: head()
# .tail() ===> R Command: tail()
train.head(10)
#train.tail(5)


# **A Small Taste of the Mystery: Some observations**
# 1. PassengerId is assigned to a person. It is expected that there will be a unique PassengerId for each passenger. This will be explored further by comparing the count of unique PassengerIds with the number of rows in the training dataset(891).
# 2. Survived column is the label/output. This column will be separated later into y_train (output) and rest of the dataframe as x_train (input). SInce labels are given for the training dataset, this is a Supervised Learning problem.
#                            x_train ========== OUR MODEL ============> y_predicted
#                            Objective: Minimize (y_train - y_predicted) i.e. Training Error
#     (PLEASE NOTE: This is a very simplistic representaion of training the model. Different issues of bias-variance tradeoff and overfitting will also come into the picture. These will be discussed later. Right now, the goal is to minimize training error to obtain a well trained model.)
#                                 
# 3. Pclass is a categorical variable with 3 values: 1,2,3. This is the same as the concept of factor() and levels in R.
# 4. Name of the passengers: SOme text cleaning and processing may be performed to get insights into survival chances.
# 5. Sex gives the gender. It appears to be a categorical variable with 2 levels.
# 6. Age is a numerical value. It can be converted into a categorical variable later on - dividing the population into different age groups: 0-5, 5-15, 15-30, 30-45, 45-60, 60-100. Also, this column has many missing values.
# 7. SibSp and Parch are also variables with numerical values.
# 8. Ticket gives the ticket number. It may not be a useful variable for our purposes as will be seen later.
# 9. Fare - numerical value. It may be strongly correlated with Pclass also. Thus, it may be left out to avoid overfitting.
# 10. Cabin - The very high number of missing values and the information itself appears to be useless for our purposes.
# 11. Embarked is also a categorical variable. It will need further exploration to determine its usefulness.
# 
# Overall, it appears clear that the following columns will be important:
# 1. Pclass
# 2. Sex
# 3. Age
# 4. Sibsp
# 5. Parch
# 6. Embarked - 'Lucky, maybe?
# 7. Name - Some information such as title may be derived from it.
# 
# The following columns can be dropped - 
# 1. Ticket
# 2. Fare
# 3. Cabin
# 
# PassengerId is the index.
# Survived is the output.
# 
