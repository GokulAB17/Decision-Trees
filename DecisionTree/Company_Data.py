#A cloth manufacturing company is interested to know about the segment or 
#attributes causes high sale. 
#Approach - A decision tree can be built with target variable Sale
# (we will first convert it in categorical variable) & 
#all other variable will be independent in the analysis.  


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#loading dataset Company_dataset
data = pd.read_csv(r"filepath\Company_Data.csv")

#Data Preprocessing and EDA
data.head()
len(data['Sales'].unique())
data.isnull().sum()
colnames = list(data.columns)
predictors = colnames[1:11]
target = colnames[0]
data["Sales"].max()
data["Sales"].min()

#new dataframe created for preprocessing
data_new=data.copy()

#Making categorical data for Target column Sales
# 1 : Sales<=5
# 2 :5>Sales<=10
# 3 : Sales>10
for i in range(0,len(data)):
    if(data_new["Sales"][i]<=5):
        data_new["Sales"][i]="<=5"
    elif(data_new["Sales"][i]<=10 and data_new["Sales"][i]>5):
        data_new["Sales"][i]="5>s<=10"
    else:    
        data_new["Sales"][i]=">10"
        
data_new.Sales.value_counts()
#Mapping columns which are categorical to dummy variables
data_new.Sales=data_new.Sales.map({"<=5":1,"5>s<=10":2,">10":3})
data_new.ShelveLoc=data_new.ShelveLoc.map({"Bad":1,"Good":3,"Medium":2})        
data_new.Urban=data_new.Urban.map({"Yes":1,"No":2})
data_new.US=data_new.US.map({"Yes":1,"No":2})        


# Splitting data into training and testing data set by 70:30 ratio

from sklearn.model_selection import train_test_split
train,test = train_test_split(data_new,test_size = 0.3)

#Decision Tree Model building and Validity checking
from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train  
np.mean(train.Sales == model.predict(train[predictors]))#1

# Accuracy = Test
np.mean(preds==test.Sales) # 0.6083
