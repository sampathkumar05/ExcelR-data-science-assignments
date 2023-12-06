# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:00:09 2023

@author: SAMPATH
"""


#DELIVERY DATA

#STEP1:IMPORTING THE DATA
import numpy as np
import pandas as pd
df=pd.read_csv("delivery_time.csv")
df
df.dtypes
list(df)
df.shape
df.head()


#STEP2:EDA(PLOTTING)
import matplotlib.pyplot as plt
plt.scatter(df['Delivery Time'],df['Sorting Time'])
plt.title("Scatter Plot Of Delivery Data")
plt.xlabel("Delivery Time")
plt.ylabel("Sorting Time")
plt.show()

df.corr()

#STEP3:DATA PARTITION
Y=df["Delivery Time"]
X=df[["Sorting Time"]]


#STEP4:MODEL FITTING
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)


#STEP5:METRICS
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n\t\t***DELIVERY DATA***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))

#SIMPLE LINEAR REGRESSION TRANSFORMATION TECHNIQUES
print("\n\n\t\t***TRANSFORMATION TECHNIQUES***")
#1.    X square   
df['Sorting Time_2']=df['Sorting Time']*df['Sorting Time']
df.head()

Y=df["Delivery Time"]
X=df[["Sorting Time_2"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n***X square transformation results***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


#2.     sqrt X    
df['Sorting Time_3']=np.sqrt(df['Sorting Time'])
df.head()

Y=df["Delivery Time"]
X=df[["Sorting Time_3"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n***sqrt X transformation results***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


#   log X    
df['Sorting Time_4']=np.log(df['Sorting Time'])
df.head()

Y=df["Delivery Time"]
X=df[["Sorting Time_4"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n***log X transformation results***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


#   inverse X    
df['Sorting Time_5']=(1/(df['Sorting Time']))
df.head()

Y=df["Delivery Time"]
X=df[["Sorting Time_5"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n***Inverse X transformation results***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))



#====================================================================




#SALARY DATA

#STEP1:IMPORTING THE DATA
import numpy as np
import pandas as pd
df=pd.read_csv("Salary_Data.csv")
df
df.dtypes
list(df)
df.shape
df.head()


#STEP2:EDA(PLOTTING)
import matplotlib.pyplot as plt
plt.scatter(df['Salary'],df['YearsExperience'])
plt.title("Scatter Plot Of Salary Data")
plt.xlabel("Salary")
plt.ylabel("YearsExperience")
plt.show()

df.corr()

#STEP3:DATA PARTITION
Y=df["Salary"]
X=df[["YearsExperience"]]

#STEP4:MODEL FITTING
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

#STEP5:METRICS
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n\t\t***SALARY DATA***")
print("Mean square error:",mse.round(2))
print("root mean squared error:",np.sqrt(mse).round(2))



#SIMPLE LINEAR REGRESSION TRANSFORMATION TECHNIQUES
print("\n\n\t\t***TRANSFORMATION TECHNIQUES***")
#1.    X square   
df['YearsExperience_2']=df['YearsExperience']*df['YearsExperience']
df.head()

Y=df["Salary"]
X=df[["YearsExperience_2"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n***X square transformation results***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


#2.     sqrt X    
df['YearsExperience_3']=np.sqrt(df['YearsExperience'])
df.head()

Y=df["Salary"]
X=df[["YearsExperience_3"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n***sqrt X transformation results***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


#   log X    
df['YearsExperience_4']=np.log(df['YearsExperience'])
df.head()

Y=df["Salary"]
X=df[["YearsExperience_4"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n***log X transformation results***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


#   inverse X    
df['YearsExperience_5']=(1/(df['YearsExperience']))
df.head()

Y=df["Salary"]
X=df[["YearsExperience_5"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n***Inverse X transformation results***")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))



