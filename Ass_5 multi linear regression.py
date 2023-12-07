# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:33:19 2023

@author: SAMPATH
"""




#startups data

#STEP1:IMPORTING THE DATA
import numpy as np
import pandas as pd
df=pd.read_csv("50_Startups.csv")
df
df.dtypes
list(df)
df.shape
df.head()

#label encoding
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['State']=LE.fit_transform(df['State'])


#splitting as x nd y
Y=df["Profit"]
X=df.iloc[:,[0,1,2]] 

#standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X

#min-max scalar
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
MM_X = MM.fit_transform(X)
MM_X=pd.DataFrame(MM_X)
MM_X

#============Multi Linear Regression using standardization==================
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(SS_X,Y)
SS_Y_pred=LR.predict(SS_X)

#Metrics
from sklearn.metrics import r2_score
print("\n\t***Multi Linear Regression using standard scalar r^2 score***")
print("R square", r2_score(Y,SS_Y_pred).round(2))


#============Multi Linear Regression using min-max scalar==================
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
MM.fit(MM_X,Y)
MM_Y_pred=LR.predict(MM_X)


#Metrics
from sklearn.metrics import r2_score
print("\n\t***Multi Linear Regression using min-max scalar r^2score***")
print("R^square", r2_score(Y,MM_Y_pred).round(2))


#==============================================================================================

#ToyotaCorolla data

#STEP1:IMPORTING THE DATA
import numpy as np
import pandas as pd
df=pd.read_csv("ToyotaCorolla.csv",encoding="latin1")
df.dtypes
list(df)
df.shape
df.head()
df

#for logistic regression(belemeter=;)


#label encoding

#splitting as x nd y
Y=df["Price"]
#[age,km,hp,cc,doord,gears,quaterlttax,weight]
X=df.iloc[:,[3,6,8,12,13,15,16,17]] 

#standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X

#min-max scalar
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
MM_X = MM.fit_transform(X)
MM_X=pd.DataFrame(MM_X)
MM_X

#============Multi Linear Regression using standardization==================
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(SS_X,Y)
SS_Y_pred=LR.predict(SS_X)

#Metrics
from sklearn.metrics import r2_score
print("\n\t***Multi Linear Regression using standard scalar r^2 score***")
print("R square", r2_score(Y,SS_Y_pred).round(2))


#============Multi Linear Regression using min-max scalar==================
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
MM.fit(MM_X,Y)
MM_Y_pred=LR.predict(MM_X)


#Metrics
from sklearn.metrics import r2_score
print("\n\t***Multi Linear Regression using min-max scalar r^2score***")
print("R^square", r2_score(Y,MM_Y_pred).round(2))
