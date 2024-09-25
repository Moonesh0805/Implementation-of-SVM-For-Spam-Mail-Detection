# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Step 1 :
Import the necessary python packages using import statements.

## Step 2 :
Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

## Step 3 :
Split the dataset using train_test_split.

## Step 4 :
Calculate Y_Pred and accuracy.

## Step 5 :
Print all the outputs.

## Step 6 :
End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MOONESH P
RegisterNumber: 212223230126
*/
```
```python
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["EmailText"].values
y=data["Label"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## DATA.HEAD() :
![image](https://github.com/user-attachments/assets/8ca107f2-77dc-40de-bb9d-dda88ecf3364)

## DATA.INFO() :
![image](https://github.com/user-attachments/assets/fd356ea6-78f2-45a4-a8b1-6a16acb4c661)

## DATA.ISNULL().SUM() :
![image](https://github.com/user-attachments/assets/3e7909d8-c081-4c69-9c24-82b78d02e17e)

## Y_PRED :
![image](https://github.com/user-attachments/assets/b9fbe932-c838-43f9-9906-54bb2fca2846)

## ACCURACY :
![image](https://github.com/user-attachments/assets/9990d171-9feb-4976-8560-ad10e33f3ccf)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
