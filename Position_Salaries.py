#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
df=pd.read_csv('Position_Salaries.csv')
X=df.iloc[:,1:2].values
y=df.iloc[:,2].values

#scaling the dataset
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=y.reshape(-1,1)
y=sc_y.fit_transform(y)
y=np.ravel(y)

#building support vector regression model
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

#predicting the result
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualising the SVR model
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('SVR regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
