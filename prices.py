import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv('house_prices.csv')
print(data.head())
print(data.describe())
X=data['Superficie'].values.reshape(-1,1) #reshape(-1,1) is used to convert 1D array to 2D array
y=data['Prix'].values.reshape(-1,1)
X_b=np.c_[np.ones((len(X),1)),X] #add x0=1 to each instance
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #normal equation
y_predict=X_b.dot(theta_best)
print("les parametres du modele sont: ",theta_best)
print("MSE: ",mean_squared_error(y,y_predict))
print("R2: ",r2_score(y,y_predict))
plt.scatter(X,y,color='blue',label='donnée réelle')
plt.plot(X,y_predict,color='red',label='modèle')
plt.xlabel('Superficie')
plt.ylabel('Prix')
plt.title('Régression linéaire')
plt.legend()
plt.show()