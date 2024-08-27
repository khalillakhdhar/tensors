import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data=pd.read_csv('house_prices_multiple_features.csv')
print(data.head())
X=data[['Superficie','Chambres','Age','Distance']].values
y=data['Prix'].values
model=LinearRegression()
model.fit(X,y)
y_predict=model.predict(X)

print("les parametres du modele sont: ",model.coef_,model.intercept_)
print("MSE: ",mean_squared_error(y,y_predict))
print("R2: ",r2_score(y,y_predict))
plt.scatter(data['Superficie'],y,color='blue',label='donnée réelle')
plt.scatter(data['Superficie'],y_predict,color='red',label='modèle')
plt.xlabel('Superficie')
plt.ylabel('Prix')
plt.title('Régression linéaire')
plt.legend()
plt.show()