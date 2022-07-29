import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

data=pd.read_csv("house_pricesbysize.csv")

data.head()
data.tail()
data.info()
data.describe()

#normalizasyon
x=data.house size
y=data.price
x=(x-x.min())/(x.max()-x.min())
y=(y-y.min())/(y.max()-y.min())

lr=LinearRegression()
lr.fit(x.values.reshape(-1,1),y.values.reshape(-1,1))
y_predicted= lr.predict(x.values.reshape(-1,1))

r2_score(y, y_predicted)

mean_squared_error(y,y_predicted,squared=False)

b0=lr.intercept_[0].round(2)
b1=lr.coef_[0][0].round(2)

random_x=np.array([0,1])
plt.scatter(x.values, y.values, color='orange',marker='*')
plt.plot(random_x,b0+b1*random_x,color='purple',label='regresyon')
plt.legend()
plt.xlabel('House Area(m2)',color='red',fontsize=15)
plt.ylabel('Price(₺)',color='red',fontsize=15)
plt.title('Relationship Between Price and House Area',color='red',fontsize=14.5)
plt.show()

print("regression model Y = {} + {}*x".format(lr.intercept_[0].round(2),lr.coef_[0][0].round(2)))