# importing needed packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#importing datasets
df = pd.read_csv('synchronous machine.csv')

#Extracting Independent and dependent Variable
x= df.iloc[:, [2]]
y= df.iloc[:,[1]]

i= x.values.ravel().argsort()
x= x.values.ravel()[i].reshape(-1,1)
y= y.values[i]

#Fitting the Linear and Polynomial Regression to the dataset
lin_reg= LinearRegression()
lin_reg.fit(x,y)

poly_regs_d2= PolynomialFeatures(degree= 2)
x_poly_d2= poly_regs_d2.fit_transform(x)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly_d2, y)

poly_regs_d3 = PolynomialFeatures(degree= 3)
x_poly_d3= poly_regs_d3.fit_transform(x)
lin_reg_3 =LinearRegression()
lin_reg_3.fit(x_poly_d3, y)

poly_regs_d4 = PolynomialFeatures(degree= 4)
x_poly_d4= poly_regs_d4.fit_transform(x)
lin_reg_4 =LinearRegression()
lin_reg_4.fit(x_poly_d4, y)

#Visulaizing the result for Linear and Polynomial Regression
fig = plt.figure(figsize = (18,8))
plt.scatter(x,y,color="tab:blue",linewidth=4)
plt.plot(x,lin_reg.predict(x), color="tab:red",linewidth=3, label='Linear')
plt.plot(x, lin_reg_2.predict(x_poly_d2),color="tab:orange",linewidth=3, label='Degree 2')
plt.plot(x, lin_reg_3.predict(x_poly_d3),color="tab:green",linewidth=3, label='Degree 3')
plt.plot(x, lin_reg_4.predict(x_poly_d4),color="tab:pink",linewidth=3, label='Degree 4' )
plt.title("Linear and Polynomial Regression",fontsize = 20)
plt.xlabel("Power factor error (e)", fontsize = 15)
plt.ylabel("Power factor (PF)", fontsize = 15)
plt.legend(loc='lower right',fontsize=15)
plt.show()