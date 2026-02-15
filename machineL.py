import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#df = pd.read_csv("1-studyhours.csv")
#print(df.head())
##ÇOKLU REGRESYON
#plt.scatter(df["Study Hours"],df["Exam Score"])
#plt.xlabel("Study Hours")
#plt.ylabel("Exam Score")
#plt.show()

# girdi her zaman data frame verilmelidir o yüzden çift köşeli parantez içine alırız

#X = df[["Study Hours"]]
#y = df["Exam Score"]

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

## Standardize the data set
#from sklearn.preprocessing import StandardScaler

#sclear = StandardScaler()

# fit_transform fit kısmı veriyi öğrenir ve istatislik kısmını yapar transform kısmı ise fit ile öğrenilen veriyi dönüştürür
# Model sadece train verisinden öğrenmeli
# Test verisinden öğrenirse veri sızıntısı (data leakage) olur
# o yüzden train kısmı fit_transform test kısmı ise trasform ile yapılandırılır

#X_train = sclear.fit_transform(X_train)
#X_test = sclear.transform(X_test)

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train,y_train)
#print("coefficients: ", regressor.coef_)
#print("intercept: ", regressor.intercept_)

#plt.scatter(X_train, y_train, color = "red")
#plt.plot(X_train, regressor.predict(X_train), color = "blue")
#print((plt.show()))

# 20 saat çalışmış bir insan kaç not alır



#print(sclear.transform([[20]]))
#print(regressor.predict(sclear.transform([[20]])))

#y_pred_test = regressor.predict(X_test)
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#mse = mean_squared_error(y_test, y_pred_test)
#mae = mean_absolute_error(y_test, y_pred_test)
#print("MSE: ", mse)
#print("MAE: ", mae)
#r2 = r2_score(y_test, y_pred_test)
#print("R2: ", r2)
#### POLİNOM REGRESYON


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline

df=pd.read_csv("3-customersatisfaction.csv")

#print(df.head())
df.drop("Unnamed: 0", axis=1, inplace=True)
plt.scatter(df["Customer Satisfaction"],df["Incentive"],color="blue")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")

plt.show()

X= df[["Customer Satisfaction"]]
y = df["Incentive"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
scaler=StandardScaler()
regressor = LinearRegression()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

score = r2_score(y_test,y_pred)
print(score)

plt.scatter(X_train,y_train)
plt.plot(X_train ,regressor.predict(X_train), color='red', )
plt.show()


poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regression = LinearRegression()
regression.fit(X_train_poly, y_train)

plt.scatter(X_train, y_train)
plt.scatter(X_train, regression.predict(X_train_poly), color = "r")
plt.show()

poly = PolynomialFeatures(degree=3, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)
score = r2_score(y_test, y_pred)
print(score)

plt.scatter(X_train, y_train)
plt.scatter(X_train, regression.predict(X_train_poly), color = "r")
plt.show()


###Yeni datayla  hesaplama yapma

new_df = pd.read_csv("3-newdatas.csv")

new_df.rename(columns={"0":"Customer Satisfaction"},inplace=True)
X_new = new_df[["Customer Satisfaction"]]
scaler.fit_transform(X_new)

X_new = scaler.transform(X_new)
X_new_poly = poly.transform(X_new)
y_new = regression.predict(X_new_poly)
plt.plot(X_new, y_new, "r", label="New Predictions")
plt.scatter(X_train, y_train, label="Training Points")
plt.scatter(X_test, y_test, label="Test Points")
plt.legend()
plt.show()


### Pipeline



