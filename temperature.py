#!/usr/bin/python3

import cgi
import subprocess
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


print("content-type: text/html")
print("Access-Control-Allow-Origin: *")
print()

f=cgi.FieldStorage()
cmd=float(f.getvalue('x'))
model_humidity=joblib.load('/TC_forecast.pkl')

#print("cmd")
X_test=np.asarray([cmd])
X_test=X_test.reshape((-1,1))
poly = PolynomialFeatures(degree=4)
Xt = poly.fit_transform(X_test)

o=model_humidity.predict(Xt)
print(o[0][0])
