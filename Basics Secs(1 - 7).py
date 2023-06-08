# Sec-1
# No_user = [1, 3, 5, 7, 9, 10, 12, 14]
# Data_rate = [2.5, 2.4, 2, 1.5, 1.1, 1, 0.8, 0.5]
# for i in range(len(No_user)):
#     if No_user[i]==10 or No_user[i]==11:
#         print("#user=",No_user[i],", Data rate=",Data_rate[i])

# from sklearn import linear_model
# import numpy as np
# No_user = [1, 3, 5, 7, 9, 10, 12, 14]
# Data_rate = [2.5, 2.4, 2, 1.5, 1.1, 1, 0.8, 0.5]
# RegressionModel=linear_model.LinearRegression()
# RegressionModel.fit(np.column_stack([No_user]),Data_rate)
# print("a = ",RegressionModel.coef_)
# print("b = ",RegressionModel.intercept_)

# a = -0.16639857
# b = 2.7437890974084005
# No_user = [10, 11]
# for i in range(len(No_user)):
#     Data_rate = a * No_user[i] + b
#     print("#user=",No_user[i],", Data rate=",Data_rate)

# from sklearn import linear_model
# import numpy as np
# No_user = [1, 3, 5, 7, 9, 10, 12, 14]
# Data_rate = [2.5, 2.4, 2, 1.5, 1.1, 1, 0.8, 0.5]
# RegressionModel=linear_model.LinearRegression()
# RegressionModel.fit(np.column_stack([No_user]),Data_rate)
# x=[10, 11]
# y=RegressionModel.predict(np.column_stack([x]))
# for i in range(len(x)):
#     print("#user=",x[i],", Data rate=",y[i])
#------------------------------------------------------------

# Sec-2
# import numpy as np
# from sklearn import linear_model
# Villa_Price=np.array([[400, 5, 3, 2],
# [500, 7, 2, 5],
# [550, 4, 1, 8],
# [700, 6, 3, 12],
# [1000, 8, 2, 20]])
# X=Villa_Price[:,:-1]
# Y=Villa_Price[:,-1]
# linear=linear_model.LinearRegression()
# linear.fit(X,Y)
# print('b0 = ',linear.intercept_)
# print('b1, b2, b3 = ',linear.coef_)

# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# stud_Score=np.array([[40, 95],
# [50, 90],
# [60, 91],
# [70, 87],
# [80, 82],
# [90, 80],
# [100, 83],
# [110, 82],
# [120, 86],
# [130, 88],
# [140, 89]])
# X=stud_Score[:,:-1]
# Y=stud_Score[:,-1]
# poly_regs= PolynomialFeatures(degree= 3)
# x_poly= poly_regs.fit_transform(X)
# lin_reg =LinearRegression()
# lin_reg.fit(x_poly, Y)
# print('a0 = ',lin_reg.intercept_)
# print('0, a1, a2 = ',lin_reg.coef_)

# import matplotlib.pyplot as plt
# a0 = lin_reg.intercept_
# a1 = lin_reg.coef_[1]
# a2 = lin_reg.coef_[2]
# Y_predict = a0 + a1 * X + a2 * (X**2)
# plt.scatter(X,Y,color='blue')
# plt.plot(X,Y_predict,color='red')
# plt.xlabel("No of students")
# plt.ylabel("Average Score")
# plt.show()
#------------------------------------------------------------

# Sec-3
# from sklearn.datasets import make_circles
# X, Y = make_circles(n_samples=500, noise=0.13, random_state=42)
# import pandas as pd
# df = pd.DataFrame(dict(X1=X[:, 0], X2=X[:, 1], Y=Y))
# from sklearn.svm import SVC
# clf = SVC(kernel='poly', degree=4)
# clf.fit(X, Y)
# print("coefficient = ",clf._get_coef())
# print("Interception = ",clf.intercept_)
# import numpy as np
# Xpred=[[np.average(df['X1']), np.average(df['X2'])]]
# print("Prediction = ",clf.predict(Xpred))

# from sklearn.linear_model import LogisticRegression
# X=[[3], [5], [8], [10], [12], [17], [30], [40], [60], [70], [80]]
# Y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
# model = LogisticRegression()
# model.fit(X,Y)
# print("Coefficient : ",model.coef_)
# print("Intercept : ",model.intercept_)

# from sklearn.linear_model import LogisticRegression
# X=[[50], [175], [150], [180], [177], [70], [60], [55], [185], [172], [65]]
# Y=[0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
# import math
# Xm=30
# a= 0.1640865 
# b= -18.29982185
# Ypred =1/(1+ math.exp(1)**(-1*(a+b*Xm)))
# print("Predicted Value is ",Ypred)
# if Ypred<0.5:
#     print("Take Care! Some kids open the door")

# X = [[50, 25],
# [175, 70],
# [150, 65],
# [180, 85],
# [177, 65],
# [70, 30],
# [60, 23],
# [55, 32],
# [185, 82],
# [172, 72],
# [65, 70]]
# Y = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
# from sklearn.svm import SVC
# clf = SVC(kernel='linear', probability=True)
# clf.fit(X, Y)
# W1 = clf._get_coef()[0][0]
# W2 = clf._get_coef()[0][1]
# b = clf.intercept_
# import numpy as np
# X1=np.zeros((len(X),1))
# X2=np.zeros((len(X),1))
# for i in range(len(X)):
#     X1[i]=X[i][0]
#     X2[i]=(-W1*X1[i]-b)/W2
# import matplotlib.pyplot as plt
# for i in range(len(Y)):
#     if Y[i]==0:
#         plt.plot(X[i][0],X[i][1],'rx')
#     if Y[i]==1:
#         plt.plot(X[i][0],X[i][1],'bo')
# plt.plot(X1,X2,color='green') 
# plt.show() 
#------------------------------------------------------------

# Sec-4
# from sklearn import neighbors
# X=[[160, 60],
# [163, 61],
# [160, 59],
# [163, 60],
# [160, 64],
# [158, 59],
# [158, 63],
# [163, 64],
# [165, 61],
# [165, 62],
# [158, 58],
# [165, 65],
# [168, 62],
# [168, 63],
# [168, 66]]
# Y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
# clf=neighbors.KNeighborsClassifier(n_neighbors=5,metric="minkowski", p=2)
# clf.fit(X,Y)
# Xnew=[[161,61]]
# Ypred=clf.predict(Xnew)
# if Ypred==1:
#     print("It is a Large")
# if Ypred==0:
#     print("It is a Medium")

# from sklearn import neighbors
# import numpy as np
# Lab_Data= np.array([
# [-54.953095, -3.130912, 90.9],
# [-54.952725, -3.133013, 102.4],
# [-54.952627, -3.133861, 106.2],
# [-54.96709, -3.027957, 144.9],
# [-54.966862, -3.029513, 138.9],
# [-54.967549, -3.024846, 162.5],
# [-54.96822, -3.020171, 161.9],
# [-54.979361, -2.937596, 178.7],
# [-54.97957, -2.936039, 182.8],
# [-54.979781, -2.934482, 182],
# [-54.979154, -2.939156, 175.7],
# [-54.967318, -3.026402, 149.4],
# [-54.968581, -3.017016, 159.9]])
# X=Lab_Data[:,:-1]
# Y=Lab_Data[:,-1]
# clf=neighbors.KNeighborsRegressor(n_neighbors=7,metric="minkowski", p=2)
# clf.fit(X,Y)
# Xnew=[[-54.96005,-3.121]]
# Ypred=clf.predict(Xnew)
# print(Ypred)

# import numpy as np
# Comp_accpt=np.array(
# [['A','A','A','Yes'],
# ['A','B','C','Yes'],
# ['C','A','C','No'],
# ['B','A','C','Yes'],
# ['B','C','B','Yes'],
# ['C','C','A','No'],
# ['B','C','C','No']])
# X = Comp_accpt[:,:-1]
# Y = Comp_accpt[:,-1]
# from sklearn import preprocessing
# X_Classes=[]
# for i in range(len(X[0])):
#      Pr_data = preprocessing.LabelEncoder()
#      X[:,i]=Pr_data.fit_transform(X[:,i])
#      X_Classes.append(Pr_data.classes_)
# Y_data = preprocessing.LabelEncoder()
# Y=Y_data.fit_transform(Y)
# from sklearn.naive_bayes import CategoricalNB
# gnb = CategoricalNB()
# gnb.fit(X, Y)
# Xpred=[['B'], ['B'], ['C']]
# Xp=np.zeros((1,len(Xpred)))
# for i in range(len(X_Classes)):
#      Xp[0][i]=np.where(X_Classes[i]==Xpred[i])[0]
# Y_pred=gnb.predict(Xp)
# print("Acceptance is ", Y_data.classes_[Y_pred])     
#------------------------------------------------------------

# Sec-5
# X = [[24], [26], [27], [23], [11], [14], [13], [30], [10], [31]]
# Y = [54, 65, 66, 47, 36, 94, 88, 66, 50, 78]
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor()
# regressor.fit(X, Y)
# Ypred = regressor.predict([[20]])
# print(Ypred)
#------------------------------------------------------------

# Sec-6
# No Codes :)
#------------------------------------------------------------

# Sec-7
# X = [
# ['I1','I2','I3'],
# ['I2','I3','I4'],
# ['I4','I5'],
# ['I1','I2','I4'],
# ['I1','I2','I3','I5'],
# ['I1','I2','I3','I4']]
# from apyori import apriori
# rules=apriori(X,min_support=0.5,min_confidence=0.9)
# results= list(rules)

# for item in results:
#     pair = item[0]
#     items = [x for x in pair]
#     if len(items)>=3:
#         print("Rule: " , items[0:])
#         print("Support: " , str(item[1]))
#         print("Confidence: " , str(item[2][0][2]))
#         print("Lift: " , str(item[2][0][3]))

# import pandas
# dataset = pandas.DataFrame({
# 'X1': [0.2, 27.2, 67.6, 104.9, 117.1, 62.3, 12, 0.6, 0.2, 28.1], 
# 'X2': [70.9, 105.2, 119.3, 63.7, 12.3, 36, 40.98, 97.08, 13.79, 17.96], 
# 'X3': [10.09, 20, 93, 57.77, 36, 43.38, 10.38, 14.02, 17.88, 10.41], 
# 'X4': [20.52, 10.8, 20.67, 20.74, 56.67, 37.71, 77.7, 21.15, 14.10, 30.93],
# 'X5': [62.25, 51.3, 51.9, 47.1, 33.6, 66.03, 21.51, 12.12, 57, 57.47],
# 'X6': [11.67, 68.73, 25.20, 19.56, 37.5, 12.78, 90.9, 18.36, 27.45, 31.5],
# 'X7': [34.5, 37.2, 24.3, 38.16, 14.28, 72.3, 12.9, 27.03, 22.44, 21.33], 
# 'X8': [55.92, 38.40, 72, 21.63, 14.55, 31.65, 64.41, 54.9, 52.2, 48.6], 
# 'X9': [39.0, 67.77, 21.48, 12.33, 13.5, 59.19, 11.64, 71.52, 27.54, 21.51],
# 'X10': [43.8, 12.51, 85.8, 19.65, 28.26, 34.2, 36.6, 36.0, 24.0, 40.14],
# 'X11': [14.58, 76.5, 13.5, 28.24, 23.22, 23, 62.31, 40.53, 80.1, 23.19],
# 'X12': [16.11, 32.97, 68.76, 55.2, 67.77, 21.48, 12.33, 13.5, 28.24, 23.22]})
# Y = [150, 320, 76, 180, 207, 210, 330, 135, 282, 232]

# from sklearn.decomposition import PCA
# pca = PCA(n_components=None)
# X_pca = pca.fit_transform(dataset)
# explained_variance_ratio=pca.explained_variance_ratio_
# import numpy as np
# cumulative_variance = np.cumsum(explained_variance_ratio)
# optimal_components = np.where(cumulative_variance > 0.95)
# optimal_components = np.add(optimal_components, 1)

# pca = PCA(n_components=optimal_components[0][0])
# X = pca.fit_transform(dataset)

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X, Y)
# print("Interception = ", model.intercept_)
# print("Coefficient = ", model.coef_)