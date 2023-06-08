# Sec-1
# import numpy as np
# import matplotlib.pyplot as plt
# X = [2, 3, 5, 8, 12, 14, 15, 17]
# Y = [5, 11, 6, 14, 8, 7, 13, 10]
# a_mean = sum(X)/len(X)
# b_mul = np.dot(X, Y)
# print("Mean of X = ", a_mean)
# print("X * Y = ", b_mul)
# ax=plt.gca()
# ax.plot(X,Y)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# table_employee =np.array([['Ahmed',32,'Male',8,1200],
# ['Rashed',40,'Male',15,1700],
# ['Beaky',36,'Female',16,1500],
# ['Jackson',28,'Male',5,1000],
# ['Giada',30,'Female',6,1200]])
# X = table_employee[:,:-1]
# Y = table_employee[:,-1]
# Y_male=[Y[i] for i in range(len(X[:,2])) if X[i,2]=='Male']
# Y_male_average=np.mean(Y_male,dtype=np.float64)
# Y_female=[Y[i] for i in range(len(X[:,2])) if X[i,2]=='Female']
# Y_female_average=np.mean(Y_female,dtype=np.float64)
# Y_gender=np.array([Y_male_average,Y_female_average])
# X_gender=np.array(['Male','Female'])

# ax1=plt.gca()
# ax1.bar(X[:,0],Y,color ='maroon',width = 0.4)
# plt.show()
# ax2=plt.gca()
# ax2.plot(X[:,1],Y,'rx')
# plt.show()
# ax3=plt.gca()
# ax3.bar(X_gender,Y_gender,color ='Green',width = 
# 0.4)
# plt.show()
# ax4=plt.gca()
# ax4.plot(X[:,3],Y,'bo')
# plt.show()
# --------------------------------------------------------------------------------------------------------------

# Sec-2
# import pandas as pd
# data_set = pd.read_csv('https://data.cdc.gov/api/views/pwn4-m3yp/rows.csv')
# print(data_set.shape)
# print(data_set.head(0))
# print(data_set.nunique())
# print(data_set.dtypes)
# print(data_set.Duration.unique())
# print(data_set.isnull().sum())

# import pandas as pd
# data_set = pd.read_csv(' https://raw.githubusercontent.com/alice/datasets/8a340248a85770a17e2d7d8e71b3740ab6afff70/Most-paidathletes/Athletes.csv')
# print(data_set)

# print(data_set.isnull().sum())
# data = data_set.dropna()
# data_set.dropna(subset=['Duration'], inplace = True)
# data_set["Duration"] = data_set["Duration"].fillna(5)
# data_set = data_set.fillna(15)
# X= data_set["Duration"].mean()
# data_set["Duration"] = data_set ["Duration"].fillna(X)
# data_set['Duration'] = pd.to_numeric(data_set ['Duration'])

# import matplotlib.pyplot as plt
# X = [i for i in range(data_set.shape[0])]
# plt.plot(X, data_set['Duration'])
# plt.show()

# print(data_set.duplicated())
# print(data_set.duplicated().sum())
# data_set=data_set.drop_duplicates()
# data_set = pd.read_csv('https://data.cdc.gov/.../rows.csv')
# print(data_set.shape)
# print(data.nunique())
# print(data.isnull().sum())
# print(data.duplicated())
# data=data.drop_duplicates()

# import pandas as pd
# data_set = pd.DataFrame({'Career': ["Engineer", None,"Doctor", "Pharmacy","Doctor"],
# 'Age': [None, 30, 40, 35, 32],
# 'Salary': [1500, 2000,1400, None, 1300]})

# print(data_set)
# print(data_set.isnull().sum())
# print(data_set.dtypes)
# print("with 1st col (object)",data_set["Career"].mean())
# print("with 2nd col (float64) is ",data_set["Age"].mean())
# print("with 1st col (object)",data_set["Career"].median())
# print("with 2nd col (float64) is ",data_set["Age"].median())
# print("with 1st col (object)",data_set["Career"].mode()[0])
# print("with 2nd col (float64) is ",data_set["Age"].mode()[0])
# --------------------------------------------------------------------------------------------------------------

# Sec-3
# import pandas as pd
# Laptop_data=pd.DataFrame(
# {
# 'Processor': ['Intel', 'AMD', 'Intel', 'Intel', 'AMD', 'Intel', 'AMD','Intel', 'AMD'],
# 'Color': ['Red', 'Black', 'Red', 'Blue', 'Black', 'Blue', 'Black', 'Red', 'Blue'],
# 'Speed (GHz)': [4.6, 4, 4.1, 4.4, 4.2, 4.3, 4.2, 4.9, 4.5],
# 'Physical Cores': [12, 8, 16, 10, 8, 6, 10, 12, 12],
# 'Wattage (W)': [90, 65, 70, 80, 85, 80, 75, 85, 90]
# }
# )

# from sklearn import preprocessing
# d_types=Laptop_data.dtypes
# for i in range(Laptop_data.shape[1]):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         Laptop_data[Laptop_data.columns[i]]=Pr_data.fit_transform(Laptop_data[Laptop_data.columns[i]])

# scaler = preprocessing.MinMaxScaler()
# Scaled_data = scaler.fit_transform(Laptop_data) 
# Scaled_data = pd.DataFrame(Scaled_data,columns=Laptop_data.columns)

# import seaborn as sns
# import matplotlib.pyplot as plt
# r=Scaled_data.corr()
# print(r)
# sns.heatmap(r, annot=True)
# plt.show()
# sns.pairplot(Scaled_data)
# plt.show()
# --------------------------------------------------------------------------------------------------------------

# Sec-4
# from sklearn.feature_selection import f_oneway
# X_best = f_oneway(X,Y)
# print("Result = ", X_best[0])

# from sklearn.feature_selection import r_regression
# r = r_regression(X,Y)
# print(“Correlation = ", r)

# import pandas as pd
# Salat=pd.DataFrame({
# 'Apple': ['A','A','C','B','B','C','B'],
# 'Orange': ['A','B','A','A','C','C','C'],
# 'Banana': ['A','C','C','C','B','A','C'],
# 'Mango': ['B','B','C','A','A','A','C'],
# 'Dish': ['A','B','C','B','B','B','C']})

# from sklearn import preprocessing
# d_types=Salat.dtypes
# for i in range(Salat.shape[1]):
#     if d_types[i]=='object':
#     Pr_data = preprocessing.LabelEncoder()
#     Salat[Salat.columns[i]]=Pr_data.fit_transform(Salat[Salat.columns[i]])
#     print("Column index-", i, ": ", Pr_data.classes_)

# from sklearn.feature_selection import chi2
# C= chi2(Salat[Salat.columns[:-1]], 
# Salat[Salat.columns[-1]])
# print("Values = ", C[0])

# X, Y = load_iris(return_X_y=True)
# from sklearn.feature_selection import SequentialFeatureSelector as SQ
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# model = SQ(knn, n_features_to_select='auto', direction="forward")
# model.fit(X,Y)
# print(model.get_support())

# X, Y = load_iris(return_X_y=True)
# from sklearn.feature_selection import SequentialFeatureSelector as SQ
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# model = SQ(knn, n_features_to_select='auto', direction="backword")
# model.fit(X,Y)
# print(model.get_support())

# from sklearn.linear_model import Lasso
# X = [[0.9], [1.4], [2.4], [1.2]]
# Y = [0.8, 1.7, 2.1, 1.1]
# reg = Lasso(alpha=0.1)
# reg.fit(X,Y)
# print(reg.coef_)
# print(reg.intercept_)

# from sklearn.linear_model import Ridge
# X = [[0.9], [1.4], [2.4], [1.2]]
# Y = [0.8, 1.7, 2.1, 1.1]
# reg = Ridge(alpha=0.1)
# reg.fit(X,Y)
# print(reg.coef_)
# print(reg.intercept_)

# from mlxtend.feature_selection import ExhaustiveFeatureSelector
# from sklearn.feature_selection import RFE
# --------------------------------------------------------------------------------------------------------------

# Sec-5
# import numpy as np
# X = np.array([[200, 10], 
# [180, 8], 
# [210, 10], 
# [150, 6], 
# [160, 6], 
# [130, 5], 
# [190, 9], 
# [140, 8], 
# [220, 10]])

# Y = np.array([5000, 4000, 5000, 3000, 3000, 2000, 4500, 2500, 5500]) 

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test=train_test_split(X, Y,test_size=0.5)
# print("X-Train = ",X_train)
# print("X-test = ",X_test)
# print("Y-Train = ",Y_train)
# print("Y-test = ",Y_test)

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test=train_test_split(X, Y,test_size=0.3)

# from sklearn.model_selection import LeavePOut
# model = LeavePOut(p=2)
# iteration_number = 0
# for index_train, index_test in model.split(X):
#     iteration_number = iteration_number + 1
#     print("iteration_number = ", iteration_number)
#     print("X-Train = ", X[index_train])
#     print("X-test = ", X[index_test])
#     print("Y-Train = ", Y[index_train])
#     print("Y-test = ", Y[index_test])
#     print("---------------------")

# from sklearn.model_selection import KFold
# model = KFold(n_splits=5)
# iteration_number = 0
# for index_train, index_test in model.split(X):
#     iteration_number = iteration_number + 1
#     print("iteration_number = ", iteration_number)
#     print("X-Train = ", X[index_train])
#     print("X-test = ", X[index_test])
#     print("Y-Train = ", Y[index_train])
#     print("Y-test = ", Y[index_test])
#     print("---------------------")

# from sklearn.model_selection import RepeatedKFold
# model = RepeatedKFold(n_splits=5, n_repeats=3)
# iteration_number = 0
# for index_train, index_test in model.split(X):
#     iteration_number = iteration_number + 1
#     print("iteration_number = ", iteration_number)
#     print("X-Train = ", X[index_train])
#     print("X-test = ", X[index_test])
#     print("Y-Train = ", Y[index_train])
#     print("Y-test = ", Y[index_test])
#     print("---------------------")    

# Y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0])
# from sklearn.model_selection import StratifiedKFold
# model = StratifiedKFold(n_splits=5)
# iteration_number = 0
# for index_train, index_test in model.split(X, Y):
#     iteration_number = iteration_number + 1
#     print("iteration_number = ", iteration_number)
#     print("X-Train = ", X[index_train])
#     print("X-test = ", X[index_test])
#     print("Y-Train = ", Y[index_train])
#     print("Y-test = ", Y[index_test])
#     print("---------------------")
# --------------------------------------------------------------------------------------------------------------

# Sec-6
# import pandas as pd
# data_set = pd.DataFrame({
# 'Birth Rate': [164.9, 161.3, 174.7, 194.8, 144.4, 176, 175.9, 183.6, 190.5],
# 'Death Rate': [6.1, 4.3, 3.7, 4.6, 6.7, 4.9, 4.2, 5.8, 5.5],
# 'Incidence rate': [411.6, 349.7, 430.4, 350.1, 505.4, 461.8, 404, 459.4, 510.9],
# 'Marriage rate': [52.5, 44.5, 54.2, 52.7, 57.8, 50.4, 54.1, 52.7, 55.9]})

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# Xc = scaler.fit_transform(data_set)

# X = Xc[:,:-1]
# Y = Xc[:,-1]

# from sklearn.linear_model import LinearRegression
# Model=LinearRegression()

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5)

# Model.fit(X_train, Y_train)

# Accuracy = Model.score(X_test, Y_test)
# print("Accuracy = ", Accuracy)

# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
# print("Accuracy = ", Accuracy)

# data_set = pd.DataFrame({
# 'X': [1, 2, 3, 4, 5, 6, 7, 8, 9],
# 'Y': [1, 20, 30, 25, 40, 60, 70, 90, 110]})

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# Xc = scaler.fit_transform(data_set)

# X = Xc[:,:-1]
# Y = Xc[:,-1]

# from sklearn.linear_model import LinearRegression
# Model=LinearRegression()

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

# Model.fit(X_train, Y_train)

# Accuracy = Model.score(X_test, Y_test)
# print("Accuracy = ", Accuracy)

# from sklearn.linear_model import LinearRegression
# Model=LinearRegression()

# from sklearn.preprocessing import PolynomialFeatures
# Model_poly = PolynomialFeatures(degree= 4)

# Xp_train = Model_poly.fit_transform(X_train)
# Xp_test = Model_poly.fit_transform(X_test)

# Model.fit(Xp_train, Y_train)

# Accuracy = Model.score(Xp_test, Y_test)
# print("Accuracy = ", Accuracy)

# data_set = pd.DataFrame({
# 'X': [1, 2, 3, 4, 5, 6, 7, 8, 9],
# 'Y': [1, 20, 30, 25, 40, 60, 70, 90, 110]})

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# Xc = scaler.fit_transform(data_set)

# X = Xc[:,:-1]
# Y = Xc[:,-1]

# from sklearn.linear_model import LinearRegression
# Model=LinearRegression()

# from sklearn.model_selection import LeavePOut
# Splitting = LeavePOut(p=3)

# import numpy as np
# Accuracy = []
# for index_train, index_test in Splitting.split(X):
#     Model.fit(X[index_train],Y[index_train])
#     Accuracy.append(Model.score(X[index_test],Y[index_test]))
# print("Accuracy = ", np.average(Accuracy))

# from sklearn.model_selection import KFold
# Splitting = KFold(n_splits=3)

# import numpy as np
# Accuracy = []
# for index_train, index_test in Splitting.split(X):
#     Model.fit(X[index_train],Y[index_train])
#     Accuracy.append(Model.score(X[index_test],Y[index_test]))
# print("Accuracy = ", np.average(Accuracy))

# from sklearn.model_selection import RepeatedKFold
# Splitting = RepeatedKFold(n_splits=3, n_repeats=5)

# import pandas as pd
# data_set = pd.DataFrame({
# 'X1': [1, 3, 4, 6, 8, 11, 13, 14, 15, 25],
# 'X2': [31, 42, 49, 63, 78, 30, 53, 74, 45, 55],
# 'X3': [21, 12, 19, 16, 17, 13, 15, 24, 10, 23], 
# 'Y': [3, 2, -3, -9, -15, 12, -8, -11, 8, -3]})

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# Xc = scaler.fit_transform(data_set)

# X = Xc[:,:-1]
# Y = Xc[:,-1]

# from sklearn.linear_model import LinearRegression
# Model=LinearRegression()

# from sklearn.model_selection import train_test_split
# for i in [0.15, 0.2, 0.25, 0.3, 0.4]:
#     X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = i)
#     Model.fit(X_train, Y_train)
#     Accuracy = Model.score(X_test, Y_test)
#     print("Spliting = ", i*100, "% , Accuracy = ", Accuracy)

# import pandas as pd
# data_set = pd.DataFrame({
# 'X': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 
# 1.2, 1.3, 1.4, 1.5],
# 'Y': [0.7, 1.2, 1.4, 1.35, 1.17, 0.95, 0.51, 0.15, -0.18, -0.64, 
# -1.3, -1.5, -1.8, -1.4, -1.7]})

# from sklearn.linear_model import LinearRegression
# Model=LinearRegression()

# from sklearn.preprocessing import PolynomialFeatures

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# Xc = scaler.fit_transform(data_set)

# X = Xc[:,:-1]
# Y = Xc[:,-1]

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

# for i in range (2,11):
#     Model_poly = PolynomialFeatures(degree= i)
#     Xp_train = Model_poly.fit_transform(X_train)
#     Xp_test = Model_poly.fit_transform(X_test)
#     Model.fit(Xp_train, Y_train)
#     Accuracy = Model.score(Xp_test, Y_test)
#     print("Degree = ", i, " , Accuracy = ", Accuracy)

# import pandas as pd
# data_set = pd.DataFrame({
# 'X1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 
# 1.2, 1.3, 1.4, 1.5],
# 'X2': [0.4, 0.3, 0.91, 0.92, 1.3, 0.92, 0.87, 0.54, 0.32, 0.11, 
# -0.21, -0.49, -0.68, -0.75, -1.4],
# 'Y': [0.67, 0.80, 0.82, 0.21, 1.51, 1.56, 0.64, -0.40, -0.53, 
# 0.55, 0.38, -0.57, -1.36, -0.83, -2.87]})

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# Xc = scaler.fit_transform(data_set)

# X = Xc[:,:-1]
# Y = Xc[:,-1]

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.35)
# Model.fit(X_train, Y_train)
# Accuracy = Model.score(X_test, Y_test)
# print("Accuracy = ", Accuracy)

# from sklearn.model_selection import LeavePOut
# Splitting = LeavePOut(p=3)
# Accuracy = []
# for index_train, index_test in Splitting.split(X):
#     Model.fit(X[index_train], Y[index_train])
#     Accuracy.append(Model.score(X[index_test], Y[index_test]))

# import numpy
# print("Accuracy = ", numpy.average(Accuracy))    

# from sklearn.model_selection import KFold
# Splitting = KFold(n_splits=4)
# Accuracy = []
# for index_train, index_test in Splitting.split(X):
#     Model.fit(X[index_train], Y[index_train])
#     Accuracy.append(Model.score(X[index_test], Y[index_test]))

# import numpy
# print("Accuracy = ", numpy.average(Accuracy))
# --------------------------------------------------------------------------------------------------------------

# Sec-7
# X = [
# [4, 4],
# [8, 5],
# [12, 3],
# [18, 5],
# [20, 8],
# [24, 3],
# [30, 8],
# [35, 10],
# [40, 5]]
# thirsty =['Extremely', 'Extremely', 'Normal', 'Normal', 
# 'Normal', 'Normal', 'Extremely', 'Extremely', 'Normal']

# Y = []
# for i in thirsty:
#     if i == 'Extremely':
#         Y.append(1)
#     if i == 'Normal':
#         Y.append(0) 
# print(Y)

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.3)
# print("X-train = ", X_train)
# print("X-test = ", X_test)
# print("Y-train = ", Y_train)

# from sklearn.svm import SVC
# clf = SVC(kernel='linear')
# clf.fit(X_train,Y_train)

# Y_pred = clf.predict(X_test)
# print("Y_pred = ", Y_pred)

# from sklearn import metrics
# TN, FP, FN, TP = metrics.confusion_matrix(Y_test, Y_pred).ravel()
# print("TN = ",TN)
# print("FP = ",FP)
# print("FN = ",FN)
# print("TP = ",TP)

# P = metrics.precision_score(Y_test, Y_pred)
# print("Precision score = ",P)
# R = metrics.recall_score(Y_test, Y_pred)
# print("Recall score = ",R)
# A = metrics.accuracy_score(Y_test, Y_pred)
# print("Accuracy = ",A)

# import numpy
# X = numpy.array([
# [4, 4],
# [8, 5],
# [12, 3],
# [18, 5],
# [20, 8],
# [24, 3],
# [30, 8],
# [35, 10],
# [40, 5]])
# thirsty = numpy.array(['Extremely', 'Extremely', 'Normal', 'Normal', 'Normal', 
# 'Normal', 'Extremely', 'Extremely', 'Normal'])

# Y = numpy.zeros((len(thirsty)))
# for i in range(len(thirsty)):
#     if thirsty[i] == 'Extremely':
#         Y[i] = 1
#     if thirsty[i] == 'Normal':
#         Y[i] = 0
# print(Y)

# from sklearn.svm import SVC
# clf = SVC(kernel='linear')

# from sklearn import metrics

# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=3)

# A = []
# for train_index,test_index in skf.split(X,Y):
#     clf.fit(X[train_index], Y[train_index])
#     Y_pred = clf.predict(X[test_index])
#     A.append(metrics.accuracy_score(Y[test_index], Y_pred))
# print("Accuracy = ",numpy.average(A))
# --------------------------------------------------------------------------------------------------------------

# Sec-8
# X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
# Y = [0, 1, 1, 0, 1, 0, 0, 0, 1, 0]

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size =0.2)

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_train,Y_train)

# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_estimator(model, X_test, Y_test)

# import matplotlib.pyplot as plt
# plt.show()

# from sklearn.metrics import roc_auc_score
# auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
# print("AUC = ", auc)

# from sklearn.metrics import roc_curve

# fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
# print("fpr = " ,fpr)
# print("tpr = " ,tpr)
# print("thresholds = " ,thresholds)

