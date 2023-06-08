# Lec-1
# import pandas as pd
# data_set=pd.read_csv('https://query.data.world/s/l65seik3ppyan5cly62oywajfbw4dk',sep=';')
# print(data_set.shape)
# print(data_set.head(0))
# print("---------------------------------------------")
# X=data_set.iloc[:,:-1].values
# Y=data_set.iloc[:,-1].values
# print(data_set)
# print("---------------------------------------------")
# print(X)
# print("---------------------------------------------")
# print(Y)

# import matplotlib.pyplot as plt
# ax=plt.gca()
# ax.plot(X[:,2],Y,'bo')
# ax.set_xlabel("Age")
# ax.set_ylabel("G3")
# plt.show()

# ax2=plt.gca()
# ax2.hist(Y)
# plt.show()
# ---------------------------------------------------------------------------------------------------------------------------------

# Lec-2
# import pandas as pd
# data=pd.read_csv("https://www.w3schools.com/python/pandas/data.csv", sep=",",header=0)
# print(data)
# print(data.shape)
# print(data.head(0))
# print(data.nunique())
# print("----------------------------")
# print(data.dtypes)
# print(data.Duration.unique())
# print(data.dropna().shape)
# print(data.isnull().sum())
# data = data.dropna()
# data.dropna(subset=['Duration'], inplace = True)
# data.Duration = data.Duration.fillna(5)
# data["Duration"] = data["Duration"].fillna(5)
# data = data.fillna(15)
# X_mean = data["Duration"].mean()
# X_median = data["Pulse"].median()
# X_mode = data["Maxpulse"].mode()[0]
# data["Duration"] = data["Duration"].fillna(X_mean)
# data["Pulse"] = data["Pulse"].fillna(X_median)
# data["Maxpulse"] = data["Maxpulse"].fillna(X_mode)
# print(X_mean, X_median, X_mode)
# print(data['Duration'][0])
# data['Duration'] = pd.to_datetime(data['Duration'])
# print("------------------")
# print(data['Duration'][0])
# import numpy as np
# print(data['Duration'][0])
# data['Duration'] = np.float32(data['Duration'])
# print("------------------")
# print(data['Duration'][0])
# import matplotlib.pyplot as plt
# X = [i for i in range(data.shape[0])]
# plt.plot(X,data['Duration'])
# plt.show()
# plt.plot(X,data['Puls'])
# plt.show()
# print(data.duplicated())
# print(data.duplicated().sum())
# print(data.shape)
# data=data.drop_duplicates()
# print(data.shape)

# import pandas as pd
# data_set = pd.read_csv('https://www.dol.gov/sites/dolgov/files/ETA/naws/pdfs/NAWS_A2E197.csv')
# print(data_set.head(0))
# data_new=data_set.select_dtypes(exclude=["object"])
# data_new=data_set.drop(["AGE"],axis=1)
# data_new=data_set.drop(["AGE","ACCOMP"],axis=1)
# data = data.drop(0)
# data = data.drop([0, 15, 20])
# data = data.drop(range(10,20))
# data_new=data_set.rename(columns={"AGE":"LifeLong"})
# print(data_set.head(0))
# print(data_new.head(0))
# ---------------------------------------------------------------------------------------------------------------------------------

# Lec-3
# import pandas as pd
# data_set = pd.read_csv('https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD')
# print(data_set.shape)
# print(data_set.dtypes)
# from sklearn import preprocessing
# Pr_data = preprocessing.LabelEncoder()
# print(data_set['Electric Vehicle Type'])
# print("-------------------------")
# data_set['Electric Vehicle Type']=Pr_data.fit_transform(data_set['Electric Vehicle Type'])
# print(data_set['Electric Vehicle Type'])
# print(Pr_data.classes_)

# from sklearn import preprocessing
# d_types=data_set.dtypes
# for i in range(data_set.shape[1]):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         data_set[data_set.columns[i]]=Pr_data.fit_transform(data_set[data_set.columns[i]])
#         print("Column index = ", i)
#         print(Pr_data.classes_)

# from sklearn import preprocessing
# X=[[5], 
# [6], 
# [12], 
# [2], 
# [8]]
# scaler = preprocessing.MinMaxScaler()
# Xc = scaler.fit_transform(X)
# print(Xc)

# from sklearn.preprocessing import StandardScaler
# X=[[5], 
# [6], 
# [12], 
# [2], 
# [8]]
# scaler = StandardScaler()
# Xc = scaler.fit_transform(X)
# print(Xc)

# scaler = preprocessing.RobustScaler()
# robust_df = scaler.fit_transform(Xc)

# scaler = preprocessing.MaxAbsScaler()
# maxabs_df = scaler.fit_transform(Xc)

# scaler = preprocessing.Normalizer()
# normalizer_df = scaler.fit_transform(Xc)

# import pandas as pd
# from sklearn import preprocessing
# data_set = pd.read_csv('https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD')
# d_types=data_set.dtypes
# for i in range(data_set.shape[1]):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         data_set[data_set.columns[i]]=Pr_data.fit_transform(data_set[data_set.columns[i]])

# scaler = preprocessing.MinMaxScaler()
# Scaled_data = scaler.fit_transform(data_set) 
# Scaled_data = pd.DataFrame(Scaled_data,columns=data_set.columns)

# r=Scaled_data.corr()
# print(r)

# import seaborn as sns
# import matplotlib.pyplot as plt
# r=Scaled_data.corr()
# sns.heatmap(r, annot=True)
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.pairplot(Scaled_data)
# plt.show()
# ---------------------------------------------------------------------------------------------------------------------------------

# Lec-4
# from sklearn.datasets import load_iris
# X, Y = load_iris(return_X_y=True)

# from sklearn.feature_selection import r_regression
# r = r_regression(X,Y)
# print("Correlation = ", r)

# from sklearn.feature_selection import SelectKBest
# X_best = SelectKBest(r_regression, k=2)
# X_best.fit_transform(X,Y)
# print("index : ", X_best.get_support())

# X_new = X_best.fit_transform(X,Y)
# print("X_new = ", X_new)

# import numpy as np
# X = np.array([[4, 3, 0],
# [2, 1, 2],
# [6, 8, 1],
# [9, 0, 4],
# [8, 9, 8],
# [7, 4, 5]])
# Y = np.array([0, 0, 1, 1, 1, 0])
# from sklearn.feature_selection import f_oneway
# X_best = f_oneway(X,Y)
# print("Result = ", X_best[0])

# from sklearn.feature_selection import SelectKBest
# X_best = SelectKBest(f_oneway, k=2)
# X_best.fit_transform(X,Y)
# print("index : ", X_best.get_support())

# X_new = X_best.fit_transform(X,Y)
# print("X_new = ", X_new)

# import numpy as np
# X = np.array([[2, 0, 1],
# [2, 1, 0],
# [1, 2, 0],
# [1, 0, 1],
# [2, 0, 2],
# [0, 1, 2]])
# Y = np.array([4, 3, 7, 6, 11, 9])
# from sklearn.feature_selection import f_oneway
# X_best = f_oneway(X,Y)
# print("Result = ", X_best[0])

# from sklearn.feature_selection import SelectKBest
# X_best = SelectKBest(f_oneway, k=2)
# X_best.fit_transform(X,Y)
# print("index : ", X_best.get_support())

# X_new = X_best.fit_transform(X,Y)
# print("X_new = ", X_new)

# import pandas as pd
# Student_Grades=pd.DataFrame({
# 'Subject-1': ['A','A','C','B','B','C','B'],
# 'Subject-2': ['A','B','A','A','C','C','C'],
# 'Subject-3': ['A','C','C','C','B','A','C'],
# 'Acceptance': ['Yes','Yes','No','Yes','Yes','No','No']})

# from sklearn import preprocessing
# d_types=Student_Grades.dtypes
# for i in range(Student_Grades.shape[1]):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         Student_Grades[Student_Grades.columns[i]]=Pr_data.fit_transform(Student_Grades[Student_Grades.columns[i]])
#         print("Column index-", i, ": ", Pr_data.classes_)

# print(Student_Grades) 

# X = Student_Grades.iloc[:,:-1]
# Y = Student_Grades.iloc[:,-1]

# from sklearn.feature_selection import chi2
# C= chi2(X, Y)
# print("Values = ", C[0])        

# from sklearn.feature_selection import SelectKBest
# Xa = SelectKBest(chi2, k=2)
# Xa.fit_transform(X,Y)
# print("Index: ", Xa.get_support())

# X_new= Xa.fit_transform(X,Y)
# print("X_new = ", X_new)

# from sklearn.datasets import load_iris
# from sklearn.feature_selection import mutual_info_regression
# X, Y = load_iris(return_X_y=True)
# Values = mutual_info_regression(X,Y)
# print("Values = ", Values)

# from sklearn.datasets import load_iris
# X, Y = load_iris(return_X_y=True)
# from sklearn.feature_selection import SequentialFeatureSelector as SQ
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# model = SQ(knn, n_features_to_select='auto', direction="forward")
# model.fit(X,Y)
# print(model.get_support())

# from sklearn.datasets import load_iris
# X, Y = load_iris(return_X_y=True)
# from sklearn.feature_selection import SequentialFeatureSelector as SQ
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# model = SQ(knn, n_features_to_select='auto', direction="backward")
# model.fit(X,Y)
# print(model.get_support())

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_iris
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
# iris = load_iris()
# X = iris.data
# y = iris.target
# knn = KNeighborsClassifier(n_neighbors=3)
# efs = EFS(knn, scoring='accuracy')
# efs = efs.fit(X, y)
# features_index = efs.feature_groups
# print("\n",features_index)

# from sklearn.svm import SVR
# from sklearn.feature_selection import RFE
# from sklearn import datasets
# X, Y = datasets.load_iris(return_X_y=True)
# estimator = SVR(kernel="linear")
# selector = RFE(estimator, n_features_to_select=None)
# selector = selector.fit(X, Y)
# print(selector.support_)

# from sklearn.linear_model import Lasso
# X = [[0.9], [1.4], [2.4], [1.2]]
# Y = [0.8, 1.7, 2.1, 1.1]
# reg = Lasso(alpha=0.1)
# reg.fit(X,Y)
# print(reg.coef_)
# print(reg.intercept_)

# import pandas as pd
# data_set=pd.DataFrame({
# 'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny'], 
# 'Temp': ['Hot', 'Mild', 'Hot', 'Mild', 'Cool'], 
# 'Humidity': ['High', 'High', 'Normal', 'Normal', 'Normal'], 
# 'Windy': ['True', 'False', 'False', 'True', 'False'], 
# 'Play': ['No', 'Yes', 'No', 'Yes', 'Yes']})

# d_types=data_set.dtypes
# for i in range(data_set.shape[1]):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         data_set[data_set.columns[i]]=Pr_data.fit_transform(data_set[data_set.columns[i]])
#         print("Column index = ", i)
#         print(Pr_data.classes_)

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectFromModel
# X = data_set.iloc[:,:-1]
# Y = data_set.iloc[:,-1]
# sel = SelectFromModel(RandomForestClassifier())
# sel.fit(X, Y)
# print(sel.get_support())
# ---------------------------------------------------------------------------------------------------------------------------------

# Lec-5
# from sklearn.model_selection import train_test_split
# Speed=[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# X_train,X_test=train_test_split(Speed,test_size=0.3)
# print("X-Train = ",X_train)
# print("X-test = ",X_test)

# from sklearn.model_selection import train_test_split
# Speed=[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# X_train,X_test=train_test_split(Speed,test_size=0.3)
# print("X-Train = ",X_train)
# print("X-test = ",X_test)

# import numpy as np
# Speed = np.array([1, 2, 3, 4, 5])
# from sklearn.model_selection import LeavePOut
# model = LeavePOut(p=2)
# iteration_number = 0
# for index_train, index_test in model.split(Speed):
#     iteration_number = iteration_number + 1
#     print("iteration_number = ", iteration_number)
#     print("train data = ", Speed[index_train])
#     print("test data = ", Speed[index_test])
#     print("---------------------")

# import numpy as np
# Speed = np.array([1, 2, 3, 4, 5])
# from sklearn.model_selection import LeavePOut
# model = LeavePOut(p=1)
# iteration_number = 0
# for index_train, index_test in model.split(Speed):
#     iteration_number = iteration_number + 1
#     print("iteration_number = ", iteration_number)
#     print("train data = ", Speed[index_train])
#     print("test data = ", Speed[index_test])
#     print("---------------------")

# import numpy as np
# Speed = np.array([1, 2, 3, 4, 5])
# from sklearn.model_selection import KFold
# model = KFold(n_splits=5)
# iteration_number = 0
# for index_train, index_test in model.split(Speed):
#     iteration_number = iteration_number + 1
#     print("iteration_number = ", iteration_number)
#     print("train data = ", Speed[index_train])
#     print("test data = ", Speed[index_test])
#     print("---------------------")

# import numpy as np
# Speed = np.array([1, 2, 3, 4, 5])
# from sklearn.model_selection import RepeatedKFold
# model = RepeatedKFold(n_splits=5, n_repeats=3)
# iteration_number = 0
# for index_train, index_test in model.split(Speed):
#     iteration_number = iteration_number + 1
#     print("iteration_number = ", iteration_number)
#     print("train data = ", Speed[index_train])
#     print("test data = ", Speed[index_test])
#     print("---------------------")

# from sklearn.model_selection import StratifiedKFold
# import numpy as np
# X = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
# y= np.array([0,0,1,0,1,1])
# skf = StratifiedKFold(n_splits=3)
# for train_index,test_index in skf.split(X,y):
#     print("Y_train = ",y[train_index])
#     print("Y_test = ", y[test_index])
#     print("-----------------------------")

# from sklearn.model_selection import RepeatedStratifiedKFold
# import numpy as np
# X = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
# y= np.array([0,0,1,0,1,1])
# skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)
# number_iteration = 0
# for train_index,test_index in skf.split(X,y):
#     number_iteration = number_iteration + 1 
#     print("Number of iteration = ", number_iteration)
#     print("Y_train = ",y[train_index])
#     print("Y_test = ", y[test_index])
#     print("-----------------------------")

# from sklearn.model_selection import LeavePGroupsOut
# import numpy as np
# X = np.arange(6)
# y = [1, 1, 1, 2, 2, 2]
# groups = [1, 1, 2, 2, 3, 3]
# lpgo = LeavePGroupsOut(n_groups=2)
# for train, test in lpgo.split(X, y, groups=groups):
#     print("%s %s" % (train, test))    

# import numpy as np
# from sklearn.model_selection import ShuffleSplit
# X = np.arange(10)
# ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
# for train_index, test_index in ss.split(X):
#     print("%s %s" % (train_index, test_index))

# from sklearn.model_selection import GroupKFold
# X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
# y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
# groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
# gkf = GroupKFold(n_splits=3)
# for train, test in gkf.split(X, y, groups=groups):
#     print("%s %s" % (train, test))

# from sklearn.model_selection import LeaveOneGroupOut
# X = [1, 5, 10, 50, 60, 70, 80]
# y = [0, 1, 1, 2, 2, 2, 2]
# groups = [1, 1, 2, 2, 3, 3, 3]
# logo = LeaveOneGroupOut()
# for train, test in logo.split(X, y, groups=groups):
#     print("%s %s" % (train, test))    

# from sklearn.model_selection import GroupShuffleSplit
# X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
# y = ["a", "b", "b", "b", "c", "c", "c", "a"]
# groups = [1, 1, 2, 2, 3, 3, 4, 4]
# gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
# for train, test in gss.split(X, y, groups=groups):
#     print("%s %s" % (train, test))

# from sklearn.model_selection import TimeSeriesSplit
# import numpy as np
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4, 5, 6])
# tscv = TimeSeriesSplit(n_splits=3)    
# TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, 
# test_size=None)
# for train, test in tscv.split(X):
#     print("%s %s" % (train, test))
# ---------------------------------------------------------------------------------------------------------------------------------

# Lec-6
# Y = [17, 15, 12, 13, 14, 18, 16]
# Ypred = [19, 10, 21, 18, 11, 20, 14]
# import numpy as np
# error = sum(np.subtract(Ypred , Y))
# print("Standard Error = ", error)

# from sklearn.metrics import mean_absolute_error
# MAE = mean_absolute_error(Y, Ypred)
# print("Mean Absolute Error= ",MAE)

# from sklearn.metrics import mean_squared_error
# errors = mean_squared_error(Y, Ypred)
# print("mean squared error = ", errors)

# from sklearn.metrics import mean_squared_error
# RMSE = mean_squared_error(Y, Ypred, squared=False)
# print("Root Mean Square Error= ",RMSE)

# from sklearn.metrics import mean_squared_log_error
# A = mean_squared_log_error(Ypred, Y)
# print("MSLE = " ,A)

# from sklearn.metrics import mean_squared_log_error
# RMSLE= mean_squared_log_error(Ypred, Y, squared=False)
# print("RMSLE= ",RMSLE)

# from sklearn.metrics import r2_score
# r2 = r2_score(Y,Ypred) 
# print("R-Squared=",r2)

# from sklearn.metrics import r2_score
# r2 = r2_score(Y,Ypred) 
# n=len(Y)
# p=1
# adj_r2_score = 1 - ((1-r2)*(n-1)/(n-p-1))
# print("Adjusted R-Squared=",adj_r2_score)

# X = [[2], [5], [6], [10], [12], [13], [16]]
# Y = [1, 4, 6, 9, 11, 12, 14]

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.3)

# print("X-train = ", X_train)
# print("X-test = ", X_test)
# print("Y-train = ", Y_train)
# print("Y-test = ", Y_test)

# from sklearn.linear_model import LinearRegression
# RegressionModel=LinearRegression()

# RegressionModel.fit(X_train,Y_train)
# b=RegressionModel.intercept_
# a=RegressionModel.coef_
# print("a = ", a)
# print("b = ", b)

# Accuracy = RegressionModel.score(X_test,Y_test)
# print("Accuracy = ", Accuracy)

# Ypred = a * X_test + b

# from sklearn.metrics import r2_score
# r2 = r2_score(Y_test,Ypred)
# print("R-squared = ", r2)

# X = [[2], [5], [6], [10], [12], [13], [16]]
# Y = [1, 4, 6, 9, 11, 12, 14]
# from sklearn.linear_model import LinearRegression
# RegressionModel=LinearRegression()
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.3)
# RegressionModel.fit(X_train,Y_train)
# b=RegressionModel.intercept_
# a=RegressionModel.coef_
# Accuracy = RegressionModel.score(X_test,Y_test)
# print("Accuracy = ", Accuracy)
# ---------------------------------------------------------------------------------------------------------------------------------

# Lec-7
# from sklearn.metrics import hamming_loss
# y_true = [0, 0, 1, 0, 1]
# y_pred = [1, 0, 0, 0, 0]
# A = hamming_loss(y_true, y_pred)
# print("Hamming Loss = ", A)

# from sklearn.metrics import zero_one_loss
# y_true = [0, 0, 1, 0, 1]
# y_pred = [1, 0, 0, 0, 0]
# A = zero_one_loss(y_true, y_pred)
# print("Zero-one Loss (Average) = " ,A)
# B = zero_one_loss(y_true, y_pred, normalize=False)
# print("Zero-one Loss (Count) = " ,B)

# from sklearn.metrics import confusion_matrix
# y_true = [0, 1, 1, 0, 1, 1]
# y_pred = [1, 1, 0, 0, 0, 1]
# A = confusion_matrix(y_true, y_pred)
# print(A)

# TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
# print('TN = ', TN)
# print('FP = ', FP)
# print('FN = ', FN)
# print('TP = ', TP)

# from sklearn import metrics
# y_true = [0, 0, 1, 0, 1, 1, 0, 0, 1]
# y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0]
# A = metrics.precision_score(y_true, y_pred)
# print("Precision score = ",A)

# B = metrics.recall_score(y_true, y_pred)
# print("Recall score = ",B)

# C = metrics.f1_score(y_true, y_pred)
# print("F1 score = ",C)
# D = metrics.fbeta_score(y_true, y_pred, beta=0.5)
# print("Fbeta score (beta = 0.5) = ",D)
# E = metrics.fbeta_score(y_true, y_pred, beta=1)
# print("Fbeta score (beta = 1) = ",E)
# F = metrics.fbeta_score(y_true, y_pred, beta=2)
# print("Fbeta score (beta = 2) = ",F)

# from sklearn.metrics import classification_report
# y_true = [0, 0, 1, 0, 1, 1, 0, 0, 1]
# y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0]
# Classes = ['class 0', 'class 1']
# print(classification_report(y_true, y_pred, target_names = Classes))

# from sklearn.metrics import confusion_matrix
# y_true = [0, 2, 1, 0, 2, 0, 1, 1, 2]
# y_pred = [1, 2, 1, 0, 1, 2, 0, 1, 2]
# A = confusion_matrix(y_true, y_pred)
# print(A)

# from sklearn.metrics import multilabel_confusion_matrix
# y_true = [[0, 1, 1, 0, 0, 0, 1, 1, 0],
# [1, 1, 1, 1, 0, 1, 0, 0, 1]]
# y_pred = [[1, 1, 1, 0, 1, 0, 0, 0, 1],
# [0, 0, 1, 0, 1, 1, 0, 1, 1]]
# A = multilabel_confusion_matrix(y_true, y_pred)
# print(A)

# X = [[2], [3], [4], [5], [6], [8], [10], [12], [13], [15]]
# Y = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.3)

# print("X-train = ", X_train)
# print("X-test = ", X_test)
# print("Y-train = ", Y_train)
# print("Y-test = ", Y_test)

# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()
# model.fit(X_train,Y_train)
# b= model.intercept_
# a= model.coef_
# print("a = ", a)
# print("b = ", b)

# Y_pred = model.predict(X_test)
# print("Y_pred = ", Y_pred)

# from sklearn import metrics
# TN,FP,FN,TP=metrics.confusion_matrix(Y_test,Y_pred).ravel()
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

# X = [[2], [3], [4], [5], [6], [8], [10], [12], [13], [15]]
# Y = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size =0.3)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_train,Y_train)
# from sklearn import metrics
# Y_pred = model.predict(X_test)
# P = metrics.precision_score(Y_test, Y_pred)
# print("Precision score = ",P)
# R = metrics.recall_score(Y_test, Y_pred)
# print("Recall score = ",R)
# A = metrics.accuracy_score(Y_test, Y_pred)
# print("Accuracy = ",A)
# ---------------------------------------------------------------------------------------------------------------------------------

# Lec-8
# from sklearn.metrics import matthews_corrcoef
# y_true = [0, 1, 0, 0]
# y_pred = [1, 0, 1, 0]
# A = matthews_corrcoef(y_true, y_pred)
# print("Matthews correlation coefficient = ",A)

# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# from sklearn.metrics import multilabel_confusion_matrix
# A = multilabel_confusion_matrix(y_true, y_pred)
# print("confusion matrix = ")
# print(A)

# from sklearn.metrics import matthews_corrcoef
# mcc = matthews_corrcoef(y_true, y_pred)
# print("MCC = ", mcc)

# X = [[0.06, 0.22], 
# [0.17, 0.06], 
# [0.26, 0.37], 
# [0.36, 0.25], 
# [0.17, 0.16], 
# [0.56, 0.31], 
# [0.65, 0.42], 
# [0.74, 0.38], 
# [0.83, 0.22], 
# [0.93, 0.35]]
# Y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]

# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size =0.3)
# from sklearn.svm import SVC
# model = SVC(kernel='linear', probability=True)
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

