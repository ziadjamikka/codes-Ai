# Lec-1
# Experience = [1, 5, 10, 15, 20, 25, 30]
# Salary = [500, 1400, 2300, 3000, 3500, 4500, 5200]
# for i in range(len(Experience)) :
#     if Experience[i] == 12 :
#         print(Salary[i])

# from sklearn import linear_model
# Experience = [[1], [5], [10], [15], [20], [25], [30]]
# Salary = [500, 1400, 2300, 3000, 3500, 4500, 5200]
# RegressionModel = linear_model.LinearRegression()
# RegressionModel.fit(Experience,Salary)
# print("a = " , RegressionModel.coef_)
# print("b = " , RegressionModel.intercept_)

# a = 156.6439523
# b = 542.248722316866
# Experience = 12
# Salary = a * Experience + b
# print(Salary)

# import pyttsx3
# import speech_recognition as sr
# from sklearn import linear_model
# import os
# Experience =[[1], [5], [10], [15], [20], [25], [30]]
# Salary=[500, 1400, 2300, 3000, 3500, 4500, 5200]
# RegressionModel=linear_model.LinearRegression()
# RegressionModel.fit(Experience,Salary)
# def listen():
#     r = sr.Recognizer()
#     mic = sr.Microphone(device_index=0)
#     with mic as source:
#         r.adjust_for_ambient_noise(source)
#         print("Try To Get Sound")
#         audio = r.listen(source)
#     try:
#         text = r.recognize_google(audio)
#         os.system('cls')
#         print(f'You said: {text}')
#         return int(text)
#     except:
#         print('[!] Error')
# def speaker(text):
#     engine = pyttsx3.init()
#     engine.say(f"{text}")
#     engine.runAndWait()
# a = RegressionModel.coef_
# b = RegressionModel.intercept_
# Experience = listen()
# Salary = a * Experience + b
# speaker(f'Salary = {Salary} For Experience {Experience}')

# from sklearn import linear_model
# Experience =[[1], [5], [10], [15], [20], [25], [30]]
# Salary=[500, 1400, 2300, 3000, 3500, 4500, 5200]
# RegressionModel=linear_model.LinearRegression()
# RegressionModel.fit(Experience,Salary)
# print("Experience=12 -> Salary=",RegressionModel.predict([[12]]))
#------------------------------------------------------------

# Lec-2
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
# Day_temperature=np.array([[1, 8],
# [100, 35],
# [180, 28],
# [250, 18],
# [340, 30]])
# X=Day_temperature[:,:-1]
# Y=Day_temperature[:,-1]
# poly_regs= PolynomialFeatures(degree= 2)
# x_poly= poly_regs.fit_transform(X)
# lin_reg =LinearRegression()
# lin_reg.fit(x_poly, Y)
# print('a0 = ',lin_reg.intercept_)
# print("0, a1, a2 = ",lin_reg.coef_)

# import matplotlib.pyplot as plt
# a0 = lin_reg.intercept_
# a1 = lin_reg.coef_[1]
# a2 = lin_reg.coef_[2]
# Y_predict = a0 + a1 * X + a2 * (X**2)
# plt.scatter(X,Y,color='blue')
# plt.plot(X,Y_predict,color='red')
# plt.xlabel("Day")
# plt.ylabel("Temperature")
# plt.show()
#------------------------------------------------------------

# Lec-3
#------------------------------------------------------------
# from sklearn.linear_model import LogisticRegression
# X=[[10], [8], [3], [1], [4], [6], [5]]
# Y=[1, 1, 0, 0, 0, 1, 1]
# model = LogisticRegression()
# model.fit(X,Y)
# print("Coefficient : ",model.coef_)
# print("Intercept : ",model.intercept_)
# print("Prediction for score=2 is ", model.predict([[2]]))
# print("Prediction for score=7 is ", model.predict([[7]]))
# print(model.predict_proba([[2]]))
# print(model.predict_proba([[7]]))

# from sklearn.linear_model import LogisticRegression
# X=[[40], [80], [43], [8], [6], [15], [25], [45], [12], [13], 
# [9], [11]]
# Y=[1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
# model = LogisticRegression()
# model.fit(X,Y)
# print("Coefficient : ",model.coef_)
# print("Intercept : ",model.intercept_)

# from sklearn.svm import SVC
# X = [[40, 16],
# [80, 20],
# [43, 14],
# [8, 4],
# [6, 5],
# [15, 12],
# [25, 15],
# [45, 18],
# [12, 10],
# [13, 8],
# [9, 7],
# [11, 9]]
# Y = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
# clf = SVC(kernel='linear')
# clf.fit(X, Y)
# print("coefficient = ",clf._get_coef())
# print("Interception = ",clf.intercept_)
# Xp = [[17,15]]
# Yp = clf.predict(Xp)
# if Yp==0:
# print("Cat")
# if Yp==1:
# print("Dog")
# print(clf.decision_function(Xp)) 

# from sklearn.datasets import make_circles
# X, Y = make_circles(n_samples=500, noise=0.13, random_state=42)
# import pandas as pd
# df = pd.DataFrame(dict(X1=X[:, 0], X2=X[:, 1], Y=Y))
# import matplotlib.pyplot as plt
# for i in range(len(df)):
#     if df['Y'][i]==0:
#         plt.plot(df['X1'][i],df['X2'][i],'bX')
#     if df['Y'][i]==1:
#         plt.plot(df['X1'][i],df['X2'][i],'ro')
# plt.show() 
#------------------------------------------------------------

# Lec-4    
# from sklearn import neighbors
# X=[ [15, 12],
# [12, 10],
# [25, 15],
# [13, 8],
# [11, 9],
# [9, 7],
# [8, 4],
# [6, 5],
# [40, 16],
# [43, 14],
# [45, 18],
# [80, 20]]
# Y = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
# clf=neighbors.KNeighborsClassifier(n_neighbors=5)
# clf.fit(X,Y)
# Xnew=[[17,15]]
# Ypred=clf.predict(Xnew)
# import matplotlib.pyplot as plt
# for i in range(len(Y)):
#     if Y[i]==0:
#         plt.plot(X[i][0],X[i][1],'ro')
#     if Y[i]==1:
#         plt.plot(X[i][0],X[i][1],'bo')
# plt.plot(Xnew[0][0],Xnew[0][1],'gx')
# plt.show()
    
# import pandas as pd
# data_set=pd.DataFrame({
# 'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 
# 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 
# 'Overcast', 'Overcast', 'Sunny'], 
# 'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 
# 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
# 'Humidity': ['High', 'High', 'High', 'High', 'Normal', 
# 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 
# 'High', 'Normal', 'High'],
# 'Windy': ['False', 'True', 'False', 'False', 'False', 
# 'True', 'True', 'False', 'False', 'False', 'True', 'True', 
# 'False', 'True'],
# 'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
# 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'] 
# })
# from sklearn import preprocessing
# d_types=data_set.dtypes
# for i in range(data_set.shape[1]):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         data_set[data_set.columns[i]]=Pr_data.fit_transform(data_set[data_set.columns[i]])
#         print("Column index = ", i)
#         print(Pr_data.classes_)
# X = data_set.iloc[:,:-1].values 
# Y = data_set.iloc[:,-1].values
# from sklearn.naive_bayes import CategoricalNB
# clf=CategoricalNB()
# clf.fit(X,Y)
# print(clf.predict([[1,0,0,1]]))

# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# Y = iris.target
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X, Y)
# print(gnb.predict([[3.5,3.7,2.6,0.8]]))
#------------------------------------------------------------

# Lec-5
# import pandas as pd
# data_set=pd.DataFrame({
# 'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 
# 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 
# 'Overcast', 'Overcast', 'Sunny'], 
# 'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 
# 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
# 'Humidity': ['High', 'High', 'High', 'High', 'Normal', 
# 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 
# 'High', 'Normal', 'High'],
# 'Windy': ['False', 'True', 'False', 'False', 'False', 
# 'True', 'True', 'False', 'False', 'False', 'True', 'True', 
# 'False', 'True'],
# 'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
# 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'] 
# })


# from sklearn import preprocessing
# d_types=data_set.dtypes
# for i in range(data_set.shape[1]-1):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         data_set[data_set.columns[i]]=Pr_data.fit_transform(data_set[data_set.columns[i]])
#         print(data_set.columns[i],": ",Pr_data.classes_)

# X = data_set.iloc[:,:-1]
# Y = data_set.iloc[:,-1]
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(criterion="entropy") #gini
# clf = clf.fit(X,Y)
# print(clf.predict([[1,0,0,1]]))


# import matplotlib.pyplot as plt
# from sklearn import tree
# fig = plt.figure(figsize=(5, 10))
# Pr_data.fit_transform(data_set[data_set.columns[-1]])
# tree.plot_tree(clf, 
# feature_names=data_set.columns[:-1], 
# class_names=Pr_data.classes_)
# plt.show()

# import pandas as pd
# data_set=pd.DataFrame({
# 'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 
# 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny'], 
# 'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 
# 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
# 'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
# 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
# 'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 
# 'False', 'False', 'True', 'True', 'False', 'True'],
# 'Target': [26, 30, 48, 46, 62, 23, 43, 36, 38, 48, 48, 62, 44, 30]})

# from sklearn import preprocessing
# d_types=data_set.dtypes
# for i in range(data_set.shape[1]-1):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         data_set[data_set.columns[i]]=Pr_data.fit_transform(data_set[data_set.columns[i]])
#         print(data_set.columns[i],": ",Pr_data.classes_)

# X = data_set.iloc[:,:-1]
# Y = data_set.iloc[:,-1]
# from sklearn.tree import DecisionTreeRegressor
# reg = DecisionTreeRegressor() 
# reg=reg.fit(X,Y)
# print(reg.predict([[1,0,0,1]]))

# from sklearn import tree
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(15, 20))
# tree.plot_tree(reg, feature_names=data_set.columns[:-1])
# plt.show()

# import pandas as pd
# data_set=pd.DataFrame({
# 'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 
# 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny'], 
# 'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 
# 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
# 'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
# 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
# 'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 
# 'False', 'False', 'True', 'True', 'False', 'True'],
# 'Target': [26, 30, 48, 46, 62, 23, 43, 36, 38, 48, 48, 62, 44, 30]})

# from sklearn import preprocessing
# d_types=data_set.dtypes
# for i in range(data_set.shape[1]-1):
#     if d_types[i]=='object':
#         Pr_data = preprocessing.LabelEncoder()
#         data_set[data_set.columns[i]]=Pr_data.fit_transform(data_set[data_set.columns[i]])
#         print(data_set.columns[i],": ",Pr_data.classes_)

# X = data_set[data_set.columns[:-1]]
# Y = data_set[data_set.columns[-1]]



# from sklearn.ensemble import RandomForestClassifier as RF
# classifier= RF(n_estimators= 5, criterion="entropy") 
# classifier.fit(X.values, Y.values)
# X_pred = [[1, 0, 0, 1]]
# Y_pred = classifier.predict(X_pred)
# print(Y_pred)

# from sklearn import tree
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7, 10))
# tree.plot_tree(classifier.estimators_[0], feature_names=X.columns)
# plt.show()
# tree.plot_tree(classifier.estimators_[1], feature_names=X.columns)
# plt.show()
# tree.plot_tree(classifier.estimators_[2], feature_names=X.columns)
# plt.show()
# tree.plot_tree(classifier.estimators_[3], feature_names=X.columns)
# plt.show()
# tree.plot_tree(classifier.estimators_[4], feature_names=X.columns)
# plt.show()

# X = [[24], [26], [27], [23], [11], [14], [13], [30], [10], [31]]
# Y = [54, 65, 66, 47, 36, 94, 88, 66, 50, 78]

# from sklearn.tree import DecisionTreeRegressor
# reg = DecisionTreeRegressor()
# reg.fit(X, Y)
# print(reg.predict([[20]]))
#------------------------------------------------------------

# Lec-6
# import numpy as np
# my_input=np.array([
# [1,1],
# [1.5,2],
# [3,4],
# [5,7],
# [3.5,5],
# [4.5,5],
# [3.5,4.5]])

# from sklearn.cluster import KMeans
# my_model=KMeans(n_clusters=2, n_init='auto')
# my_model.fit(my_input)
    
# print(my_model.cluster_centers_)

# print(my_model.labels_)

# print(my_model.predict([[5.7,1.3]]))

# print(KMeans.transform(my_model,[[5.7,1.3]]))

# import pandas as pd
# data_set = pd.read_csv("https://raw.githubusercontent.com/tirthajyoti/MachineLearning-with-Python/master/Datasets/Mall_Customers.csv")

# X = data_set.iloc[:, [3, 4]].values

# from sklearn.cluster import KMeans
# wcss_list = [] 
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, n_init='auto')
#     kmeans.fit(X)
#     wcss_list.append(kmeans.inertia_)

# import matplotlib.pyplot as mtp
# mtp.plot(range(1, 11), wcss_list)
# mtp.title('The Elobw Method Graph')
# mtp.xlabel('Number of clusters(k)')
# mtp.ylabel('wcss_list')
# mtp.show()

# import pandas as pd
# data_set = pd.read_csv("https://raw.githubusercontent.com/tirthajyoti/MachineLearning-with-Python/master/Datasets/Mall_Customers.csv")

# X = data_set.iloc[:, [3, 4]].values

# import scipy.cluster.hierarchy as shc
# dendro = shc.dendrogram(shc.linkage(X, method="average"))

# import matplotlib.pyplot as mtp
# mtp.title("Dendrogrma Plot")
# mtp.ylabel("Euclidean Distances")
# mtp.xlabel("Customers")
# mtp.show()

# import pandas as pd
# data_set = pd.read_csv("https://raw.githubusercontent.com/tirthajyoti/MachineLearning-with-Python/master/Datasets/Mall_Customers.csv")

# X = data_set.iloc[:, [3, 4]].values

# from sklearn.cluster import AgglomerativeClustering as AC
# hc = AC(n_clusters=7, metric='euclidean', linkage='average')
# yp = hc.fit_predict(X)
# print(yp)

# import matplotlib.pyplot as mtp
# yp = hc.fit_predict(X)
# Co=['blue','green','red','cyan','magenta','yellow','black']
# Cl=['Cluster0','Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6']
# for i in range(7):
#     mtp.scatter(X[yp == i, 0], X[yp == i, 1], s = 100, c = Co[i], label = Cl[i])

# mtp.title('Clusters of customers')
# mtp.xlabel('Annual Income (k$)')
# mtp.ylabel('Spending Score (1-100)')
# mtp.legend()
# mtp.show()
#------------------------------------------------------------

# Lec-7
# import numpy
# transactions = numpy.array([
# ['M','O','N','K','E','Y'],
# ['D','O','N','K','E','Y'],
# ['M','A','K','E'],
# ['M','U','C','K','Y'],
# ['C','O','O','K','E']])

# from apyori import apriori
# rules=apriori(transactions,min_support=0.6,min_confidence=0.8)
# results= list(rules)

# for item in results:
#     pair = item[0]
#     items = [x for x in pair]
#     if len(items)>=3:
#         print("Rule: ", items[0:])
#         print("Support: " , str(item[1]))
#         print("Confidence: " , str(item[2][0][2]))
#         print("Lift: " , str(item[2][0][3]))
#         print("=====================================")

# rules = apriori(transactions = transactions, min_support=0.003, min_confidence = 0.2, min_lift=3, min_length=2, max_length=3)   

# import numpy as np
# X=np.array([
# [1,2],
# [5,6],
# [8,9]])
# Y1 = np.cov(X)
# Y2 = np.cov(X.T)
# print("Y1 = \n", Y1)
# print("Y2 = \n", Y2)

# import pandas
# X = pandas.DataFrame({
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
# print(X.shape)

# from sklearn.decomposition import PCA
# my_model=PCA(n_components=4)
# my_model.fit(X)
# X_PCA=my_model.transform(X)
# print(X_PCA.shape)

# print(X_PCA)

# import pandas
# X = pandas.DataFrame({
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
# print(X.shape)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=None)
# X_pca = pca.fit_transform(X)
# explained_variance_ratio=pca.explained_variance_ratio_
# import numpy as np
# cumulative_variance = np.cumsum(explained_variance_ratio)
# optimal_components = np.where(cumulative_variance > 0.95)
# optimal_components = np.add(optimal_components, 1)
# print('Optimal number of components:', optimal_components)
# pca = PCA(n_components=optimal_components[0][0])
# Xpca = pca.fit_transform(X)
# print(Xpca.shape)

# import numpy as np
# dataset =np.array([
# [14.23, 1.71, 2.43, 15.6, 127, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065],
# [13.20, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050],
# [13.16, 2.36, 2.67, 18.6, 101, 2.80, 3.24, 0.30, 2.81, 5.68, 1.03, 3.17, 1185],
# [14.37, 1.95, 2.50, 16.8, 113, 3.85, 3.49, 0.24, 2.18, 7.80, 0.86, 3.45, 1480],
# [13.24, 2.59, 2.87, 21.0, 118, 2.80, 2.69, 0.39, 1.82, 4.32, 1.04, 2.93, 735],
# [13.71, 5.65, 2.45, 20.5, 95, 1.68, 0.61, 0.52, 1.06, 7.70, 0.64, 1.74, 740],
# [13.40, 3.91, 2.48, 23.0, 102, 1.80, 0.75, 0.43, 1.41, 7.30, 0.70, 1.56, 750],
# [13.27, 4.28, 2.26, 20.0, 120, 1.59, 0.69, 0.43, 1.35, 10.20, 0.59, 1.56, 835],
# [13.17, 2.59, 2.37, 20.0, 120, 1.65, 0.68, 0.53, 1.46, 9.30, 0.60, 1.62, 840],
# [14.13, 4.10, 2.74, 24.5, 96, 2.05, 0.76, 0.56, 1.35, 9.20, 0.61, 1.60, 560]])
# X = dataset[:, 0:12]
# Y = dataset[:, 12]

# from sklearn.decomposition import PCA
# pca = PCA(n_components = 1)
# Xr = pca.fit_transform(X)

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(Xr, Y)

# Xnew = np.array([[13.43, 1.17, 2.34, 16.51, 120, 
# 3.10, 3.16, 0.24, 3.21, 6.41, 2.41, 4.19]])
# Xc = pca.transform(Xnew)
# Yp = model.predict(Xc)
# print(Yp)

