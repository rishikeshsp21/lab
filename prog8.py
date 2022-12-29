import matplotlib.pyplot as plt
import pandas as pd
datset=pd.read_csv("decisionTree_Data.csv")
x=datset.iloc[:,[0,1]].values
y=datset.iloc[:,2].values
y


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

from sklearn.tree import DecisionTreeClassifier, plot_tree
clf=DecisionTreeClassifier()
plt.figure(figsize=(40,20))
clf=clf.fit(X_train,y_train)
plot_tree(clf,filled=True)
plt.title("decision tree training")
plt.show()

from sklearn.tree import DecisionTreeClassifier,plot_tree
clf=DecisionTreeClassifier()
plt.figure(figsize=(40,20))
clf=clf.fit(X_test,y_test)
plot_tree(clf,filled=True)
plt.title("decision tree testing")
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.savefig('confusion.png')
