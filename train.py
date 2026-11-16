#step 1
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import(f1_score,accuracy_score,classification_report,confusion_matrix)
#step 2
data=load_breast_cancer()
X=pd.DataFrame(
  data.data,
  columns=data.feature_names
)
y=pd.Series(data.target,name="target")
print("Dataset shape:",X.shape)
print("\n First 5 rows:")
print(X.head())
print("\n Target values:")
print(y.value_counts())
#step 3
X_train,X_test,y_train,y_test=train_test_split(
  X,
  y,
  test_size=0.20
  random_state=42,
  stratify=y
)
print("\n Training samples:",X_train.shape[0])
print("Testing samples:"X_test.shape[0])
#step 4
models= {
  "Logistic Regression":LogisticRegression(
    max_iter=5000
  ),
"KNN":KNeighborsClassifier(),
"Decision Tree":DecisionTreeClassifier(
  random_state=42
),
"Random Forest":RandomForestClassifier(
  random_state=42
),
"SVM":SVC()
}
#step 5
param_grids={
  "Logistic Regression":{
    "model__C":[0.1,1,10],
    "model__solver":["liblinear"]
  },
  "KNN":{
    "model__n_neighbors":[3,5,7,9],
    "model__weights":["uniform","distance"]
  },
  "Decision Tree":{
    "model__max_depth":[None,3,5,10],
    "model__min_samples_split":[2,5,10]
  },
  "Random Forest":{
    "model__n_estimators":[100,200],
    "model__max_depth":[None,5,10],
     "model__min_samples_split":[2,5]
  },
  "SVM":{
    "model__C":[0.1,1,10],
    "model__kernel":["linear","rbf"],
    "model__gamma":["scale","auto"]
  }
}
#step 6

    
  
    
  
  
  
  
