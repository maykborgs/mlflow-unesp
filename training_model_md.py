# Databricks notebook source
# DBTITLE 1,Imports
# Import das libs
from sklearn import tree
from sklearn import ensemble
import mlflow
from sklearn import model_selection
from sklearn import metrics

#%sql

#select avg(radiant_win) from sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")
df = sdf.toPandas()
sdf.display()

# COMMAND ----------

#df.info(memory_usage='deep')
print(df)

# COMMAND ----------

# DBTITLE 1,Variables Definitions
targetCol = 'radiant_win'
idCol = 'match_id'

featCol = list(set(df.columns.tolist()) - set([targetCol, idCol]))

y = df[targetCol]
X = df[featCol]


# COMMAND ----------

# DBTITLE 1,Split Test & Train


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

print("numero de linhas em Xtrain:", X_train.shape[0]) 
print("numero de linhas em Xtest:", X_test.shape[0] )
print("numero de linhas em ytrain:", y_train.shape[0]) 
print("numero de linhas em ytest:", y_test.shape[0])

# COMMAND ----------



model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Setup Experiment MLFlow
mlflow.set_experiment("/Users/maykon.douglas@unesp.br/dota-unesp-maykon")


# COMMAND ----------

# DBTITLE 1,Experiment Run
with mlflow.start_run():
    
    mlflow.sklearn.autolog()
    
    # model = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.7)
    model = tree.DecisionTreeClassifier()
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)

    print("Acurácia em treino", acc_train)
    
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    acc_test = metrics.accuracy_score(y_test, y_test_pred)

    print("Acurácia em test", acc_test)
    
