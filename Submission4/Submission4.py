#data preparation

import pandas as pd
import numpy as np

train_path = "Dataset-Pure/train.csv/train.csv"
predict_path = "Dataset-Pure/predict-case.csv/predict-case.csv"

train = pd.read_csv(train_path,sep="|")
predicts = pd.read_csv(predict_path,sep="|")

from sklearn.impute import SimpleImputer

feature = ["location", "salary_currency", "career_level","experience_level","education_level","employment_type","job_function","job_benefits","company_process_time", "company_industry","company_size"]

simpan = train["salary"].copy()

train=train[feature]
predict= predicts[feature]
tcop = train.copy()
pcop = predict.copy()
# Imputation
my_imputer = SimpleImputer(strategy = "most_frequent")
train = pd.DataFrame(my_imputer.fit_transform(train))
predict = pd.DataFrame(my_imputer.transform(predict))

# Imputation removed column names; put them back
train.columns = tcop.columns
predict.columns = pcop.columns

train["salary"] = simpan

def getnum(st):
    num = [int(s) for s in st.split() if s.isdigit()]
    return num[0]

train["experience_level"].fillna("0 tahun",inplace=True)
train["experience_level"] = train["experience_level"].apply(getnum)

predict["experience_level"].fillna("0 tahun",inplace=True)
predict["experience_level"] = predict["experience_level"].apply(getnum)

def getmany(st):
    a = st.count(";")
    return a

train['job_benefits'].fillna("0",inplace=True)
train['job_benefits'] = train["job_benefits"].apply(getmany)

predict['job_benefits'].fillna("0",inplace=True)
predict['job_benefits'] = predict["job_benefits"].apply(getmany)


def getmany(st):
    a = st.count(",")
    return a

train['education_level'].fillna("0",inplace=True)
train['education_level'] = train["education_level"].apply(getmany)

predict['education_level'].fillna("0",inplace=True)
predict['education_level'] = predict["education_level"].apply(getmany)


def getback(st):
    num = [int(s) for s in st.split() if s.isdigit()]
    return num[-1]

train['company_size'].fillna("0 - 0",inplace=True)
train['company_size'] = train["company_size"].apply(getback)

train['company_process_time'].fillna("0 hari",inplace=True)
train['company_process_time'] = train["company_process_time"].apply(getback)

predict['company_size'].fillna("0 - 0",inplace=True)
predict['company_size'] = predict["company_size"].apply(getback)

predict['company_process_time'].fillna("0 hari",inplace=True)
predict['company_process_time'] = predict["company_process_time"].apply(getback)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories='auto',unknown_value=200,handle_unknown ="use_encoded_value")

s = (train.dtypes == 'object')
object_cols = list(s[s].index)

train[object_cols] = ordinal_encoder.fit_transform(train[object_cols])
predict[object_cols] = ordinal_encoder.transform(predict[object_cols])

train.dropna(axis=0, subset=["salary"],inplace=True)

x_train = train[feature]
y_train = train["salary"]

x_p = predict[feature]

from sklearn.model_selection import train_test_split
x_train, x_p, y_train, y_p = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

import xgboost as xgb
from scipy.stats import uniform, randint

xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state = 0)


params = {
    "learning_rate": [0.05,0.10,0.15,0.2,0.25,0.3,0.5,0.75,1],
    "max_depth" : [3,4,5,6,8,10,12,15],
    "min_child_weight" : [1,3,5,7,9],
    "gamma" : [0,0.1,0.2,0.3,0.4],
    "colsample_bytree" : [0.3,0.4,0.5,0.7,1],
    "n_estimators" : [100,200,300,500,700, 1000]
}

from sklearn.model_selection import RandomizedSearchCV

xgb = RandomizedSearchCV(estimator = xgb_model, param_distributions = params,scoring='neg_mean_squared_error', n_iter = 25, cv = 5, verbose=2, random_state=42, n_jobs = 1)

xgb.fit(x_train, y_train)
print(xgb.best_params_)
predictions=xgb.predict(x_p)
from sklearn import metrics
import numpy as np

print('MAE:', metrics.mean_absolute_error(y_p, predictions))
print('MSE:', metrics.mean_squared_error(y_p, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_p, predictions)))
print("R2 : ",metrics.r2_score(y_p, predictions))

# output = pd.DataFrame({'id': predicts.id,
#                        'salary': predictions})
# output.to_csv('Submission4/submission4.csv', index=False)