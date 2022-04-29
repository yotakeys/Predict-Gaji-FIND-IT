#data preparation

import pandas as pd
import numpy as np

train_path = "Dataset-Pure/train.csv/train.csv"
predict_path = "Dataset-Pure/predict-case.csv/predict-case.csv"

train = pd.read_csv(train_path,sep="|")
predict = pd.read_csv(predict_path,sep="|")

feature = ["location", "salary_currency", "career_level","experience_level","education_level","employment_type","job_function","job_benefits","company_process_time", "company_industry","company_size"]

train.dropna(axis=0, subset=["salary"],inplace=True)

x_train = train[feature]
y_train = train["salary"]

x_p = predict[feature]

#data proccessing

def getnum(st):
    num = [int(s) for s in st.split() if s.isdigit()]
    return num[0]

x_train["experience_level"].fillna("0 tahun",inplace=True)
x_train["experience_level"] = x_train["experience_level"].apply(getnum)

x_p["experience_level"].fillna("0 tahun",inplace=True)
x_p["experience_level"] = x_p["experience_level"].apply(getnum)


def getmany(st):
    a = st.count(";")
    return a

x_train['job_benefits'].fillna("0",inplace=True)
x_train['job_benefits'] = x_train["job_benefits"].apply(getmany)

x_p['job_benefits'].fillna("0",inplace=True)
x_p['job_benefits'] = x_p["job_benefits"].apply(getmany)


def getback(st):
    num = [int(s) for s in st.split() if s.isdigit()]
    return num[-1]

x_train['company_size'].fillna("0 - 0",inplace=True)
x_train['company_size'] = x_train["company_size"].apply(getback)

x_p['company_size'].fillna("0 - 0",inplace=True)
x_p['company_size'] = x_p["company_size"].apply(getback)


x_train['company_process_time'].fillna("0 hari",inplace=True)
x_train['company_process_time'] = x_train["company_process_time"].apply(getback)

x_p['company_process_time'].fillna("0 hari",inplace=True)
x_p['company_process_time'] = x_p["company_process_time"].apply(getback)


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories='auto',unknown_value=165,handle_unknown ="use_encoded_value")

s = (x_train.dtypes == 'object')
object_cols = list(s[s].index)

x_train[object_cols] = ordinal_encoder.fit_transform(x_train[object_cols])
x_p[object_cols] = ordinal_encoder.transform(x_p[object_cols])
x_train.fillna(0,inplace=True)
x_p.fillna(0,inplace=True)

# Model

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
rfregressor = RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rfgs=RandomizedSearchCV(estimator = rfregressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rfgs.fit(x_train,y_train)

print(rfgs.best_params_)

salary_preds = rfgs.predict(x_p)

print("Succes")
output = pd.DataFrame({'id': predict.id,
                       'salary': salary_preds})
output.to_csv('Submission3/submission3.csv', index=False)