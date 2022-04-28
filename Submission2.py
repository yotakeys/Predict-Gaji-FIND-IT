#data preparation
import pandas as pd

train_path = "Dataset-Pure/train.csv/train.csv"
predict_path = "Dataset-Pure/predict-case.csv/predict-case.csv"

train = pd.read_csv(train_path, sep = '|')
predict = pd.read_csv(predict_path, sep = '|')

trainC = train.dropna(axis=0,subset=["salary"])

x_cols_used = ["location","salary_currency","career_level","experience_level","education_level","employment_type","job_function","company_industry","company_size"]

x_train = trainC[x_cols_used]
y_train = trainC["salary"]

x_predict = predict[x_cols_used]

#data proccessing
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories='auto',unknown_value=165,handle_unknown ="use_encoded_value")
#because all column used are object, we dont need :
# s = (x_train.dtypes == 'object')
# object_cols = list(s[s].index)

x_train[x_cols_used] = ordinal_encoder.fit_transform(x_train[x_cols_used])
x_predict[x_cols_used] = ordinal_encoder.transform(x_predict[x_cols_used])

x_train.fillna(0,inplace=True)
x_predict.fillna(0,inplace=True)

#make model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(x_train,y_train)
salary_preds = model.predict(x_predict)

print("Succes")
output = pd.DataFrame({'id': predict.id,
                       'salary': salary_preds})
output.to_csv('submission2.csv', index=False)