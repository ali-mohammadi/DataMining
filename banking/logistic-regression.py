import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('banking.csv')
data['education'] = np.where(data['education'] == 'basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education'] == 'basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education'] == 'basic.4y', 'Basic', data['education'])

cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for var in cat_vars:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data_temp = data.join(cat_list)
    data = data_temp
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = data[to_keep]

X = data_final.loc[:, data_final.columns != 'y']
Y = data_final.loc[:, data_final.columns == 'y']
os = SMOTE(random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
columns = X_train.columns

os_data_X, os_data_Y = os.fit_sample(X_train, Y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_Y = pd.DataFrame(data=os_data_Y, columns=['y'])

# log_reg = LogisticRegression()
# rfe = RFE(log_reg, 20)
# rfe = rfe.fit(os_data_X, os_data_Y.values.ravel())

cols = ['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no',
        'default_unknown',
        'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
        'month_mar',
        'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
X = os_data_X[cols]
Y = os_data_Y['y']

# logistic_model = sm.Logit(Y, X)
# result = logistic_model.fit()
# print(result.summary2())

cols = ['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate',
        'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar',
        'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
X = os_data_X[cols]
Y = os_data_Y['y']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

Y_predict = log_reg.predict(X_test)

print(classification_report(Y_test, Y_predict))
