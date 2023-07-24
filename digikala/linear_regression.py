import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# products_dataset = pd.read_csv('products.csv')
orders_dataset = pd.read_csv('orders.csv')
comments_dataset = pd.read_csv('comments.csv')
prices_dataset = pd.read_csv('prices.csv')

'''
print(products_dataset.columns.values)
print(orders_dataset.columns.values)
print(comments_dataset.columns.values)
print(prices_dataset.columns.values)

category_encode = LabelEncoder()
products_dataset['category_title_fa'] = category_encode.fit_transform(
    products_dataset['category_title_fa'].astype('str'))

brand_encode = LabelEncoder()
products_dataset['brand_name_en'] = brand_encode.fit_transform(products_dataset['brand_name_en'].astype('str'))
'''

prices_avg = prices_dataset[prices_dataset['active'] == 1].groupby(['product_id'], as_index=False)[
    'selling_price'].mean()

comments_total_avg = comments_dataset.groupby(['product_id'], as_index=False).size().reset_index(name='comments_total')
comments_recommended_avg = comments_dataset[comments_dataset['recommend'] == 'recommended'].groupby(['product_id'],
                                                                                                    as_index=False).size().reset_index(
    name='recommended_total')
comments_not_recommended_avg = comments_dataset[comments_dataset['recommend'] == 'not_recommended'].groupby(
    ['product_id'], as_index=False).size().reset_index(name='not_recommended_total')

comments_dataset = comments_total_avg.merge(comments_recommended_avg, how='outer').merge(comments_not_recommended_avg,
                                                                                         how='outer')
comments_dataset = comments_dataset.fillna(0)

''' Filling missing fields with the most relevant data
comments_dataset_not_null = comments_dataset.dropna()
for index, row in comments_dataset.iterrows():
    if math.isnan(row['recommended_total']):
        nearest_index = abs(comments_dataset_not_null['comments_total'] - row['comments_total']).idxmin()
        comments_dataset.at[index, 'recommended_total'] = comments_dataset.loc[nearest_index, ['recommended_total']]
    if math.isnan(row['not_recommended_total']):
        nearest_index = abs(comments_dataset_not_null['comments_total'] - row['comments_total']).idxmin()
        comments_dataset.at[index, 'not_recommended_total'] = comments_dataset.loc[nearest_index, ['not_recommended_total']]
'''

comments_dataset['recommended_total'] = comments_dataset['recommended_total'] / comments_dataset['comments_total']
comments_dataset['not_recommended_total'] = comments_dataset['not_recommended_total'] / comments_dataset[
    'comments_total']

orders_dataset = orders_dataset.groupby(['ID_Item'], as_index=False)['Quantity_item'].sum()

final_datatset = comments_dataset.merge(prices_avg).merge(orders_dataset, left_on='product_id', right_on='ID_Item')
final_datatset = final_datatset[['comments_total', 'recommended_total', 'not_recommended_total', 'selling_price', 'Quantity_item']]

headers = list(final_datatset.columns.values)
features = headers[:-1]
target = headers[-1]

x_train, x_test, y_train, y_test = train_test_split(final_datatset[features], final_datatset[target], test_size=0.3)

model = LinearRegression()
model.fit(x_train, y_train)
model_score = model.score(x_train, y_train)

print("Accuracy", model_score)
