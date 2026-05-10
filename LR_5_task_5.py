import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
# 1. Завантаження даних 
input_file = r'C:\Users\sofia\Downloads\traffic_data.txt'
if not os.path.exists(input_file):
    print(f"Файл не знайдено: {input_file}")
    exit()
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)
data = np.array(data)
# 2. Кодування текстових ознак у числові 
label_encoder = []
X_encoded = np.empty(data.shape)
for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoder.append(le)
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)
# 3. Розбиття на навчальну та тестову вибірки 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
# 4. Навчання регресора на основі гранично випадкових лісів 
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)
# 5. Обчислення ефективності 
y_pred = regressor.predict(X_test)
print(f"Mean absolute error: {round(mean_absolute_error(y_test, y_pred), 2)}")
# 6. Тестування на конкретному прикладі 
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = np.zeros(len(test_datapoint))
count = 0
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:
        test_datapoint_encoded[i] = int(label_encoder[count].transform([test_datapoint[i]])[0])
        count += 1
predicted_traffic = int(regressor.predict([test_datapoint_encoded])[0])
print(f"Predicted traffic for {test_datapoint}: {predicted_traffic}")
