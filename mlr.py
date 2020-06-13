import numpy as np
import pandas as pd
import model

""" model : [1, size, rooms, floor, bldg_type, district] """

FEATURES = 6
bldg_type = {'кирпичный': 1,
             'панельный': 2,
             'блочный': 3,
             'монолитный': 4,
             'деревянный': 5}
district = {'Ленинский': 1,
            'Октябрьский': 2,
            'Пролетарский': 3}

X = []
Y = []

df = pd.read_csv('dataset.csv')

for data in df.values:
    x_data = np.copy(data)
    x_data[0] = 1
    x_data[4] = bldg_type[x_data[4]]
    x_data[5] = district[x_data[5]]

    X.append(x_data)
    Y.append(data[0] / 1e6)

X = np.array(X)
Y = np.array(Y)

model = model.Model(X.shape[1], 0.0001)
model.train(X, Y)
print('Цена: ' + '{:,}'.format(int(model.predict([1, 60, 2, 2, bldg_type['кирпичный'], district['Октябрьский']]) * 1e6)))
