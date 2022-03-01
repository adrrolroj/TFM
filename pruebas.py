import datetime

import torch
import pandas as pd
import numpy as np
from scipy.stats import stats

import matplotlib.pyplot as plt
import db_calls as db
from analysis import analyze_dataframe
from neural_model.dataset import ContaminationDataset
from utils import complete_dates_meteo, interpolate_meteo, complete_dates_pollution, interpolate_pollution, \
    delete_outliers_values, delete_outliers_values_pollution

'''dataset = ContaminationDataset('NO2')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

for i, data in enumerate(trainloader, 0):
    out = data[-1]
    data = data[:-1]
    x_train = x_train.float().to('cpu')
    y_train = y_train.float().to('cpu')
    time_train = time_train.float().to('cpu')
    dens_train = dens_train.float().to('cpu')
    out = out.float().to('cpu')'''

'''meteo = db.get_table_meteo()
print(meteo['Average_Wind_Speed'].isnull().values.sum())
meteo = intersect_atributes_null_meteo(meteo)
print(meteo['Average_Wind_Speed'].isnull().values.sum())
print(meteo['Precipitation'].isnull().values.sum())'''

'''df = pd.DataFrame([(0.0, np.nan, -1.0, 1.0),
                   (np.nan, 2.0, np.nan, np.nan),
                   (2.0, 3.0, np.nan, 9.0),
                   (np.nan, 4.0, -4.0, 16.0),
                   (np.nan, 4.0, -4.0, 16.0),
                   (np.nan, 4.0, np.nan, 16.0),
                   (4.0, 4.0, np.nan, 16.0),
                   (np.nan, 4.0, np.nan, 16.0),
                   (np.nan, 4.0, np.nan, 16.0),
                   (np.nan, 4.0, np.nan, 16.0),
                   (np.nan, 4.0, -23.0, 16.0),
                   (np.nan, 4.0, 0.0, 16.0),
                   (np.nan, 4.0, np.nan, 16.0),
                   (np.nan, 4.0, np.nan, 16.0),
                   (100, 4.0, np.nan, 16.0)],
                  columns=list('abcd'))

print(df)
df['a'] = df['a'].interpolate(method='linear', limit_direction='both',limit=2, axis=0)
print(df)'''

table= 'Datos_pacientes_asma_2021'
data = db.get_table_geoasma_1(f'select * from "{table}"')
db.update_table_GeoAsma(data, table, geom=False)
