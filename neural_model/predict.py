import os, sys, inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import db_calls as db
import matplotlib.pyplot as plt

from neural_model.net_model import netModel
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime
import pandas as pd

atribute = 'O3'
date = '2016-07-19 00:00:00.000000'

date_time_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
day = date_time_obj.timetuple().tm_yday
year = date_time_obj.year

max_year = 2021
min_year = 2010
INPUT_SIZE = 9


def get_scaler(min_element, max_element):
    scaler = MinMaxScaler()
    scaler.fit([[max_element], [min_element]])
    return scaler


model = netModel(input_size=INPUT_SIZE)
model.load_state_dict(torch.load(f'neural_model/models/{atribute}_model.pth'))
model.eval()

points = db.get_center_density_censal(str(date_time_obj))
grid_points = np.zeros([len(points), 3])
cusec_points = []
df_max_min = db.get_min_max_censal()
scaler_density_neigh = get_scaler(df_max_min['max_dn'][0], df_max_min['min_dn'][0])
scaler_density = get_scaler(df_max_min['max_d'][0], df_max_min['min_d'][0])
scaler_x = get_scaler(df_max_min['max_x'][0], df_max_min['min_x'][0])
scaler_y = get_scaler(df_max_min['max_y'][0], df_max_min['min_y'][0])
scaler_day = get_scaler(366, 1)
scaler_year = get_scaler(max_year, min_year)
scaler_temp = get_scaler(df_max_min['max_temp'][0], df_max_min['min_temp'][0])
scaler_wind = get_scaler(df_max_min['max_wind'][0], 0.0)
scaler_rain = get_scaler(df_max_min['max_rain'][0], 0.0)

day = torch.tensor(scaler_day.transform(np.array(day).reshape(-1, 1)))
year = torch.tensor(scaler_year.transform(np.array(year).reshape(-1, 1)))

for index, point in points.iterrows():
    input_point = list()
    x = torch.tensor(scaler_x.transform(np.array(point['x']).reshape(-1, 1)))
    input_point.append(x.float())  # X
    y = torch.tensor(scaler_y.transform(np.array(point['y']).reshape(-1, 1)))
    input_point.append(y.float())  # Y

    input_point.append(day.float())  # day
    input_point.append(year.float())  # year

    denst_n = torch.tensor(scaler_density.transform(np.array(point['density_neigh']).reshape(-1, 1)))
    input_point.append(denst_n.float())  # density neigh
    denst = torch.tensor(scaler_density.transform(np.array(point['density']).reshape(-1, 1)))
    input_point.append(denst.float())  # density

    temp = np.array(point['temp']).reshape(-1, 1)
    temp[np.isnan(temp)] = 0.0
    temp = torch.tensor(scaler_density.transform(temp))
    wind = np.array(point['wind']).reshape(-1, 1)
    wind[np.isnan(wind)] = 0.0
    wind = torch.tensor(scaler_density.transform(wind))
    rain = np.array(point['rain']).reshape(-1, 1)
    rain[np.isnan(rain)] = 0.0
    rain = torch.tensor(scaler_density.transform(rain))
    input_point.append(temp.float())  # temp
    input_point.append(wind.float())  # wind
    input_point.append(rain.float())  # rain

    with torch.no_grad():
        value = model(input_point).detach().numpy()[0][0]
    cusec_points.append(point['cusec'])
    grid_points[index] = np.array([point['x'], point['y'], value])

cusec_points = np.array(cusec_points)
print('Saving in censal...')
df_grid = pd.DataFrame({'cusec': cusec_points, atribute: grid_points[:, 2]})
db.update_table_to_analysis(df_grid, 'temp_grid', geom=False)
df = db.join_grid_with_censal(atribute)

db.drop_table_analysis('temp_grid')
df.plot(column=atribute, legend=True)
plt.show()
db.update_table_to_analysis(df, 'censal_pollution')
