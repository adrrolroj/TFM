import math
import os, sys, inspect

from sklearn.preprocessing import MinMaxScaler
from torch import manual_seed

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import db_calls as db
import numpy as np
from torch.utils.data import Dataset, random_split


class ContaminationDataset(Dataset):

    def __init__(self, atribute, station_names=None):
        if station_names is None:
            df = db.get_pollution_xy_time_density_value(atribute)
        else:
            df = db.get_pollution_xy_time_density_value(atribute, station_names)
        df_max_min = db.get_min_max_censal()

        # Primera entrada coordenada X
        scaler = MinMaxScaler()
        scaler.fit([[df_max_min['max_x'][0]], [df_max_min['min_x'][0]]])
        self.X = scaler.transform(df['x'].values.reshape(-1, 1))

        # Segunda entrada coordenada Y
        scaler.fit([[df_max_min['max_y'][0]], [df_max_min['min_y'][0]]])
        self.Y = scaler.transform(df['y'].values.reshape(-1, 1))

        # Tercera entrada dia del año
        scaler.fit([[366], [1]])
        self.day = scaler.transform(df['doy'].values.reshape(-1, 1))

        # Cuarta entrada año
        scaler.fit([[max(df['year'])], [min(df['year'])]])
        self.year = scaler.transform(df['year'].values.reshape(-1, 1))

        # Quinta entrada densidad media de poblacion de mis vecinos
        scaler.fit([[df_max_min['max_dn'][0]], [df_max_min['min_dn'][0]]])
        self.density_neigh = scaler.transform(df['density_neigh'].values.reshape(-1, 1))

        # Sexta entrada densidad media de poblacion de mis vecinos
        scaler.fit([[df_max_min['max_d'][0]], [df_max_min['min_d'][0]]])
        self.density = scaler.transform(df['density'].values.reshape(-1, 1))

        # Septima entrada temperatura
        scaler.fit([[df_max_min['max_temp'][0]], [df_max_min['min_temp'][0]]])
        self.temp = scaler.transform(df['temp'].values.reshape(-1, 1))

        # Octava entrada velocidad del viento
        scaler.fit([[df_max_min['max_wind'][0]], [0]])
        self.wind = scaler.transform(df['wind'].values.reshape(-1, 1))

        # Novena entrada precipitaciones
        scaler.fit([[df_max_min['max_rain'][0]], [0]])
        self.rain = scaler.transform(df['rain'].values.reshape(-1, 1))

        # Salida de la contaminacion del gas de entrada
        self.value = df[atribute.lower()]
        del df
        del df_max_min

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = np.array(self.X[idx])
        Y = np.array(self.Y[idx])
        month = np.array(self.day[idx])
        year = np.array(self.year[idx])
        dens_n = np.array(self.density_neigh[idx])
        dens = np.array(self.density[idx])
        temp = np.array(self.temp[idx])
        out = np.array(self.value[idx])
        wind = np.array(self.wind[idx])
        wind[np.isnan(wind)] = 0.0
        rain = np.array(self.rain[idx])
        rain[np.isnan(rain)] = 0.0
        return X, Y, month, year, dens_n, dens, temp, wind, rain, out  # El ultimo elemento siempre debe ser la salida de la red


def get_train_test_data(atribute, percent_test, seed):
    station_names = db.get_station_pollution_names_if_not_empty(atribute)['name'].values
    test_size = math.floor(len(station_names) * percent_test)
    train_size = len(station_names) - test_size
    train_names, test_names = random_split(station_names, [train_size, test_size], generator=manual_seed(seed))
    train_dataset = ContaminationDataset(atribute, train_names)
    test_datasets = {name: ContaminationDataset(atribute, [name, name]) for name in test_names}
    return train_dataset, test_datasets

