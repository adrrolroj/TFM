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


class ContaminationInterpolationDataset(Dataset):

    def __init__(self, atribute, station_names=None, add_data=False):
        df = db.get_table_gas_train(atribute, station_names)
        scaler = MinMaxScaler()
        self.dist_1 = df[f'1_dist_{atribute}'].values.reshape(-1, 1)
        self.dist_2 = df[f'2_dist_{atribute}'].values.reshape(-1, 1)
        self.dist_3 = df[f'3_dist_{atribute}'].values.reshape(-1, 1)

        scaler.fit([[45000.0], [0.0]])
        self.density_neigh = scaler.transform(df['neigh_density'].values.reshape(-1, 1))

        max_value = [193.5 if atribute == 'NO2' else 337.7 if atribute == 'PART' else 224]
        scaler.fit([[max_value[0]], [0.0]])
        self.gas_1 = scaler.transform(df[f'1_{atribute}'].values.reshape(-1, 1))
        self.gas_2 = scaler.transform(df[f'2_{atribute}'].values.reshape(-1, 1))
        self.gas_3 = scaler.transform(df[f'3_{atribute}'].values.reshape(-1, 1))

        # Variables metteorologicas
        scaler.fit([[24.0], [0.0]])
        self.wind = scaler.transform(df[f'wind'].values.reshape(-1, 1))
        scaler.fit([[296.0], [0.0]])
        self.rain = scaler.transform(df[f'rain'].values.reshape(-1, 1))
        scaler.fit([[1201.0], [0.0]])
        self.altitude = scaler.transform(df[f'altitude'].values.reshape(-1, 1))
        self.value = df[atribute].values.reshape(-1, 1)

        # Vamos añadir datos extra
        num_ejemplos_extra = 40
        if add_data:
            self.value = np.concatenate((self.value, np.array([[24.0] * num_ejemplos_extra, [32.0] * num_ejemplos_extra, [5.0] * num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.gas_1 = np.concatenate((self.gas_1, np.array([[24.0/max_value[0]]*num_ejemplos_extra, [20.0/max_value[0]]*num_ejemplos_extra, [24.0/max_value[0]]*num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.gas_2 = np.concatenate((self.gas_2, np.array([[45.0/max_value[0]]*num_ejemplos_extra, [32.0/max_value[0]]*num_ejemplos_extra, [26.0/max_value[0]]*num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.gas_3 = np.concatenate((self.gas_3, np.array([[8.0/max_value[0]]*num_ejemplos_extra, [5.0/max_value[0]]*num_ejemplos_extra, [5.0/max_value[0]]*num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.dist_1 = np.concatenate((self.dist_1, np.array([[0.0] * num_ejemplos_extra, [0.12] * num_ejemplos_extra, [0.3] * num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.dist_2 = np.concatenate((self.dist_2, np.array([[0.54] * num_ejemplos_extra, [0.0] * num_ejemplos_extra, [0.08] * num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.dist_3 = np.concatenate((self.dist_3, np.array([[0.10] * num_ejemplos_extra, [0.21] * num_ejemplos_extra, [0.0] * num_ejemplos_extra]).reshape(-1, 1)), axis=0)

            self.wind = np.concatenate((self.wind, np.array([[0.08] * num_ejemplos_extra, [0.0] * num_ejemplos_extra, [0.60] * num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.rain = np.concatenate((self.rain, np.array([[0.12] * num_ejemplos_extra, [0.0] * num_ejemplos_extra, [0.60] * num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.altitude = np.concatenate((self.altitude, np.array([[0.30] * num_ejemplos_extra, [0.04] * num_ejemplos_extra, [0.50] * num_ejemplos_extra]).reshape(-1, 1)), axis=0)
            self.density_neigh = np.concatenate((self.density_neigh, np.array([[0.40] * num_ejemplos_extra, [0.50] * num_ejemplos_extra, [0.12] * num_ejemplos_extra]).reshape(-1, 1)), axis=0)
        del df

    def __len__(self):
        return len(self.value)

    def __getitem__(self, idx):
        dist_1 = np.array(self.dist_1[idx])
        dist_2 = np.array(self.dist_2[idx])
        dist_3 = np.array(self.dist_3[idx])
        gas_1 = np.array(self.gas_1[idx])
        gas_2 = np.array(self.gas_1[idx])
        gas_3 = np.array(self.gas_1[idx])
        neigh_density = np.array(self.density_neigh[idx])
        wind = np.array(self.wind[idx])
        wind[np.isnan(wind)] = 0.0
        rain = np.array(self.rain[idx])
        rain[np.isnan(rain)] = 0.0
        altitude = np.array(self.altitude[idx])
        altitude[np.isnan(altitude)] = 0.08
        out = np.array(self.value[idx])
        return dist_1, dist_2, dist_3, gas_1, gas_2, gas_3, neigh_density, wind, rain, altitude, out  # El ultimo elemento siempre debe ser la salida de la red


def get_train_test_data_interpolation(atribute, percent_test, seed):
    station_names = db.get_station_pollution_names_if_not_empty(atribute)['name'].values
    test_size = math.floor(len(station_names) * percent_test)
    train_size = len(station_names) - test_size
    train_names, test_names = random_split(station_names, [train_size, test_size], generator=manual_seed(seed))
    train_dataset = ContaminationInterpolationDataset(atribute, train_names, add_data=True)
    test_datasets = {name: ContaminationInterpolationDataset(atribute, [name, name]) for name in test_names}
    return train_dataset, test_datasets

