import datetime

import torch
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import stats

from torchsummary import summary
import matplotlib.pyplot as plt
import db_calls as db
from analysis import analyze_dataframe
from clustering import elbow_diagram_KMEANS
from neural_model.dataset import ContaminationDataset, get_train_test_data
from neural_model.net_model import netModel, model_LSTM
from rest_service import localizate, export_direction_csv
import utils as utils
import json

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

'''name = input('Press enter to continue')
data = db.get_geoasma_with_latlon_null()
print('ok')
name = input('Press enter to continue')
data = localizate(data)
#name = input('Press enter to continue')
#db.update_table_GeoAsma(data, 'datos_pacientes_asma_2021')'''

'''data = utils.interpolate_pollution_censal_by_nearest_points('NO2', '07-07-2015')
db.update_table_to_geohealth(data, 'borrar')

data = db.get_table('select * from borrar', 'GeoHealth')
data.plot(column='value', legend=True)
plt.show()

'''
'''input = torch.rand(64, 1, 10)
model = model_LSTM(10, 10, 2)
print(repr(model))
print(model(input))'''

'''columns = ['Alnus', 'Alternaria', 'Artemisa',
               'Betula', 'Carex', 'Castanea', 'Cupresaceas', 'Fraxinus', 'Gramineas',
               'Mercurialis', 'Morus', 'Olea', 'Palmaceas', 'Pinus', 'Plantago', 'Platanus',
               'Populus', 'Amarantaceas', 'Quercus', 'Rumex', 'Ulmus', 'Urticaceas']
data = db.get_table_pollen()
data.index = pd.to_datetime(data.date, utc=True)
start_date = '01-01-2018'
end_date = '01-01-2022'
mask = (data.index > start_date) & (data.index <= end_date)
data = data.loc[mask]
print(data['Artemisa'].values)
data['Artemisa'].plot.line()
plt.xlabel("Time")
plt.title('Evolucion del polen "Artemisa" entre 2018 y 2022')
plt.ylabel("Values")
plt.show()'''


def dataset_to_list_points(dir_dataset):
    """
    Read a txt file with a set of points and return a list of objects Point
    :param dir_dataset:
    """
    points = list()
    for index, row in dir_dataset.iterrows():
        point = []
        for col in row.values:
            point.append(col)
        points.append(point)
    return points

def plot_dendrogram(dataset):
    points = dataset_to_list_points(dataset)

    # Calculate distances between points or groups of points
    Z = linkage(points, metric='euclidean', method='ward')

    plt.title('Dendrogram')
    plt.xlabel('Points')
    plt.ylabel('Euclidean Distance')

    # Generate Dendrogram
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True
    )

    plt.show()

def plot_gap(dataset):
    points = dataset_to_list_points(dataset)

    # Calculate distances between points or groups of points
    Z = linkage(points, metric='euclidean', method='ward')

    # Obtain the last 10 distances between points
    last = Z[-10:, 2]
    num_clustres = np.arange(1, len(last) + 1)

    # Calculate Gap
    gap = np.diff(last, n=2)  # second derivative
    plt.plot(num_clustres[:-2] + 1, gap[::-1], 'ro-', markersize=8, lw=2)
    plt.show()

colums_poblation = ['pob2', 'pob3', 'pob5', 'pob7', 'pob8', 'edu1', 'edu2', 'edu3']
columns_life_condition = ['apa1', 'apa2', 'apa3', 'hom1', 'hom3', 'lif1', 'lif2', 'lif3']


df_g = db.get_table_censal()
df_norm_po = utils.normalize_dataframe(df_g, colums_poblation)
df_norm_li = utils.normalize_dataframe(df_g, columns_life_condition)
plot_dendrogram(df_norm_po[colums_poblation])
plot_gap(df_norm_po[colums_poblation])
