import os, sys, inspect
import time
import db_calls as db
import matplotlib.pyplot as plt

from neural_model.net_model import netModel, netModel_I
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime
import pandas as pd

from utils import normalize_dataframe_train, transform_dataframe_to_tensor
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)




atribute = 'NO2'
date = '2015-07-07 00:00:00.000000'
date_time_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
INPUT_SIZE = 10

model = netModel_I(input_size=INPUT_SIZE)
model.load_state_dict(torch.load(f'neural_model/models/{atribute}_model_inter.pth'))
model.eval()

points = db.get_center_density_censal(str(date_time_obj))
grid_points = np.zeros([len(points), 3])
cusec_points = []
points = normalize_dataframe_train(points)

for index, point in points.iterrows():
    input_point = transform_dataframe_to_tensor(point)
    with torch.no_grad():
        value = model(input_point).detach().numpy()[0][0]
    cusec_points.append(point['cusec'])
    grid_points[index] = np.array([point['x'], point['y'], value])

cusec_points = np.array(cusec_points)
print('Saving in censal...')
df_grid = pd.DataFrame({'cusec': cusec_points, atribute: grid_points[:, 2]})
db.update_table(df_grid, 'temp_grid', 'GeoHealth', geom=False)
df = db.join_grid_with_censal(atribute)

db.drop_table('temp_grid', 'GeoHealth')
df.plot(column=atribute, legend=True)
plt.show()
db.update_table(df, 'censal_pollution', 'GeoHealth')
