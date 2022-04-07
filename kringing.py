import numpy as np
import pykrige.kriging_tools as kt
from pykrige import UniversalKriging
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import db_calls as db
from utils import kriging_interpolation
from geopandas import GeoDataFrame

data = db.get_x_y_value_from_pollution('NO2', '07-07-2015')
points = db.get_x_y_cusec_from_censal()


values = kriging_interpolation(data, points)
points['values'] = np.array(values)

db.update_table_to_geohealth(points, 'temp_table')
points = db.get_table('select * from temp_table', 'GeoHealth')
points.plot(column='values', legend=True)
db.drop_table('temp_table', 'GeoHealth')
plt.show()

