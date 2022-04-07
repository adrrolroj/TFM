import db_calls as db
import matplotlib.pyplot as plt
import clustering as cluster

from analysis import profiling_centers
from utils import normalize_dataframe

# En este fichero realizamos todas las segmentaciones y las almacenamos en la base de datos
# Cambiando el method y las los lineas siguienes con su respectivo metodo podemos sacar la clusterizacion

colums_poblation_all = ['pob1', 'pob2', 'pob3', 'pob4', 'pob5', 'pob6', 'pob7', 'pob8', 'edu1', 'edu2', 'edu3']
columns_life_condition_all = ['apa1', 'apa2','apa3', 'apa4', 'hom1', 'hom2', 'hom3', 'lif1', 'lif2', 'lif3']

colums_poblation = ['pob2', 'pob3', 'pob5', 'pob7', 'pob8', 'edu1', 'edu2', 'edu3']
columns_life_condition = ['apa1', 'apa2', 'apa3', 'hom1', 'hom3', 'lif1', 'lif2', 'lif3']

df_g = db.get_table_censal()
df_norm_po = normalize_dataframe(df_g, colums_poblation)
df_norm_li = normalize_dataframe(df_g, columns_life_condition)

method = 'aglomerative'
df_g, centers_po = cluster.aglomerative_clustering_dataframe(df_g, df_norm_po, colums_poblation, 3, 'poblation')
df_g, centers_li = cluster.aglomerative_clustering_dataframe(df_g, df_norm_li, columns_life_condition, 3, 'life_condition')

plt.figure(figsize=(100, 70))
df_g.plot(column='poblation')
plt.savefig(f'exports/clustering/{method}_poblation_map_clusters.png')
plt.show()
plt.figure(figsize=(100, 70))
df_g.plot(column='life_condition')
plt.savefig(f'exports/clustering/{method}_life_condition_map_clusters.png')
plt.show()
profiling_centers(centers_po, df_norm_po, colums_poblation, f'{method}_poblation')
profiling_centers(centers_li, df_norm_li, columns_life_condition, f'{method}_life_condition')
db.update_table(df_g, f'ana_{method}_analysis', 'GeoHealth')
