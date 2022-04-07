import pandas as pd
import math as math
import db_calls as db
import numpy as np
from utils import kriging_interpolation_method, lineal_interpolation_method, parameters_net_method, \
    interpolation_net_method, methods_combinated
import matplotlib.pyplot as plt
# Aqui realizamos el calculo del indice AQHI basandonos en las interpolaciones seleccionadas para los distintos gases

# Opciones: combinados, interpolacion lineal, kriging, red por parametros, red por puntos
METHOD_NO2 = 'combinados'
METHOD_O3 = 'combinados'
METHOD_PART = 'kriging'
FECHA = '01-25-2018' # MES-DIA-AÑO

gases = ['NO2', 'O3', 'PART']
NO2_data = pd.DataFrame()
O3_data = pd.DataFrame()
PART_data = pd.DataFrame()
for i, gas in enumerate(gases):
    print(f'Calculando para {gas}...')
    if globals()[f'METHOD_{gas}'] == 'interpolacion lineal':
        globals()[f'{gas}_data'] = lineal_interpolation_method(gas, FECHA)

    elif globals()[f'METHOD_{gas}'] == 'kriging':
        globals()[f'{gas}_data'] = kriging_interpolation_method(gas, FECHA)

    elif globals()[f'METHOD_{gas}'] == 'red por parametros':
        globals()[f'{gas}_data'] = parameters_net_method(gas, FECHA)

    elif globals()[f'METHOD_{gas}'] == 'red por puntos':
        globals()[f'{gas}_data'] = interpolation_net_method(gas, FECHA)

    else:
        globals()[f'{gas}_data'] = methods_combinated(gas, FECHA)

    plt.figure(i)
    globals()[f'{gas}_data'].plot(column='values', legend=True)
    plt.title(f'Para {gas}')

censal_data = db.get_table_censal()
values_AQHI = []
level_AQHI = []
print('Calculando indice AQHI')
for index, censal in censal_data.iterrows():
    NO2_value = NO2_data.iloc[NO2_data.index[NO2_data['cusec'] == censal['cusec']]]['values']
    O3_value = O3_data.iloc[O3_data.index[O3_data['cusec'] == censal['cusec']]]['values']
    PART_value = PART_data.iloc[PART_data.index[PART_data['cusec'] == censal['cusec']]]['values']
    AQHI = (1000.0/10.4) * ((math.exp(0.000537 * O3_value) - 1) + (math.exp(0.000871 * NO2_value) - 1) + (math.exp(0.000297 * PART_value) - 1))
    level = 1.0 if AQHI <= 3.0 else 2.0 if AQHI <= 6.0 else 3.0 if AQHI < 10.0 else 4.0

    values_AQHI.append(AQHI)
    level_AQHI.append(level)
censal_data['AQHI'] = np.array(values_AQHI)
censal_data['level'] = np.array(level_AQHI)
censal_data.plot(column='AQHI', vmin=1.0, vmax=10.0, legend=True)
plt.title(f'Indice AQHI')

censal_data.plot(column='level', vmin=1.0, vmax=4.0, legend=True)
plt.title(f'Nivel de riesgo')
plt.show()
