from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from sqlalchemy import create_engine
import psycopg2
import os
import time
import datetime
from datetime import date, timedelta,datetime
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Función para crear el rango de fechas:
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

# Función principal:
def get_pollen(year,month,day):

    # Chequea si los parametros son correctos y crea las variables para modificar el fichero script.js
    if len(str(month)) == 1:
        month = '0'+str(month)
    if len(str(day)) == 1:
        day = '0' + str(day)
    loop_year_start = str(year)
    loop_month_start = str(month)
    loop_day_start = str(day)
    loop_year_end = str(year)
    loop_month_end = str(month)
    loop_day_end = str(day)

    data_def = pd.DataFrame()

    # Bucle para cada estacion de captacion:
    for loop_hospital in ['4', '22', '23', '30', '41', '47', '58', '60', '62', '65']:

        # Abre el fichero script.js:
        with open('pollen/script.js') as file:
            lines = file.readlines()

        # Modifical el fichero script.js con los nuevos parametros
        with open('pollen/script.js', 'w') as f:
            year_start = 'var year_start = \''+loop_year_start+'\';\n'
            month_start = 'var month_start = \''+loop_month_start+'\';\n'
            day_start = 'var day_start = \''+loop_day_start+'\';\n'
            year_end = 'var year_end = \''+loop_year_end+'\';\n'
            month_end = 'var month_end = \''+loop_month_end+'\';\n'
            day_end = 'var day_end = \''+loop_day_end+'\';\n'
            hospital = 'var hospital = \''+loop_hospital+'\';\n'
            for line in lines:
                if line.startswith('var year_start'):
                    line = '{}'.format(year_start)
                elif line.startswith('var month_start'):
                    line = '{}'.format(month_start)
                elif line.startswith('var day_start'):
                    line = '{}'.format(day_start)
                elif line.startswith('var year_end'):
                    line = '{}'.format(year_end)
                elif line.startswith('var month_end'):
                    line = '{}'.format(month_end)
                elif line.startswith('var day_end'):
                    line = '{}'.format(day_end)
                elif line.startswith('var hospital'):
                    line = '{}'.format(hospital)
                f.write(line)

        # Ruta de descarga de los ficheros en el servidor.
        # download_path = '/home/victor/Descargas'
        download_path = './downloads'

        # Necesaria la creacion de un webdriver de selenium para que el servidor pueda abrir una sesion de chrome y
        # utilizar el webdocket:
        try:
            driver = webdriver.Remote(
                command_executor='http://selenium-hub-etl:4444/wd/hub',
                desired_capabilities=DesiredCapabilities.CHROME,
            )

            driver.implicitly_wait(30)
            # Abrimos nuestro html donde se ejecuta el websocket
            driver.get('file://'+'/code/websockets-polenes/dist/index.html')
            print(driver)
            cont = 0
            data = None
            # Intentamos 40 veces lo que implica 200 segundos, para extraer el fichero, si no asumimos error o falta de datos.
            while cont < 40:
                file = loop_hospital+'_'+loop_year_start+loop_month_start+loop_day_start+'_'+loop_year_end+loop_month_end+loop_day_end+'.txt'
                if file in os.listdir(download_path):
                    # Si existe, lo abrimos
                    downloaded = open(download_path+'/'+loop_hospital+'_'+loop_year_start+loop_month_start+loop_day_start+'_'+loop_year_end+loop_month_end+loop_day_end+'.txt', 'r')
                    downloaded = [line.split(',') for line in downloaded.readlines()]
                    # Dada la estructura del fichero, vamos eliminando linea a linea los elementos no deseados y
                    # los introducimos en un dataframe.
                    for line in downloaded:
                        for elem in line:
                            if 'added' in elem:
                                data = line
                                data = [element.replace('\\',"").replace('"',"") .replace("\'","")  for element in data]
                                data = data[3:]
                                data[0] = data[0].split('{')[1]
                                data[-1] = data[-1].split('}')[0]
                                data = pd.DataFrame({element.split(':')[0]:element.split(':')[1] for element in data}, columns=[element.split(':')[0] for element in data],index = [0])
                                data = data[['idEstacion','fecha']+list(data.columns[:-2])]
                    break
                else:
                    time.sleep(5)
                    cont += 1

            if isinstance(data, pd.DataFrame):
                # Si existe lo introducimos en la tabla final para poder pasar a la siguiente estacion.
                data_def = pd.concat([data_def,data],axis = 0).reset_index(drop=True)
                # Eliminamos el fichero para no ocupar memoria.
                os.remove(download_path+'/'+loop_hospital+'_'+loop_year_start+loop_month_start+loop_day_start+'_'+loop_year_end+loop_month_end+loop_day_end+'.txt')
                driver.quit()
            else:
                os.remove(download_path + '/' + loop_hospital + '_' + loop_year_start + loop_month_start + loop_day_start + '_' + loop_year_end + loop_month_end + loop_day_end + '.txt')
                driver.quit()

        except ValueError as v:
            print(v)
            print('Error en la conexión')
            driver.quit()

    # Si los datos han sido creados, generamos una tabla con las coordenadas de cada estación.
    if data_def.shape[0] > 0:

        info = {'id': ['4', '22', '23', '30', '41', '47', '58', '60', '62', '65'],
                'name': ['Almeria', 'Jaen', 'Jaen-Hospital', 'Malaga', 'Sevilla-Macarena', 'Sevilla-Tomillar', 'Cadiz','Huelva', 'Cordoba', 'Granada'],
                'province': ['Almeria', 'Jaen', 'Jaen', 'Malaga', 'Sevilla', 'Sevilla', 'Cadiz', 'Huelva', 'Cordoba','Granada'],
                'ccaa': ['Andalucia', 'Andalucia', 'Andalucia', 'Andalucia', 'Andalucia', 'Andalucia', 'Andalucia','Andalucia', 'Andalucia', 'Andalucia'],
                'geometry': [Point(-2.4596435389546327, 36.847114768612194),
                             Point(-3.7798110193798204, 37.79649750505875),
                             Point(-3.7942113706324627, 37.77617996718808),
                             Point(-4.431633598417232, 36.72698443725751),
                             Point(-5.987224074598175, 37.40629267557687),
                             Point(-5.901279099222474, 37.2985901612703),
                             Point(-6.277720859801188, 36.508570296898256),
                             Point(-6.926413589932919, 37.27899689683054),
                             Point(-4.798242228848024, 37.86846639203616),
                             Point(-3.6044834663997625, 37.18624002257025)]}

        stations = gpd.GeoDataFrame(info).set_crs(epsg=4326)
        data_def.rename(columns={'idEstacion': 'id', 'fecha': 'date', '1': 'Alnus', '2': 'Alternaria', '3': 'Artemisa','4': 'Betula',
                                 '5': 'Carex', '6': 'Castanea', '7': 'Cupresaceas', '8': 'Fraxinus', '9': 'Gramineas','10': 'Mercurialis',
                                 '11': 'Morus', '12': 'Olea', '13': 'Palmaceas', '14': 'Pinus', '15': 'Plantago','16': 'Platanus', '17': 'Populus',
                                 '18': 'Amarantaceas', '19': 'Quercus', '20': 'Rumex', '21': 'Ulmus','22': 'Urticaceas'}, inplace=True)
        data_def['date'] = pd.to_datetime(data_def['date'], format='%Y%m%d')

        alchemyEngine = create_engine('postgresql://docker:docker@10.232.5.113:25434/Open_Data_v2')

        # Lo cargamos en la BBDD
        try:
            con = psycopg2.connect(database='Open_Data_v2', user="docker", password="docker", host="10.232.5.113",port="25434")
            sql = "SELECT id from stations_pollen"
            sql_df = pd.read_sql(sql=sql, con=con)
            con.close()
            # sql_df = pd.DataFrame(columns=['id','name','province','ccaa','geometry'])
            # Si ya existen todas las estaciones no hacemos nada
            if stations.shape[0] == len(sql_df['id']):
                print('Stations Exists...')
            elif stations.shape[0] != len(sql_df['id']):
                print('Stations does not match. Appending new...')
                stations[~stations['id'].isin(sql_df['id'])].to_postgis(
                    con=alchemyEngine,
                    name="stations_pollen",
                    index=False,
                    if_exists='append',
                )

        except ValueError as vx:
            print(vx)
            con.close()
        except Exception as ex:
            print(ex)
            con.close()

        try:
            # Cargamos los datos estacion a estacion para comprobar si los datos para esa estacion y fecha ya existen.
            for name in data_def['id'].unique():

                con = psycopg2.connect(database='Open_Data_v2', user="docker", password="docker", host="10.232.5.113",port="25434")
                sql = "SELECT id,date from data_pollen WHERE date = '" + data_def['date'].min().strftime('%Y-%m-%d') + "' AND id = '" + name + "'"
                sql_df = pd.read_sql(sql=sql, con=con)
                con.close()
                # sql_df=pd.DataFrame()
                if data_def[data_def['id'] == name].shape[0] == sql_df.shape[0]:
                    print('Data Exists...')
                elif sql_df.shape[0] == 0:
                    # print('Loading Data...')
                    data_def[data_def['id'] == name].to_sql(
                        con=alchemyEngine,
                        name="data_pollen",
                        index=False,
                        if_exists='append',
                    )
                elif data_def[data_def['id'] == name].shape[0] != sql_df.shape[0]:
                    print('Data does not match. Replacing...')
                    con = psycopg2.connect(database='Open_Data_v2', user="docker", password="docker", host="10.232.5.113",port="25434")
                    sql = "DELETE FROM data_pollen WHERE date = '" + data_def['date'].min().strftime('%Y-%m-%d') + "' AND id = '" + name + "'"
                    con.cursor().execute(sql)
                    con.commit()
                    con.close()
                    data_def[data_def['id'] == name].to_sql(
                        con=alchemyEngine,
                        name="data_pollen",
                        index=False,
                        if_exists='append',
                    )

        except ValueError as vx:
            print(vx)
            con.close()
        except Exception as ex:
            print(ex)
            con.close()

    return None

if __name__ == "__main__":

    date = datetime.today() - timedelta(days=30)
    get_pollen(date.year,date.month,date.day)
