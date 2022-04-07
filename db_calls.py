import psycopg2 as psycopg2
import geopandas as gpd
import pandas as pd
from dateutil.relativedelta import relativedelta
from geoalchemy2 import WKTElement
from sqlalchemy import create_engine

### DATABASE ###
## GEOHEALTH ##
database_geohealth = 'GeoHealth'
user_geohealth = 'docker'
pass_geohealth = 'docker'
host_geohealth = 'localhost'
port_geohealth = '25434'
## GEOASMA ##
database_geoasma = 'GeoAsma'
user_geoasma = 'docker'
pass_geoasma = 'docker'
host_geoasma = '10.232.5.113'
port_geoasma = '25434'


def get_table(query, data_base):
    if data_base == 'GeoHealth':
        conection = psycopg2.connect(database=database_geohealth, user=user_geohealth, password=pass_geohealth,
                                     host=host_geohealth, port=port_geohealth)
    else:
        conection = psycopg2.connect(database=database_geoasma, user=user_geoasma, password=pass_geoasma,
                                     host=host_geoasma, port=port_geoasma)
    try:
        data = gpd.GeoDataFrame.from_postgis(query, conection, geom_col='geometry')
        conection.close()
        return data
    except Exception as ex:
        conection.close()
        print('An exception occurred')
        print(ex)


def get_table_without_geometry(query, data_base):
    if data_base == 'GeoHealth':
        engine = create_engine(f'postgresql://{user_geohealth}:{pass_geohealth}@{host_geohealth}:{port_geohealth}'
                               f'/{database_geohealth}', echo=False)
    else:
        engine = create_engine(f'postgresql://{user_geoasma}:{pass_geoasma}@{host_geoasma}:{port_geoasma}'
                               f'/{database_geoasma}', echo=False)
    try:
        data = pd.read_sql_query(query, con=engine)
        return data
    except Exception as ex:
        print('An exception occurred')
        print(ex)


def get_value(query, data_base):
    if data_base == 'GeoHealth':
        conection = psycopg2.connect(database=database_geohealth, user=user_geohealth, password=pass_geohealth,
                                     host=host_geohealth, port=port_geohealth)
    else:
        conection = psycopg2.connect(database=database_geoasma, user=user_geoasma, password=pass_geoasma,
                                     host=host_geoasma, port=port_geoasma)
    try:
        cur = conection.cursor()
        cur.execute(query)
        result = cur.fetchone()[0]
        conection.close()
        return result
    except Exception as ex:
        conection.close()
        print('An exception occurred')
        print(ex)


def get_table_censal():
    query = 'SELECT * FROM censal_full'
    return get_table(query, 'GeoHealth')


def get_table_meteo():
    query = 'SELECT * FROM meteo'
    return get_table(query, 'GeoHealth')


def get_table_pollen():
    query = 'SELECT * FROM pollen'
    return get_table(query, 'GeoHealth')


def get_geometry_by_cusec(cusec):
    query = f'select t.geometry from censal t where t."CUSEC" =\'{cusec}\''
    return get_table(query, 'GeoHealth')


def get_table_pollen_by_id(id):
    id = f"'{id}'"
    query = f'SELECT p.* FROM pollen p where p.id={id}'
    return get_table(query, 'GeoHealth')


def get_table_basics_health_region():
    query = 'SELECT * FROM basics_health_region'
    return get_table(query, 'GeoHealth')


def get_table_health_districts_region():
    query = 'SELECT * FROM health_districts_region'
    return get_table(query, 'GeoHealth')


def get_table_postal():
    query = 'SELECT * FROM postal'
    return get_table(query, 'GeoHealth')


def get_table_neighborhood_region():
    query = 'SELECT * FROM neighborhood_region'
    return get_table(query, 'GeoHealth')


def get_avg_atribute_from_neighbor(geometry, atribute):
    atribute = f'"{atribute}"'
    geometry = f"'{geometry}'"
    query = f'SELECT avg(a.{atribute}) from censal_full as a ' \
            f'where ST_Intersects(a.geometry, ST_GeomFromText({geometry}, 4326))'
    return get_value(query, 'GeoHealth')


################# PARA ENTRENAMIENTO DE RED ###################################
def get_pollution_xy_time_density_value(atribute, station_names=None):
    if station_names is None:
        query = f'select EXTRACT(DOY FROM p."Date") as doy, EXTRACT(YEAR FROM p."Date") as year, ' \
                f'p."{atribute}" as {atribute}, ST_X(p.geometry) as X,ST_Y(p.geometry) as Y, ' \
                f'p.density_neigh as density_neigh , p.density as density, m."Average_Temperature" as temp,' \
                f' m."Average_Wind_Speed" as wind, m."Altitude" as altitude, m."Precipitation" as rain ' \
                f'from  pollution_for_train p ' \
                f'join meteo m on ' \
                f'm.geometry=ST_ClosestPoint((select st_collect(m.geometry) from station_meteo m), p.geometry)' \
                f'where p."{atribute}" is not null and m."Date"=p."Date"'
    else:
        station_names = tuple(station_names)
        query = f'select EXTRACT(DOY FROM p."Date") as doy, EXTRACT(YEAR FROM p."Date") as year, ' \
                f'p."{atribute}" as {atribute}, ST_X(p.geometry) as X,ST_Y(p.geometry) as Y, ' \
                f'p.density_neigh as density_neigh , p.density as density, m."Average_Temperature" as temp,' \
                f' m."Average_Wind_Speed" as wind, m."Altitude" as altitude, m."Precipitation" as rain ' \
                f'from  pollution_for_train p ' \
                f'join meteo m on ' \
                f'm.geometry=ST_ClosestPoint((select st_collect(m.geometry) from station_meteo m), p.geometry)' \
                f'where p."{atribute}" is not null and m."Date"=p."Date" and p."Name" in {station_names}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_input_from(station_name, date):
    station_name = f"'{station_name}'"
    date = f"'{date}'"
    query = f'select EXTRACT(DOY FROM p."Date") as doy, EXTRACT(YEAR FROM p."Date") as year, ST_X(p.geometry) as X,' \
            f'ST_Y(p.geometry) as Y, p.density_neigh as density_neigh , p.density as density, ' \
            f'm."Average_Temperature" as temp, m."Average_Wind_Speed" as wind, m."Altitude" as altitude, ' \
            f'm."Precipitation" as rain from  pollution_for_train p join meteo m on ' \
            f'm.geometry=ST_ClosestPoint((select st_collect(sm.geometry) from meteo sm where sm."Date"={date}), p.geometry) ' \
            f'and m."Date"=p."Date" where p."Name"={station_name} and p."Date"={date}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_input_from_interpol(station_name, date, gas):
    station_name = f"'{station_name}'"
    date = f"'{date}'"
    query = f'select p.*, c.neigh_density from "pollution_{gas}_train" p' \
            f' join station_pollution c on c."Name"=p."Name" where p."Name" = {station_name}' \
            f' and p."Date"={date} '
    return get_table_without_geometry(query, 'GeoHealth')


def get_min_max_censal():
    query = f'select max(c.neigh_density) as max_dn, min(c.neigh_density) as min_dn, ' \
            f'max(c.pob8) as max_d, min(c.pob8) as min_d, ' \
            f'max(ST_XMax(c.geometry)) as max_x, min(ST_XMin(c.geometry)) as min_x,' \
            f' max(ST_YMax(c.geometry)) as max_y, min(ST_YMin(c.geometry)) as min_y from censal_full c'
    query_2 = f'select  max(m."Average_Wind_Speed") as max_wind, max(m."Precipitation") as max_rain, ' \
              f'max(m."Altitude") as max_altitude, ' \
              f'min(m."Average_Temperature") as min_temp, max(m."Average_Temperature") as max_temp from meteo m'
    data = get_table_without_geometry(query, 'GeoHealth')
    data = pd.concat([data, get_table_without_geometry(query_2, 'GeoHealth')], axis=1)
    return data


def get_center_density_censal(date_from):
    date_from = f"'{date_from}'"
    query = f'select ST_X(ST_Centroid(c.geometry)) as X, ST_Y(ST_Centroid(c.geometry)) as Y, c.neigh_density ' \
            f'as density_neigh, EXTRACT(DOY FROM m."Date") as doy, EXTRACT(YEAR FROM m."Date") as year, ' \
            f'c.pob8 as density, m."Average_Wind_Speed" as wind, m."Precipitation" as rain, ' \
            f'm."Average_Temperature" as temp, m."Altitude" as altitude, c.cusec as cusec, c.geometry as geometry ' \
            f'from censal_full c left join meteo m on m.geometry=ST_ClosestPoint((select st_collect(m.geometry) ' \
            f'from station_meteo m), ST_Centroid(c.geometry)) and m."Date"={date_from}'
    return get_table_without_geometry(query, 'GeoHealth')


######################################################################
def get_table_meteo_by_name(name):
    name = f"'{name}'"
    query = f'select m.* from meteo m where m."Name"={name}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_table_pollution_by_name(name):
    name = f"'{name}'"
    query = f'select m.* from pollution m where m."Name"={name}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_table_censal_with_near_pollen_stations_in_date(date):
    date = f"'{date}'"
    query = f'select * from (select *, row_number() over (partition by cusec order by distance) as rank ' \
            f'from (select c.cusec, c.geometry, st.id, st.name, st.province, st.ccaa, st.date, st."Alnus", ' \
            f'st."Alternaria", st."Artemisa", st."Betula", st."Carex", st."Castanea", st."Cupresaceas", ' \
            f'st."Fraxinus", st."Gramineas", st."Mercurialis", st."Morus", st."Olea", st."Palmaceas", ' \
            f'st."Pinus", st."Plantago", st."Platanus", st."Populus", st."Amarantaceas", st."Quercus", ' \
            f'st."Rumex", st."Ulmus", st."Urticaceas", st_distance(st.geometry,ST_Centroid(c.geometry)) ' \
            f'as distance from (select p.* from pollen p where p."date"={date}) as st join censal_full c ' \
            f'on  st_closestpoint(st.geometry,ST_Centroid(c.geometry))=st.geometry) as n) as w where rank <= 3'
    return get_table(query, 'GeoHealth')


def get_xy_density_from_grid_points():
    query = f'select ST_X(p.geometry) as X,ST_Y(p.geometry) as Y, c.neigh_density as density ' \
            f'from  grid_points p join censal_full c on st_within( p.geometry, c.geometry) '
    return get_table_without_geometry(query, 'GeoHealth')


def update_data_for_training():
    '''query = 'select p.*, c.neigh_density as density_neigh, c.pob8 as density from pollution p ' \
            'join censal_full c on ST_Contains(c.geometry, p.geometry)'
    data = get_table(query, 'GeoHealth')
    update_table_to_geohealth(data, 'pollution_for_train')'''
    query = 'select p.*, m."Average_Temperature" as temp, m."Average_Wind_Speed" as wind, ' \
            'm."Precipitation" as rain, m."Altitude" as altitude ' \
            'from pollution_for_train p join meteo m ' \
            'on m.geometry=ST_ClosestPoint((select st_collect(sm.geometry) from station_meteo sm), ' \
            'p.geometry) and m."Date"=p."Date"'
    data = get_table(query, 'GeoHealth')
    update_table_to_geohealth(data, 'pollution_for_comparation')


def get_table_gas_train(gas, station_names):
    station_names = tuple(station_names)
    query = f'select p.*, c.neigh_density from "pollution_{gas}_train" p join station_pollution c ' \
            f'on c."Name"=p."Name" where p."Name" in {station_names}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_table_pollution_in_stations(name_stations, gas):
    gas = f'"{gas}"'
    name_stations = tuple(name_stations)
    query = f'select p.*, ST_X(p.geometry) as X, ST_Y(p.geometry) as Y, EXTRACT(DOY FROM p."Date") as doy, ' \
            f'EXTRACT(YEAR FROM p."Date") as year from pollution_for_comparation p where ' \
            f'p.{gas} is not null and p."Name" in {name_stations}'
    return get_table(query, 'GeoHealth')


def get_geo_val_from_pollution_in_not_stations(gas, date, stations):
    gas = f'"{gas}"'
    date = f"'{date}'"
    stations = tuple(stations)
    query = f'select ST_X(p.geometry) as x, ST_Y(p.geometry) as y, p.{gas} as value from pollution p ' \
            f'where p."Date"={date} and p.{gas} is not null and p."Name" not in {stations}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_avg_atribute(atribute):
    atribute = f'"{atribute}"'
    query = f'SELECT avg(a.{atribute}) from censal_full as a'
    return get_value(query, 'GeoHealth')


def get_table_pollution():
    query = 'SELECT * FROM pollution'
    return get_table(query, 'GeoHealth')


def get_x_y_value_from_pollution(gas, date):
    gas = f'"{gas}"'
    date = f"'{date}'"
    query = f'select st_x(p.geometry) as x, st_y(p.geometry) as y, p.{gas} as value ' \
            f'from pollution p where p."Date"={date} and p.{gas} is not null'
    return get_table_without_geometry(query, 'GeoHealth')


def get_x_y_value_from_pollen(date):
    date = f"'{date}'"
    query = f'select st_x(p.geometry) as x, st_y(p.geometry) as y, p.* ' \
            f'from pollen p where p."date"={date}'
    return get_table(query, 'GeoHealth')


def get_voronoi_censal_pollen(date):
    date = f"'{date}'"
    query = f'select c.*, st.date, st."Alnus", st."Alternaria", st."Artemisa", ' \
            f'st."Betula", st."Carex", st."Castanea", st."Cupresaceas", st."Fraxinus", ' \
            f'st."Gramineas", st."Mercurialis", st."Morus", st."Olea", st."Palmaceas", ' \
            f'st."Pinus", st."Plantago", st."Platanus", st."Populus", st."Amarantaceas", ' \
            f'st."Quercus", st."Rumex", st."Ulmus", st."Urticaceas" from censal_full c ' \
            f'left join pollen st on st.geometry=st_closestpoint((select st_collect(s.geometry) ' \
            f'from pollen s where s.date={date}), st_centroid(c.geometry)) and st.date={date}'
    return get_table(query, 'GeoHealth')


def get_x_y_cusec_from_censal():
    query = 'select st_x(ST_Centroid(p.geometry)) as x, ' \
            'st_y(ST_Centroid(p.geometry)) as y, p."cusec", p.geometry from censal_full p'
    return get_table_without_geometry(query, 'GeoHealth')


def get_table_pollution_from_dates(date_from, date_to):
    query = f'SELECT p."Name", p."Province", p."Municipality",' \
            f'avg(p."SH2") as "SH2", avg(p."SO2") as "SO2", ' \
            f'avg(p."NO2") as "NO2", avg(p."CO") as "CO", avg(p."PART") as "PART", ' \
            f'avg(p."O3") as "O3", p.geometry from pollution p' \
            f' where (p."Date" BETWEEN \'{date_from}\' AND \'{date_to}\') ' \
            f'GROUP BY p."Name", p."Province", p."Municipality", p.geometry'
    return get_table(query, 'GeoHealth')


def get_table_pollution_by_name_date_gas(station_name, date, gas):
    gas = f'"{gas}"'
    date = f"'{date}'"
    station_name = f"'{station_name}'"
    query = f'select p.* from pollution p where p."Date"={date} and p.{gas} is not null and p."Name"={station_name}'
    return get_table(query, 'GeoHealth')


def get_mean_meteo_from_date(atribute, name, date_from, date_to):
    name = f"'{name}'"
    date_from = f"'{date_from}'"
    date_to = f"'{date_to}'"
    query = f'select avg(m."{atribute}") from meteo m where m."Name"={name} ' \
            f'and (m."Date"={date_from} or m."Date"={date_to})'
    return get_value(query, 'GeoHealth')


def compare_cluster_from_analysis(cluster1, cluster2, segmentation):
    query = f'select a.{segmentation} as first, b.{segmentation} as second,' \
            f' count(a.{segmentation}) from {cluster1} a join {cluster2} b ' \
            f'on a.geometry = b.geometry group by a.{segmentation}, b.{segmentation}'
    return get_table_without_geometry(query, 'GeoHealth')


def get_station_pollution_names():
    query = 'select m."Name" as name from pollution m group by m."Name"'
    return get_table_without_geometry(query, 'GeoHealth')


def get_station_pollution_names_if_not_empty(atribute):
    query = f'select m."Name" as name from pollution m where m."{atribute}" is not null group by m."Name"'
    return get_table_without_geometry(query, 'GeoHealth')


def get_station_pollution():
    query = 'select p."Name", p.geometry from pollution p group by p."Name", p.geometry'
    return get_table(query, 'GeoHealth')


def get_censal_pollution_nearest_points(atribute, date):
    atribute = f'"{atribute}"'
    date = f"'{date}'"
    query = f'select * from (select *, row_number() over (partition by cusec order by distance) as rank ' \
            f'from (select c.cusec, c.geometry, st.name, st.{atribute}, st_distance(st.geometry,ST_Centroid(c.geometry)) as distance ' \
            f'from (select p."Name" as name, p.geometry, p.{atribute} from pollution p where p.{atribute} is not null ' \
            f'and p."Date"={date} group by p."Name", p.geometry, p.{atribute}) st join censal_full c ' \
            f'on  st_closestpoint(st.geometry,ST_Centroid(c.geometry))=st.geometry) as n) as w where rank<=3'
    return get_table(query, 'GeoHealth')


def get_censal_for_net_interpolation(gas, date):
    gas = f'"{gas}"'
    date = f"'{date}'"
    query = f'select rank_censal.*, m."Average_Wind_Speed" as wind, m."Precipitation" as rain, m."Altitude" as altitude ' \
            f'from (select * from (select *, row_number() over (partition by cusec order by distance) as rank ' \
            f'from (select c.cusec, c.geometry, c.neigh_density, st.name, st.{gas}, st_distance(st.geometry,ST_Centroid(c.geometry)) as distance ' \
            f'from (select p."Name" as name, p.geometry, p.{gas} from pollution p where p.{gas} is not null ' \
            f'and p."Date"={date} group by p."Name", p.geometry, p.{gas}) st join censal_full c ' \
            f'on  st_closestpoint(st.geometry,ST_Centroid(c.geometry))=st.geometry) as n) as w where rank<=3) ' \
            f'as rank_censal left join meteo m on m.geometry=ST_ClosestPoint((select st_collect(m.geometry) ' \
            f'from station_meteo m), ST_Centroid(rank_censal.geometry)) and m."Date"={date}'
    return get_table(query, 'GeoHealth')


def get_pollution_interpolated_in_station(station_name, date, gas):
    station_name = f"'{station_name}'"
    date = f"'{date}'"
    gas = f'"{gas}"'
    query = f'select s.*, st_distance(s.geometry, (select st.geometry from station_pollution st ' \
            f'where st."Name"={station_name})) as distance from pollution s where s."Name"!={station_name} ' \
            f'and s."Date"={date} and s.{gas} is not null order by distance limit 3'
    return get_table(query, 'GeoHealth')


def make_grid(meters):  # Crea un cuadrado de puntos en andalucia con una distancia dada
    conection = psycopg2.connect(database=database_geohealth, user=user_geohealth, password=pass_geohealth,
                                 host=host_geohealth, port=port_geohealth)
    query = f'select st_transform(makegrid(st_transform(ST_Union(ST_SnapToGrid(a.geometry,0.000001)), 2062),' \
            f' {meters}), 4326) as geometry from health_districts_region a'
    try:
        data = gpd.GeoDataFrame.from_postgis(query, conection, geom_col='geometry')
        data = data.explode(ignore_index=True)
        conection.close()
        return data
    except Exception as ex:
        conection.close()
        print('An exception occurred')
        print(ex)


def get_pollution_intersect(date, name, atribute):
    atribute = f'"{atribute}"'
    date_from = date - relativedelta(months=1)
    date_to = date + relativedelta(months=1)

    query = f'SELECT avg(p.{atribute}) from pollution_in_month p' \
            f' where (p."Date" BETWEEN \'{date_from}\' AND \'{date_to}\') and p."Name" = \'{name}\''
    return get_value(query, 'GeoHealth')


def get_station_pollution_by_name(station_name):
    station_name = f"'{station_name}'"
    query = f'select * from station_pollution st where st."Name"={station_name}'
    return get_table(query, 'GeoHealth')


def get_value_pollution_station_date(station_name, date, gas):
    station_name = f"'{station_name}'"
    date = f"'{date}'"
    gas = f'"{gas}"'
    query = f'select p.{gas} from pollution p where p."Name"={station_name} and p."Date"={date}'
    return get_value(query, 'GeoHealth')


def get_table_analysis_by_CUSEC(table, cusec):
    query = f'SELECT * FROM {table} t where t."cusec" =\'{cusec}\''
    return get_table(query, 'GeoHealth')


def join_grid_with_censal(atribute):
    query = f'select c.*, t."{atribute}" from censal_full c join temp_grid t on t.cusec=c.cusec'
    return get_table(query, 'GeoHealth')


def update_table_to_geohealth(data, name_table, geom=True):
    update_table(data, name_table, 'GeoHealth', geom)


def update_table(data, name_table, data_base, geom=True):
    if data_base == 'GeoHealth':
        engine = create_engine(f'postgresql://{user_geohealth}:{pass_geohealth}@{host_geohealth}:{port_geohealth}'
                               f'/{database_geohealth}', echo=False)
    else:
        engine = create_engine(f'postgresql://{user_geoasma}:{pass_geoasma}@{host_geoasma}:{port_geoasma}'
                               f'/{database_geoasma}', echo=False)
    if geom:
        data['geometry'] = data['geometry'].apply(
            lambda geo: str(WKTElement(geo.wkt, srid=4326)) if not type(geo) == str else geo)
    try:
        data.to_sql(name_table, engine, if_exists='replace', index=False)
        if geom:
            with engine.connect() as conn, conn.begin():
                sql = f'ALTER TABLE {name_table} ALTER COLUMN geometry TYPE Geometry ' \
                      f'USING ST_SetSRID(geometry::Geometry, 4326)'
                conn.execute(sql)
        print(f'Table {name_table} updated succesfully!')
    except Exception as ex:
        print('An exception occurred')
        print(ex)


def drop_table(name_table, data_base):
    if data_base == 'GeoHealth':
        conection = psycopg2.connect(database=database_geohealth, user=user_geohealth, password=pass_geohealth,
                                     host=host_geohealth, port=port_geohealth)
    else:
        conection = psycopg2.connect(database=database_geoasma, user=user_geoasma, password=pass_geoasma,
                                     host=host_geoasma, port=port_geoasma)
    query = f'DROP TABLE IF EXISTS {name_table}'
    try:
        cur = conection.cursor()
        cur.execute(query)
        conection.close()
    except Exception as ex:
        conection.close()
        print('An exception occurred')
        print(ex)


#####################################
############# GEOASMA ###############
#####################################

def update_table_GeoAsma(data, name_table, geom=True):
    engine = create_engine(f'postgresql://docker:docker@10.232.5.113:25434/GeoAsma', echo=False)

    if geom:
        data['geometry'] = data['geometry'].apply(
            lambda geo: None if geo is None else geo if type(geo) == str else str(WKTElement(geo.wkt, srid=4326)))
    try:
        data.to_sql(name_table, engine, if_exists='replace', index=False)
        if geom:
            with engine.connect() as conn, conn.begin():
                sql = f'ALTER TABLE {name_table} ALTER COLUMN geometry TYPE Geometry ' \
                      f'USING ST_SetSRID(geometry::Geometry, 4326)'
                conn.execute(sql)
        print(f'Table {name_table} updated successfully!')
    except Exception as ex:
        print('An exception occurred')
        print(ex)


def get_table_geoasma(query):
    engine = create_engine(f'postgresql://docker:docker@10.232.5.113:25434/GeoAsma', echo=False)
    try:
        data = pd.read_sql_query(query, con=engine)
        return data
    except Exception as ex:
        print('An exception occurred')
        print(ex)


def get_geoasma_with_geometry_by_latlon():
    return get_table_geoasma('select d.*, case when d."COD_LATITUD"!=\'-1\' '
                             'then ST_Transform(ST_SetSRID(ST_Point(REPLACE(d."COD_LONGITUD", \',\', \'.\')::float,'
                             ' REPLACE(d."COD_LATITUD", \',\', \'.\')::float), 25830), 4326) else null end as geometry '
                             'from "Datos_pacientes_asma_2021" d')


def get_geoasma_with_latlon_null():
    return get_table_geoasma('select d.* from datos_pacientes_asma_2021 d where d."COD_LATITUD"=\'-1\' '
                             'or d."COD_LATITUD" is null')
