from itertools import permutations
from os import walk

import matplotlib.image as mpimg
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import db_calls as db


def analyze_dataframe(dataframe, colums_to_analyze, title):
    data = pd.DataFrame(columns=colums_to_analyze)
    for colum in colums_to_analyze:
        data[colum] = dataframe[colum]
    print('Starting analysis...')
    # Analizamos los datos y exportamos a csv
    stadistics = pd.DataFrame.describe(data)
    stadistics.to_csv(f'exports/analysis/{title}-analysis_stadistics.csv')
    print('Dataframe statistics completed')
    # Analisis del histograma de los atributos
    colums = data.columns.to_list()
    square = math.sqrt(len(colums))
    fig, axs = plt.subplots(math.ceil(square), math.ceil(square))
    for i, colum in enumerate(colums):
        x = math.floor(i / square)
        y = i % int(square)
        axs[x, y].hist(data[colum], density=True, bins=30)
        axs[x, y].set_title(colum)
    fig.savefig(f'exports/analysis/{title}-histogram_data.png')
    fig.show()
    print('Histogram completed')
    # Ahora se analiza los coeficientes de correlacion
    plt.figure(figsize=(15, 15))
    plt.title("Pearson's Correlation Coefficient between attributes", y=1.05, size=15)
    sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap='viridis', linecolor='white', annot=True)
    plt.savefig(f'exports/analysis/{title}-correlation_coefficients.png')
    plt.show()
    print('Analisis completed')


def profiling_centers(centers, data, atributes, name):
    cluster_points = np.array(
        [[center, atr, color] for color, cat in enumerate(centers) for atr, center in enumerate(cat)])
    mean_values = data[atributes].mean().values
    mean_points = np.array([[mean, atr] for atr, mean in enumerate(mean_values)])

    plt.figure(figsize=(10, 5))
    plt.yticks(np.array(range(0, len(atributes))), change_atribute_names(atributes))
    plt.title(f'Profiling of {name}')
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_points[:, 2], alpha=0.8, marker='o', s=100)
    plt.scatter(mean_points[:, 0], mean_points[:, 1], c='#000000', marker='s', alpha=0.8, s=100)
    cmap = matplotlib.cm.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(centers) - 1)

    handles = []
    for i in range(0, len(centers)):
        handles.append(mpatches.Patch(color=cmap(norm(i)), label=f'Cluster {i}', capstyle='round'))
    handles.append(mpatches.Patch(color='#000000', label=f'Mean value', capstyle='round'))
    plt.legend(handles=handles)
    plt.savefig(f'exports/clustering/{name}_profiling_centers.png')
    plt.show()


def change_atribute_names(atributes):
    names = {"pob1": "Males percent", "pob2": "Females percent", "pob3": "<16 years", "pob4": "16-64 years",
             "pob5": ">64 years",
             "pob6": "Spanish percent", "pob7": "Foreigner percent", "pob8": "Pop. density", "edu1": "No studies",
             "edu2": "middle studies", "edu3": "complete studies", "apa1": "1st properties", "apa2": "2nd properties",
             "apa3": "Empty properties",
             "apa4": "Prop. density", "hom1": "1-2 persons per home", "hom2": "+3 persons per home",
             "hom3": "home density", "lif1": "incomes per person",
             "lif2": "Unemployment", "lif3": "Mortality rate"}
    return [names[x] for x in atributes]


def pie_graph(category, number, name):
    cluster = []
    for c in category:
        cluster.append(f'Cluster {c}')
    cmap = matplotlib.cm.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(cluster) - 1)
    colors = [cmap(norm(i)) for i in range(0, len(cluster))]
    plt.pie(number, labels=cluster, colors=colors)
    plt.title(f'Circle diagram of cluster {name}')
    plt.savefig(f'exports/clustering/{name}_pie_graph.png')
    plt.show()


def compare_clustering(name1, name2, k, segmentation, show=True):
    cluster_1 = db.get_table(f'SELECT * FROM {name1}', 'Analysis')[['geometry', segmentation]]
    cluster_2 = db.get_table(f'SELECT * FROM {name2}', 'Analysis')[['geometry', segmentation]]
    comparative = db.compare_cluster_from_analysis(name1, name2, segmentation).groupby(['first'])
    combinations = list(permutations(range(k), k))
    values = []
    for combination in combinations:
        value = []
        for i in range(k):
            group = comparative.get_group(i)
            try:
                value.append(group['count'][group[group['second'] == combination[i]].index.values].values[0])
            except:
                value.append(0)
        values.append(sum(value))

    combination = combinations[values.index(max(values))]
    print(f'Para {segmentation}')
    [print(f'Cluster {a} : Cluster {b} ') for a, b in enumerate(combination)]
    cluster_1[segmentation] = cluster_1[segmentation].map(lambda x: combination[x])
    dif = []
    for i in range(len(cluster_1)):
        if cluster_1[segmentation][i] == cluster_2[segmentation][i]:
            dif.append('Same')
        else:
            dif.append('Distinct')

    fig, ax = plt.subplots(1, 3)
    ax[0].axis('off')
    ax[0].set_title(f'{name1}')
    cluster_1.plot(column=segmentation, ax=ax[0])
    ax[1].axis('off')
    ax[1].set_title(f'{name2}')
    cluster_2.plot(column=segmentation, ax=ax[1])
    ax[2].axis('off')
    ax[2].set_title(f'Diferencia')
    cluster_1[segmentation] = dif
    cluster_1.plot(column=segmentation, ax=ax[2], legend=True)
    fig.suptitle(segmentation, fontsize=16)
    if show: plt.show()
    return combination


def visualize_cluster_with_images(name_cluster, show=True, transform_po=None, transform_li=None):
    names_images = next(walk('images/'), (None, None, []))[2]
    square = math.sqrt(len(names_images))
    fig, axs = plt.subplots(math.ceil(square), math.ceil(square))
    for i, name_image in enumerate(names_images):
        image = mpimg.imread(f'images/{name_image}')
        name = name_image.split(")")[0][1:]
        cusec = name_image.split(")")[1].split(".")[0]
        data = db.get_table_analysis_by_CUSEC(name_cluster, cusec)

        x = math.floor(i / math.ceil(square))
        y = i % math.ceil(square)
        axs[x, y].imshow(image, aspect='equal')
        axs[x, y].axes.get_xaxis().set_ticks([])
        axs[x, y].axes.get_yaxis().set_ticks([])
        axs[x, y].set_title(name)

        if transform_li is None:
            axs[x, y].set_xlabel(
                f'Cluster poblacion: {data["poblation"].values[0]} \n Cluster condiciones de vida: {data["life_condition"].values[0]}')
        else:
            axs[x, y].set_xlabel(
                f'Cluster poblacion: {transform_po[data["poblation"].values[0]]} \n Cluster condiciones de vida: {transform_li[data["life_condition"].values[0]]}')
    fig.suptitle(name_cluster, fontsize=16)
    fig.tight_layout()
    if show: plt.show()
