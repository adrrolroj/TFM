import db_calls as db
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from analysis import visualize_cluster_with_images

# Opciones kmeans, spectral, aglomerative
method = 'spectral'

df = db.get_table_analysis(f'SELECT * FROM {method}_analysis')
pie_po = mpimg.imread(f'exports/clustering/{method}_poblation_pie_graph.png')
profiling_po = mpimg.imread(f'exports/clustering/{method}_poblation_profiling_centers.png')
pie_li = mpimg.imread(f'exports/clustering/{method}_life_condition_pie_graph.png')
profiling_li = mpimg.imread(f'exports/clustering/{method}_life_condition_profiling_centers.png')

fig, axs = plt.subplots(1, 2)
df.plot(column='poblation', ax=axs[0])
axs[0].axis('off')
axs[0].set_title('Segmentacion poblacion')
df.plot(column='life_condition', ax=axs[1])
axs[1].axis('off')
axs[1].set_title('Segmentacion condiciones de vida')


fig2, axs2 = plt.subplots(2, 2)
axs2[0, 0].axis('off')
axs2[1, 0].axis('off')
axs2[0, 0].imshow(pie_po, aspect='equal')
axs2[1, 0].imshow(profiling_po, aspect='auto')
axs2[0, 1].axis('off')
axs2[1, 1].axis('off')
axs2[0, 1].imshow(pie_li, aspect='equal')
axs2[1, 1].imshow(profiling_li, aspect='auto')


visualize_cluster_with_images(f'{method}_analysis')
plt.show()

