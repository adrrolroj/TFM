from analysis import visualize_cluster_with_images, compare_clustering
import matplotlib.pyplot as plt

cluster_1 = 'kmeans'
cluster_2 = 'spectral'
k = 3

comb_po = compare_clustering(f'{cluster_1}_analysis', f'{cluster_2}_analysis', k, 'poblation', show=False)
comb_li = compare_clustering(f'{cluster_1}_analysis', f'{cluster_2}_analysis', k, 'life_condition', show=False)
visualize_cluster_with_images(f'{cluster_1}_analysis', show=False, transform_po=comb_po, transform_li=comb_li)
visualize_cluster_with_images(f'{cluster_2}_analysis', show=False)
plt.show()
