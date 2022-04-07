import matplotlib.pyplot as plt
import utils as utils

#data = utils.voronoi_pollen('03-25-2016')
#data.plot(column='Cupresaceas', legend=True)
data = utils.interpolate_pollen_censal_by_nearest_points('07-07-2020')
data.plot(column='Artemisa', legend=True)
plt.show()
