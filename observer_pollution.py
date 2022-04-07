import db_calls as db
import matplotlib.pyplot as plt

atribute = 'NO2'
df = db.get_table('select * from censal_pollution', 'GeoHealth')
df.plot(column=atribute, legend=True)
plt.show()