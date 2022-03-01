import db_calls as db
import matplotlib.pyplot as plt

atribute = 'NO2'
df = db.get_table_analysis('select * from censal_pollution')
df.plot(column=atribute, legend=True)
plt.show()