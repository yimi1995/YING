#先选出前7个cell的数据
import pandas as pd
from pandas import DataFrame
df = pd.read_csv('./data/timeseries.csv')
print(df.head())
#用前1400个时间间隔的数据做训练
df = df.ix[0:2200, 0:2]
df = DataFrame(df)
df = df.to_csv('./data/ID01.csv', index=False)
