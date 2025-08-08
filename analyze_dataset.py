import pandas as pd

df = pd.read_excel('Fe-HCl-317_dataset.xlsx')

print(df.head())
print(df.info())
print(df.describe())

