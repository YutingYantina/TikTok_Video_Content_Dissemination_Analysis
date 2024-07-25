import matplotlib.pyplot as plt
numeric_columns = ['likes', 'comment', 'collect', 'share']
data[numeric_columns].boxplot(figsize=(10, 6))
plt.show()
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_cleaned = data[~((data[numeric_columns] < lower_bound) | (data[numeric_columns] > upper_bound)).any(axis=1)]
