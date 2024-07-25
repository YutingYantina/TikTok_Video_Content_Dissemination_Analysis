import pandas as pd
input_file = '/content/cleaned_alldata.csv'
data = pd.read_csv(input_file)
train_data = data.iloc[::2] 
test_data = data.iloc[1::2] 
train_data.to_csv('/content/train.csv', index=False)
test_data.to_csv('/content/test.csv', index=False)