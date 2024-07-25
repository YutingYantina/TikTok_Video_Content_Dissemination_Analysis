I first use IQR in clean_outliers.py to clean outliers data in alldata since some data are outliers and there are 22 titles missing.
I use K-means to cluster the BERT embeddings, and analyze each cluster (50 clusters)'s high frequency words(first 5 each) and their average likes number
Then I use cleaned_alldata.csv as the dataset to turn each title into BERT embeddings, and merge them with original data and output cleaned_merged_data
And I use decision tree model to make the models and then I can predict likes, comments, collect, share when given a predicted title!
