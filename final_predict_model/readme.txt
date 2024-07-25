Based on last week's experiment, I tried to find out a way to analysis performance of model.
Instead of regression, I try to analyze the relationship between x(columns=['likes', 'comment', 'collect', 'share']) and log ys (log merged_data['likes'], log merged_data['comment'], ..., etc ).
However, I find out that the average square error among xy, x log y, and log x y are basically similar.
平均方差
Likes - Mean Squared Error: 14184152.1649458
Comments - Mean Squared Error: 50005.15565657904
Collects - Mean Squared Error: 436468.154548054
Shares - Mean Squared Error: 104967.77307561308

平均标准差
Likes: 3766.19
Comments: 223.61
Collects: 660.66
Shares: 323.99

logy 
Likes - Mean Squared Error: 13286690.956819542
Comments - Mean Squared Error: 53008.50581858607
Collects - Mean Squared Error: 413712.3852035056
Shares - Mean Squared Error: 100974.21348091432

log x
Likes - Mean Squared Error (MSE): 14018073.617261875
Comments - Mean Squared Error (MSE): 51817.64630428566
Collects - Mean Squared Error (MSE): 432347.3639115956
Shares - Mean Squared Error (MSE): 128069.46158814344

possible reasons
决策树回归模型是一种非参数模型，它可以非常灵活地拟合数据。
即使目标变量进行了对数变换，决策树模型仍然能够捕捉到数据中的模式，因此两种模型的预测误差分布可能非常相似。
对数变换的效果：
对数变换可以平滑数据中的极端值，但在目标值的整体分布上可能没有显著的变化。
这意味着对数变换后的回归模型和原始回归模型可能会对同样的特征有相似的预测能力，从而导致相似的标准差。

So I try to find a way to analyze performance on accuracy
So I change the model a little bit to DecisionTreeClassifier model, by discrete datas in different ways, including discrete them evenly in 3 segments, 10 segments and manually define the like bins.
3 dimensions for high median low
Likes - Accuracy Score: 0.8976420880468856
Comments - Accuracy Score: 0.8912362000817773
Collects - Accuracy Score: 0.8900095406842033
Shares - Accuracy Score: 0.8889191767752488


likes_bins = [0, 500, 1000, 5000, 10000, np.inf]
Likes - Accuracy Score: 0.8859206760256235
Comments - Accuracy Score: 0.8912362000817773
Collects - Accuracy Score: 0.8900095406842033
Shares - Accuracy Score: 0.8889191767752488

10 dimensions
Likes - Accuracy Score: 0.8319476625323702
Comments - Accuracy Score: 0.8259506610331198
Collects - Accuracy Score: 0.8249965926127845
Shares - Accuracy Score: 0.8200899550224887

Since above 65%, this can be considered as a nice accuracy, and 90% is well done, I output the model called predictor 
Sample:
input_sentence = "中华本草，小故事"
"中华本草，小故事"
Predicted Likes: 1000 - 5000
Predicted Comments: 14.0 - 82.0
Predicted Collects: 0.0 - 56.0
Predicted Shares: 22.0 - 128.0

Then I wanna see the performance of logistic regression classifier model vs behavior of decision tree classify model
Accuracy score of logistic regression classifier model, 5 segments
Likes - Accuracy Score: 0.6046067875153333
Comments - Accuracy Score: 0.581981736404525
Collects - Accuracy Score: 0.678751533324247
Shares - Accuracy Score: 0.6753441461087638
It seems the performance of  decision tree classify model is the best, with lowest error and se can get a range of possible likes!