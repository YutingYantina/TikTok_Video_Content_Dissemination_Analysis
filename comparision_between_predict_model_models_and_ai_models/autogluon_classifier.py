from autogluon.tabular import TabularPredictor
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score
from google.colab import files
train_data = pd.read_csv('/content/sample_data/train.csv')
test_data = pd.read_csv('/content/sample_data/train.csv')
print(train_data.head())
print(test_data.head())
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
for column in ['likes', 'comment', 'collect', 'share']:
    train_data[f'{column}_bin'] = discretizer.fit_transform(train_data[[column]])
    test_data[f'{column}_bin'] = discretizer.transform(test_data[[column]])
print(train_data.head())
print(test_data.head())
predictors = {}
targets = ['likes_bin', 'comment_bin', 'collect_bin', 'share_bin']
for target in targets:
    predictors[target] = TabularPredictor(label=target).fit(train_data=train_data, presets='best_quality')
for target in targets:
    predictors[target].save(f'classifier_model_{target}')
predictions = {}
accuracies = {}
for target in targets:
    predictor = TabularPredictor.load(f'classifier_model_{target}')
    predictions[target] = predictor.predict(test_data)
    test_data[f'predicted_{target}'] = predictions[target]
    accuracies[target] = accuracy_score(test_data[target], test_data[f'predicted_{target}'])
print(test_data.head())
for target in targets:
    print(f"Accuracy for {target}: {accuracies[target] * 100:.2f}%")