from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_pd
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from google.colab import files
train_data = pd.read_csv('/content/sample_data/test.csv')
test_data = pd.read_csv('/content/sample_data/train.csv')
predictors = {}
targets = ['likes', 'comment', 'collect', 'share']
for target in targets:
    predictors[target] = MultiModalPredictor(label=target)
    predictors[target].fit(train_data=train_data)
predictions = {}
for target in targets:
    predictions[target] = predictors[target].predict(test_data)
    test_data[f'predicted_{target}'] = predictions[target]
accuracy_metrics = {}
for target in targets:
    r2 = r2_score(test_data[target], test_data[f'predicted_{target}'])
    mape = mean_absolute_percentage_error(test_data[target], test_data[f'predicted_{target}'])
    accuracy_metrics[target] = {'R²': r2, 'MAPE': mape}
for target, metrics in accuracy_metrics.items():
    print(f"Accuracy metrics for {target}:")
    print(f"R²: {metrics['R²'] * 100:.2f}%")
    print(f"MAPE: {metrics['MAPE'] * 100:.2f}%\n")