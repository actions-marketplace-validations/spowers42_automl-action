import os, ast
from uuid import uuid4 as uuid
import pandas as pd

dataset = os.environ['INPUT_DATASET']
target = os.environ['INPUT_TARGET']
usecase = os.environ['INPUT_USECASE']
repo = os.environ['GITHUB_REPOSITORY']

dataset_path = f'https://raw.githubusercontent.com/{repo}/master/{INPUT_DATASET}.csv'
data = pd.read_csv(dataset_path)
data.head()

if usecase == 'regression':
    from pycaret.regression import *
elif usecase == 'classification':
    from pycaret.classification import *
else:
    raise NotImplementedError("The use case {usecase} is not currently supported")

exp1 = setup(data, target=target, session_id=uuid(), silent=True, html=False, log_experiment=True, experiment_name='exp_github')
best = compare_models()
best_model = finalize_model(best)

save_model(best_model, 'model')
logs_exp_github = get_logs(save=True)