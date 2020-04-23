from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from numpy import logspace
import pandas as pd
import plotly.io as pio

PLOTLY_TEMPLATE = pio.templates.default = 'plotly_white'
PANDAS_TEMPLATE = pd.set_option('display.float_format', '{:.5f}'.format)

random_seed = 1
best_model_file_name = 'final_model.pkl'

baseline_classifiers = {
    "LogisticRegression": LogisticRegression(random_state=random_seed),
    "SVM": SVC(probability=True, kernel='linear'),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_seed)
}

feature_coefs = [
    "DecisionTreeClassifier",
]

LogisticRegression_grid = {
    "penalty": ['l2'],
    "C": logspace(0, 4, 10)
}

LogisticRegression_rndm_params = {
    "penalty": ['l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

model_metrics = {
    'AUC':'roc_auc',
    'RECALL':'recall',
    'PRECISION':'precision',
    'F1':'f1'
}