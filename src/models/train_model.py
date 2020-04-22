import pandas as pd
from constants import (
    baseline_classifiers,
)
from functions import baseline_trainer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('../../data/processed/processed.csv')

for classifier in baseline_classifiers.keys():
    print(classifier)
    print(
        baseline_trainer(
            processed_df=df,
            algorithm=baseline_classifiers[classifier],
            cf='coefficients',
            threshold_plot=True
        )
    )
    break
