import logging
import pandas as pd
from constants import (
    baseline_classifiers,
    feature_coefs
)
from functions import baseline_trainer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():

    logger = logging.getLogger(__name__)

    df = pd.read_csv('../../data/processed/processed.csv')

    logger.info(f'Training baseline models')

    for classifier in baseline_classifiers.keys():
        logger.info(f'Classifier: {classifier}')
        if classifier in feature_coefs:
            baseline_trainer(
                processed_df=df,
                algorithm=baseline_classifiers[classifier],
                cf='features',
                name=classifier

        )

        else:
            baseline_trainer(
                processed_df=df,
                algorithm=baseline_classifiers[classifier],
                cf='coefficients',
                name=classifier
        )

    logger.info(f'DOWNLOAD PLOTLY REPORTS')

    return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()