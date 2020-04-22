import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler

def main():
    """ Runs feature building and places data in ../processed
    """
    logger = logging.getLogger(__name__)
    raw_df = pd.read_csv('../../data/raw/Telco-Customer-Churn.csv')
    rob_scaler = RobustScaler()
    le = LabelEncoder()

    logger.info('Cleaning Data')

    raw_df['TotalCharges'] = raw_df["TotalCharges"].replace(" ", np.nan)
    raw_df.dropna(axis=0, inplace=True)
    raw_df["TotalCharges"] = raw_df["TotalCharges"].astype(float)
    raw_df.reset_index(drop=True, inplace=True)

    logger.info('Building Features')

    replace_cols = [
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]
    raw_df["SeniorCitizen"] = raw_df["SeniorCitizen"].replace({1: "Yes", 0: "No"})

    for col in replace_cols:
        raw_df[col] = raw_df[col].replace({'No internet service': 'No'})

    id_col = ['customerID']
    target_col = ["Churn"]

    categorical_cols = raw_df.nunique()[raw_df.nunique() < 5].keys().tolist()
    categorical_cols = [col for col in categorical_cols if col not in target_col]

    numerical_cols = [col for col in raw_df.columns if col not in categorical_cols + target_col + id_col]
    binary_cols = raw_df.nunique()[raw_df.nunique() == 2].keys().tolist()
    multi_cols = [col for col in categorical_cols if col not in binary_cols]

    for col in binary_cols:
        raw_df[col] = le.fit_transform(raw_df[col])

    raw_df = pd.get_dummies(data=raw_df, columns=multi_cols)

    scaled = rob_scaler.fit_transform(raw_df[numerical_cols])
    scaled = pd.DataFrame(scaled, columns=numerical_cols)

    raw_df = raw_df.drop(columns=numerical_cols, axis=1)
    raw_df = raw_df.merge(scaled, left_index=True, right_index=True, how="left")

    raw_df.sample(frac=1, replace=False, random_state=0, axis=0)

    raw_df.to_csv(f'../../data/processed/processed.csv', index=False)

    logger.info(f'Features built. \n Features: {raw_df.columns}')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()