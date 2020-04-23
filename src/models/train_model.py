import pandas as pd
import warnings
import logging
from constants import LogisticRegression_grid
from functions import create_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    logger = logging.getLogger(__name__)


    processed_df = pd.read_csv(f'../../data/processed/processed.csv')

    id_col = ['customerID']
    target_col = ["Churn"]
    cols = [i for i in processed_df.columns if i not in id_col + target_col]

    cate_cols = processed_df.nunique()[processed_df.nunique() == 2].keys().tolist()
    cate_cols = [col for col in cate_cols if col not in target_col]
    cate_cols_idx = [processed_df.columns.get_loc(col) for col in cate_cols]

    smote_X = processed_df[cols]
    smote_Y = processed_df[target_col]

    smote_train_X, smote_test_X, smote_train_Y, smote_test_Y = train_test_split(smote_X, smote_Y,
                                                                                test_size=.25,

                                                                                random_state=111)
    logger.info(f'Applying SMOTE')

    os = SMOTENC(categorical_features=cate_cols_idx, sampling_strategy='minority', random_state=0)
    os_smote_X, os_smote_Y = os.fit_sample(smote_train_X, smote_train_Y)
    os_smote_X = pd.DataFrame(data=os_smote_X, columns=cols)
    os_smote_Y = pd.DataFrame(data=os_smote_Y, columns=target_col)

    logger.info(f'Fitting Logistic Regression and Tuning')

    lr = LogisticRegression(max_iter=500)

    clf = GridSearchCV(
        estimator=lr,
        param_grid=LogisticRegression_grid,
        cv=5
    )

    best_model = clf.fit(os_smote_X.values, os_smote_Y.values.ravel())

    logger.info(f'Best Parameters: {best_model.best_params_}')

    metrics = create_report(best_model, smote_test_X, smote_test_Y)
    logger.info(f'{metrics}')
    f = open(f'../../models/logistigregression_best_metrics.txt', 'w')
    f.write(metrics)
    f.close()
    joblib.dump(best_model, f'../../models/logsticreg_best.pkl', compress=9)
    logger.info(f'Model and Evaluation saved to "models/"')

    return

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
