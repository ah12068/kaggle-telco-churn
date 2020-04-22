import logging
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from constants import (PLOTLY_TEMPLATE, PANDAS_TEMPLATE)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve


def baseline_trainer(processed_df, algorithm, cf):
    logger = logging.getLogger(__name__)
    id_col = ['customerID']
    target_col = ["Churn"]

    train, test = train_test_split(processed_df, test_size=.25, random_state=111)

    cols = [i for i in processed_df.columns if i not in id_col + target_col]
    train_X = train[cols]
    train_Y = train[target_col]
    test_X = test[cols]
    test_Y = test[target_col]

    logger.info('Building and Validating Model')

    algorithm.fit(train_X, train_Y)
    predictions = algorithm.predict(test_X)
    probabilities = algorithm.predict_proba(test_X)

    if cf == "coefficients":
        coefficients = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features":
        coefficients = pd.DataFrame(algorithm.feature_importances_)

    column_df = pd.DataFrame(cols)
    coef_sumry = (pd.merge(coefficients, column_df, left_index=True,
                           right_index=True, how="left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    print(algorithm)
    print(f"\n Classification report : \n, {classification_report(test_Y, predictions)}")
    print(f"Accuracy Score : {accuracy_score(test_Y, predictions)}\n")

    conf_matrix = confusion_matrix(test_Y, predictions)
    print(f'Confusion Matrix:\n{conf_matrix}')
    model_roc_auc = roc_auc_score(test_Y, predictions)
    print(f"Area under curve :\n{model_roc_auc} \n")
    fpr, tpr, thresholds = roc_curve(test_Y, probabilities[:, 1])

    logger.info('Producing Evaluation Report')

    trace1 = go.Heatmap(z=conf_matrix,
                        x=["Not Churn", "Churn"],
                        y=["Not Churn", "Churn"],
                        showscale=False,
                        colorscale="Picnic",
                        name="matrix")

    # plot roc curve
    trace2 = go.Scatter(x=fpr, y=tpr,
                        name="Roc : " + str(model_roc_auc),
                        line=dict(color='rgb(22, 96, 167)', width=2))
    trace3 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color='rgb(205, 12, 24)', width=2,
                                  dash='dot'))

    # plot coeffs
    trace4 = go.Bar(x=coef_sumry["features"], y=coef_sumry["coefficients"],
                    name="coefficients",
                    marker=dict(color=coef_sumry["coefficients"],
                                colorscale="Picnic",
                                line=dict(width=.6, color="black")))

    # subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                        subplot_titles=('Confusion Matrix',
                                        'Receiver operating characteristic',
                                        'Feature Importances'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 2, 1)

    fig['layout'].update(showlegend=False, title="Model performance",
                         autosize=False, height=900, width=800,
                         plot_bgcolor='rgba(240,240,240, 0.95)',
                         paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195))
    fig["layout"]["xaxis2"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid=True, tickfont=dict(size=10),
                                        tickangle=90))
    fig.layout['hovermode'] = 'x'
    fig.show()


    return algorithm
