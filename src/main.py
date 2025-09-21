

# imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


FILE_NAMES = ['2016Q1', '2016Q2', '2016Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4']
TRAIN_FILE_NAMES = ['2016Q1', '2016Q2', '2016Q3', '2016Q4', '2017Q1', '2017Q2']
TEST_FILE_NAMES = ['2017Q3']
BACKTEST_FILE_NAMES = ['2017Q4']

# Reasoning in SUMMARY.md
COLS_TO_RMV = ['pub_rec_bankruptcies', 'acc_now_delinq', 'pub_rec',
                'sec_app_chargeoff_within_12_mths', 'chargeoff_within_12_mths', 'out_prncp',
                'sec_app_collections_12_mths_ex_med', 'tot_cur_bal', 'avg_cur_bal',
                'tot_coll_amt', 'mths_since_last_record', 'last_fico_range_low',
                'last_fico_range_high', 'last_credit_pull_d']

# Categorical features
CAT_FEATS = ['term', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership',
                'verification_status', 'issue_d', 'purpose', 'title', 'zip_code',
                'addr_state', 'earliest_cr_line', 'initial_list_status',
                'application_type', 'verification_status_joint', 'sec_app_earliest_cr_line']

# Values of loan_status that result in default/non-default status
DEFAULT_STATUS = ['Charged Off', 'Late (31-120 days)']
NONDEFAULT_STATUS = ['Fully Paid', 'Current', 'In Grace Period', 'Late (16-30 days)']

BUDGET = 50000


def process_data():
    """
    Loads in the data from CSV files, cleans up data (dropping columns, converting types, and
    handling missing values), and splits data into training and test data.
    """
    # Handle types, missing values, and obvious outliers.
    # Document any dropped columns and why.
    # Produce a quick data summary (rows, columns, target prevalence).

    train_data = pd.DataFrame
    test_data = pd.DataFrame
    backtest_data = pd.DataFrame

    # Load in train data
    for file_name in TRAIN_FILE_NAMES:
        if train_data.empty:
            train_data = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
        else:
            temp = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
            train_data = pd.concat([train_data, temp])

    # Load in test data
    for file_name in TEST_FILE_NAMES:
        if test_data.empty:
            test_data = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
        else:
            temp = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
            test_data = pd.concat([test_data, temp])

    # Load in backtest data
    for file_name in BACKTEST_FILE_NAMES:
        if backtest_data.empty:
            backtest_data = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
        else:
            temp = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
            backtest_data = pd.concat([backtest_data, temp])

    # Cleanup data
    train_data.drop(columns=COLS_TO_RMV, inplace=True)
    train_data['int_rate'] = train_data['int_rate'].str.rstrip('%').astype(float) / 100
    train_data['revol_util'] = train_data['revol_util'].str.rstrip('%').astype(float) / 100
    train_data['term'] = train_data['term'].str.rstrip(' months').astype(int)
    train_data.dropna(axis=1, how='all')

    # test_data.drop(columns=COLS_TO_RMV, inplace=True)
    test_data['int_rate'] = test_data['int_rate'].str.rstrip('%').astype(float) / 100
    test_data['revol_util'] = test_data['revol_util'].str.rstrip('%').astype(float) / 100
    test_data['term'] = test_data['term'].str.rstrip(' months').astype(int)
    test_data.dropna(axis=1, how='all')

    # backtest_data.drop(columns=COLS_TO_RMV, inplace=True)
    backtest_data['int_rate'] = backtest_data['int_rate'].str.rstrip('%').astype(float) / 100
    backtest_data['revol_util'] = backtest_data['revol_util'].str.rstrip('%').astype(float) / 100
    backtest_data['term'] = backtest_data['term'].str.rstrip(' months').astype(int)
    backtest_data.dropna(axis=1, how='all')

    # Change loan_status into default
    train_data['default_status'] = train_data['loan_status'].map(lambda x:
                        1 if x in DEFAULT_STATUS
                        else 0)

    test_data['default_status'] = test_data['loan_status'].map(lambda x:
                        1 if x in DEFAULT_STATUS
                        else 0)

    backtest_data['default_status'] = test_data['loan_status'].map(lambda x:
                        1 if x in DEFAULT_STATUS
                        else 0)

    X_train = train_data.drop(['loan_status', 'default_status'], axis=1)
    y_train = train_data[['default_status']]
    X_test = test_data.drop('default_status', axis=1)
    y_test = test_data[['default_status']]
    X_backtest = test_data.drop('default_status', axis=1)
    y_backtest = test_data[['default_status']]

    return X_train, y_train, X_test, y_test, X_backtest, y_backtest


def pipeline(X_train, y_train, X_test, y_test):
    """
    Finish preprocessing the data, train the model, and calculate/print scores. Option of
    doing no calibration, isotonic calibration, and/or sigmoid calibration.
    """
    # Handle categorical data
    cat_tf = Pipeline(
        steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
            # SelectKBest performed better than SelectPercentile
            ('selector', SelectKBest(chi2, k=20))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_tf, CAT_FEATS),
        ]
    )

    clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=10000, random_state=0)),
        ]
    )

    # Train Model and Calculate Scores/Probablities
    clf.fit(X_train, y_train.values.ravel())
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]
    clf_roc_score = roc_auc_score(y_test, prob_pos_clf)
    clf_brier_score = brier_score_loss(y_test, prob_pos_clf)

    prob_true, prob_pred = calibration_curve(y_test, prob_pos_clf, n_bins=25)

    print('Accuracy Score: ', clf.score(X_test, y_test))
    print('ROC-AUC Score: ', clf_roc_score)
    print('Calibration (Reliability Curve):')
    print('  Fraction of Positives: ', prob_true)
    print('  Mean Predicted Probability: ', prob_pred)
    print('Brier Score: ', clf_brier_score)
    print()

    disp = CalibrationDisplay.from_predictions(y_test, prob_pos_clf, n_bins=25)
    plt.savefig('calibration_curve.png')

    return clf


def backtest(clf, X_backtest, y_backtest):
    """
    Determines which loans to invest in based on the minimum PD and calculates the ROI proxy
    for these loans.
    """
    # Find loans to invest in
    prob_pos_clf = clf.predict_proba(X_backtest)[:, 1]
    X_backtest['prob_default'] = prob_pos_clf

    # To test that this will work when the PD is not the same for the k_smallest
    #   rows, uncomment this code; it drops all but one of the smallest PD rows
    #   this also has an example of a loan that defaults and therefore has a
    #   negative ROI proxy
    # condition1 = X_backtest['prob_default'] == 0.016910529536703012
    # condition2 = X_backtest['id'] != 111865260
    # indices_to_drop = X_backtest[condition1 & condition2].index
    # X_backtest.drop(indices_to_drop, inplace=True)

    k_smallest = X_backtest.nsmallest(1, 'prob_default', keep='all')
    sum_k_smallest = k_smallest['funded_amnt'].sum()
    count = 2

    while sum_k_smallest < BUDGET:
        k_smallest = X_backtest.nsmallest(count, 'prob_default', keep='all')
        sum_k_smallest = k_smallest['funded_amnt'].sum()
        count += 1

    k_smallest = k_smallest.sort_values("funded_amnt")

    amt_sum = 0
    indexes = []
    for index, row in k_smallest.iterrows():
        amt_sum += row['funded_amnt']
        if amt_sum >= BUDGET:
            break
        indexes.append(index)

    loans_to_invest_in = X_backtest.loc[indexes]

    print('Number of Loans Selected:', len(loans_to_invest_in))
    print()

    # Selected default rate vs. overall default rate
    print('Default rate(s) for selected loans:', loans_to_invest_in['prob_default'].unique())
    print('Minimum default rate:', X_backtest['prob_default'].min())
    print('Maximum default rate:', X_backtest['prob_default'].max())
    print('Average default rate:', X_backtest['prob_default'].mean())
    print()

    loan_info_w_roi = loans_to_invest_in[['id', 'loan_status', 'funded_amnt', 'prob_default']].copy()
    loan_info_w_roi['collected_payments'] = np.nan
    loan_info_w_roi['ROI_proxy'] = np.nan

    for index, row in loans_to_invest_in.iterrows():
        collected_payments = 0

        # if default
        if row['loan_status'] in DEFAULT_STATUS:
            collected_payments = 0.30 * row['installment'] * row['term']
        # if not default
        else:
            collected_payments = row['installment'] * row['term']

        ROI_proxy = (collected_payments - row['funded_amnt']) / row['funded_amnt']

        loan_info_w_roi.loc[index, 'collected_payments'] = collected_payments
        loan_info_w_roi.loc[index, 'ROI_proxy'] = ROI_proxy

    print('Loans Selected to Invest In')
    print(loan_info_w_roi.to_markdown(index=False))


def feature_importance(clf, X_train, y_train):
    """
    Calculates permutation feature importance and prints the features in order of most to
        least importance.
    """
    # This takes a really long time to run, the contents of importances_mean is the
    #   return of perm_imp['importances_mean']
    # perm_imp = permutation_importance(clf, X_train, y_train, random_state=0, scoring='roc_auc', n_repeats=20, n_jobs=-2, max_samples=10000)
    # print(perm_imp['importances_mean'])
    importances_mean = [0.00866953, 0.00866953, 0.01056244, 0.00866953, 0.00866953, 0.10624905,
        0.00866953, 0.00866953, 0.02943634, 0.00866953, 0.02572951, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953, 0.00866953,
        0.00866953, 0.00866953, 0.00866953, 0.00866953]
    sorted_im = pd.DataFrame(sorted(zip(importances_mean, X_train.columns), reverse=True), columns=['importance_mean', 'feature'])
    print()
    print(sorted_im)


def main():
    X_train, y_train, X_test, y_test, X_backtest, y_backtest = process_data()

    clf = pipeline(X_train, y_train, X_test, y_test)

    backtest(clf, X_backtest, y_backtest)

    feature_importance(clf, X_train, y_train)



if __name__ == '__main__':
    main()