import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
pio.renderers.default = "browser"


def plot_proportion_difference(df, target_col):
    """
    works on binary columns
    """
    yes = df[df[target_col] == 1].iloc[:,:-1]
    no = df[df[target_col] == 0].iloc[:,:-1]
    total1 = len(yes)
    total2 = len(no)
    sr1 = yes.apply(lambda sr: pd.to_numeric(sr).sum() / total1)
    sr2 = no.apply(lambda sr: pd.to_numeric(sr).sum() / total2)
    fig = px.bar(sr1 - sr2)
    fig.update_layout(yaxis_range=[-0.5, 0.5])
    fig.show()


def plotratio(df):
    """
    works on binary columns
    """
    total = len(df)
    sr = df.apply(lambda sr: pd.to_numeric(sr).sum() / total)
    fig = px.bar(sr)
    fig.update_layout(yaxis_range=[0,1])
    fig.show()


def plothisto(df, x, y):
    tf = df.loc[:, [x, y]].dropna()
    fig = px.histogram(tf, x=x, color=y)
    fig.show()


def plot_corrmat(df, threshold):
    cormat = df.corr().values
    cormat[np.abs(cormat) < threshold] = np.nan
    fig = px.imshow(cormat, x=df.columns.astype(str), y=df.columns.astype(str))
    return fig


def remove_customer_id(df):
    df.drop('customerID',axis=1, inplace=True)


def exclude_columns(df, excluded_columns):
    new_columns = [col for col in df.columns if col not in excluded_columns]
    return df.loc[:,new_columns]


# -----------------------------------------------------------------
def clean_df_mk1(df):
    """
    Cleaner function to be used to generate training data
    """
    
    # drop row where all col is missing
    df.dropna(how="all", inplace=True)

    # NaN of total charge is assumed to be zero.
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # set senior citizen parameter to be binary
    df['SeniorCitizen'] = df.SeniorCitizen.astype('int')

    # customer id should not be included for building model
    df.drop('customerID',axis=1, inplace=True)

    # set lowercase to avoid typing Uppercase
    df = df.applymap(lambda s:s.lower() if type(s) == str else s)

    # some values can be simplified as 0 (False)
    binary_mapping = {
        'yes' : 1,
        'no'  : 0,
        'male': 1,
        'female' : 0,
        'no phone service':0,
        'no internet service':0
    }
    for colname, col in df.iteritems():
        if col.dtype == object and len(col.unique()) <= 3:
            df[colname] = col.replace(binary_mapping)

    # rest of non-binary is onehot encoded
    df = pd.get_dummies(df)
    #print(df.columns)
    # clean the result of get_dummies from high multicollinearity and redundancies
    rlist = [
        'InternetService_0',
        'Contract_month-to-month',
        'PaymentMethod_mailed check'
    ]
    df = df.drop(rlist, axis=1, errors='ignore')

    # simplify some long column names
    rename_dict = {
        'PaymentMethod_bank transfer (automatic)' : 'paymethod_transfer',
        'PaymentMethod_credit card (automatic)' : 'paymethod_cc',
        'PaymentMethod_electronic check' : 'paymethod_echeck',
        'PaymentMethod_mailed check' : 'paymethod_mailcheck',
        'InternetService_fiber optic' : 'internet_fiberoptic',
        'InternetService_dsl' : 'internet_dsl',
        'Contract_two year' : 'contract_two_year',
        'Contract_one year' : 'contract_one_year',
        'Contract_month-to-month':'contract_monthly'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # tidying up numeric variables to the end of the dataframe
    colnames = list(df.columns)
    moved_colnames = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Default']
    new_colnames = [name for name in colnames if name not in moved_colnames] + moved_colnames
    df = df.reindex(columns=new_colnames)
    return df


def clean_df_deploy(df):
    """
    Cleaner function to be used to clean request data
    """

    # drop row where all col is missing
    df.dropna(how="all", inplace=True)

    # fix all numerical strings
    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].replace(np.nan, 0, regex=True, inplace=True)

    # set senior citizen parameter to be binary integer
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

    # customer id should not be included for building model
    df.drop('customerID',axis=1, inplace=True)

    # gender is not included based on eda result
    df.drop('gender',axis=1, inplace=True)

    # set lowercase to avoid typing Uppercase
    df = df.applymap(lambda s:s.lower() if type(s) == str else s)

    # some values can be simplified as 0 (False)
    binary_mapping = {
        'yes' : 1,
        'no'  : 0,
        'male': 1,
        'female' : 0,
        'no phone service':0,
        'no internet service':0
    }
    for colname, col in df.iteritems():
        if col.dtype == object and len(col.unique()) <= 3:
            df[colname] = col.replace(binary_mapping)

    # rest of non-binary is onehot encoded
    df = pd.get_dummies(df)
    #print(df.columns)
    # clean the result of get_dummies from high multicollinearity and redundancies
    rlist = [
        'InternetService_0',
        'Contract_month-to-month',
        'PaymentMethod_mailed check'
    ]
    df = df.drop(rlist, axis=1, errors='ignore')

    # simplify some long column names
    rename_dict = {
        'PaymentMethod_bank transfer (automatic)' : 'paymethod_transfer',
        'PaymentMethod_credit card (automatic)' : 'paymethod_cc',
        'PaymentMethod_electronic check' : 'paymethod_echeck',
        'PaymentMethod_mailed check' : 'paymethod_mailcheck',
        'InternetService_fiber optic' : 'internet_fiberoptic',
        'InternetService_dsl' : 'internet_dsl',
        'Contract_two year' : 'contract_two_year',
        'Contract_one year' : 'contract_one_year',
        'Contract_month-to-month':'contract_monthly'
    }
    df.rename(columns=rename_dict, inplace=True)

    # tidying up numeric variables to the end of the dataframe
    colnames = list(df.columns)
    moved_colnames = ['tenure', 'MonthlyCharges', 'TotalCharges']
    new_colnames = [name for name in colnames if name not in moved_colnames] + moved_colnames
    df = df.reindex(columns=new_colnames)
    return df


def map_inputframe_modelframe(frames):
    """
    This function attempts to handle incoming json data to be predicted.
    May cause error if the input is tweaked to be different with the one used
    for training.
    """
    input_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]

    mapped_columns = [
        'SeniorCitizen', 'Partner','Dependents','PhoneService','MultipleLines',
        'OnlineSecurity','OnlineBackup','DeviceProtection', 'TechSupport',
        'StreamingTV',  'StreamingMovies','PaperlessBilling', 'internet_dsl',
        'internet_fiberoptic','contract_one_year','contract_two_year','paymethod_transfer',
        'paymethod_cc','paymethod_echeck', 'tenure', 'MonthlyCharges', 'TotalCharges'
    ]
    input_df = pd.DataFrame(frames, columns=input_columns)
    input_df = clean_df_deploy(input_df.copy())
    input_df =  input_df.reindex(labels=mapped_columns, axis=1, fill_value=0)
    return input_df
