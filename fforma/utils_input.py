import pandas as pd

def _check_valid_df(df):

    is_pandas_df = isinstance(df, pd.DataFrame)
    assert  is_pandas_df or isinstance(df, pd.Series)

    return is_pandas_df

def _check_valid_columns(df,
                         cols=['unique_id','ds', 'y'],
                         cols_index=['unique_id', 'ds']):

    correct_cols_df = all([item in df.columns for item in cols])
    correct_cols_index = all([item in df.index.names for item in cols_index])

    assert correct_cols_df or correct_cols_index

def _check_same_type(df_x, df_y):
    assert type(df_x) == type(df_y)

def _check_passed_dfs(df_x, df_y):

    for df in [df_x, df_y]:
        is_pandas_df = _check_valid_df(df)
        _check_valid_columns(df)

    _check_same_type(df_x, df_y)

    return is_pandas_df
