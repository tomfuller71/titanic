import pandas as pd
import numpy as np


def passenger_hash(df: pd.DataFrame, col='Name', new_col='P_Hash'):
    """
    Create a passenger hash of name and last three chars of ticket number
    :param df: The dataframe
    :param col: The column name - default is 'name'
    :param new_col: The new column name
    :return: The dataframe with the column stripped -default is 'parsed_name'
    """
    if df[col].dtype == 'O':
        df[new_col] = df[col].str.lower()
        df[new_col] = df[new_col].str.replace(r'[^a-z]', '')
        df[new_col] = df[new_col] + df['Ticket'].str[-3:]
        return df
    else:
        print('Column is not of type object')
        return df


def main():
    full_titanic = pd.read_csv('../base_data/titanic.csv')
    kaggle_test = pd.read_csv('../base_data/test_kaggle.csv')

    # Rename the columns in the full_titanic dataframe
    full_titanic.columns = full_titanic.columns.str.title()
    full_titanic.rename(columns={'Sibsp': 'SibSp'}, inplace=True)

    # Replace '?' with "" to be consistent with the kaggle_test dataframe
    full_titanic = full_titanic.replace('?', "")


    # Create consistent 'P_Hash' column in both dataframes
    full_titanic = passenger_hash(full_titanic)
    kaggle_test = passenger_hash(kaggle_test)


    # Get row index mask of matched 'P_Hash' values for the kaggle_test dataframe
    indexes = full_titanic[full_titanic['P_Hash'].isin(kaggle_test['P_Hash'])].index

    # Separate the full_titanic dataframe into two dataframes train and test based on the indexes
    train = full_titanic.drop(indexes).reindex()
    test = full_titanic.loc[indexes]

    print(f"Kaggle test dataframe length: {kaggle_test.shape[0]}")
    print(f"Test dataframe length: {test.shape[0]}")
    print(f"Train dataframe length: {train.shape[0]}")
    assert kaggle_test.shape[0] == test.shape[0], "Missing data in test dataframe"

    # Create the PassengerId column in the train and test dataframes
    train['PassengerId'] = train.index + 1

    # Merge the kaggle_test PassengerId column with the test dataframe on the P_Hash column
    test = pd.merge(test, kaggle_test[['PassengerId', 'P_Hash']], on='P_Hash', how='left')

    # Sort the test dataframe by the PassengerId column
    test = test.sort_values(by='PassengerId')

    # Reorder / drop the columns in the train and test dataframes
    kaggle_columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked', 'Survived']
    train = train[kaggle_columns]
    test = test[kaggle_columns]

    # Save the train and test dataframes to csv files
    train.to_csv('../input/train.csv', index=False)
    test.to_csv('../input/test.csv', index=False)


if __name__ == '__main__':
    main()



