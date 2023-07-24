#data preprocessing lib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def data_preprocessing(df):
    # Assign 'id' as the index
    df.set_index('id', inplace=True)

    # Drop the specified columns
    columns_to_drop = ['name', 'host_id', 'host_name']
    df.drop(columns=columns_to_drop, inplace=True)

    # last review -> only year
    df['last_review'] = pd.to_datetime(df['last_review']).dt.year

    # Dropping the rows from the original DataFrame
    condition = df[(df['number_of_reviews'] == 0) & df['last_review'].isnull() & (df['reviews_per_month'].isnull()) & (df['availability_365'] == 0)]
    df = df.drop(condition.index)

    # Drop the specified columns
    columns_to_drop = ['reviews_per_month', 'last_review']
    df.drop(columns=columns_to_drop, inplace=True)

    # Drop outliers
    df = df.drop(df[(df['price'] >= 5000) & (df['neighbourhood_group'] == 'Brooklyn') & (df['room_type'] == 'Private room')].index)
    df = df.drop(df[(df['price'] == 9999) & (df['neighbourhood_group'] == 'Manhattan') & (df['room_type'] == 'Private room')].index)
    df = df.drop(df[(df['price'] == 10000) & (df['neighbourhood_group'] == 'Queens') & (df['room_type'] == 'Private room')].index)
    # label encoding

    le_neighbourhood_group = LabelEncoder()
    df['neighbourhood_group_str'] = df['neighbourhood_group']
    df['neighbourhood_group'] = le_neighbourhood_group.fit_transform(df.neighbourhood_group.values)

    le_neighbourhood = LabelEncoder()
    df['neighbourhood_str'] = df['neighbourhood'] 
    df['neighbourhood'] = le_neighbourhood.fit_transform(df.neighbourhood.values)

    le_room_type = LabelEncoder()
    df['room_type_str'] = df['room_type'] 
    df['room_type'] = le_room_type.fit_transform(df.room_type.values)    
    return df
