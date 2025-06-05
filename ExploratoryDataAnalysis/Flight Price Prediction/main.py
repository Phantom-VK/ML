import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

#Dataset downloaded form kaggle
df = pd.read_csv('FlightDataset.csv')
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df['flight'].nunique())
# print(df.describe())

#I dont think flight codes does matter much so we'll just drop it
df.drop('flight', axis=1, inplace=True)


#Airline Column, Source City, Departure Time, Arrival Time, class its categorical data
# We'll use One Hot Encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
oh_encoded = ohe.fit_transform(df[['airline','departure_time','arrival_time', 'class']])
ohe_df = pd.DataFrame(oh_encoded, columns=ohe.get_feature_names_out())

#Now we will add encoded data in original df
#first drop original one
df.drop(['airline','departure_time','arrival_time', 'class'], axis=1, inplace=True)
df_encoded = pd.concat([df, ohe_df], axis=1)

#Now lets use ordinal encoder on class
df['stops'] = OrdinalEncoder(categories=[['zero', 'one', 'two_or_more']]).fit_transform(df[['stops']])

#Now source_city and destination_city affects price directly so we can use Target Guided Encoding
# Encode source_city based on mean price
source_map = df.groupby('source_city')['price'].mean().to_dict()
df['source_city_encoded'] = df['source_city'].map(source_map)

# Encode destination_city similarly
dest_map = df.groupby('destination_city')['price'].mean().to_dict()
df['destination_city_encoded'] = df['destination_city'].map(dest_map)

#Lets drop old columns
df.drop(['source_city', 'destination_city'], axis=1, inplace=True)


df.to_csv('CleanedFlightDataset.csv')

