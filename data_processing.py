import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("artifacts/collisions_cleaned.csv")

columns_to_del = ["OBJECTID", "ACCIDENTNUM", "ACCIDENT_YEAR", "ACCIDENT_MONTH", "ACCIDENT_DAY", "ACCIDENT_HOUR", "ACCIDENT_MINUTE", "ACCIDENT_SECOND", "XCOORD", "YCOORD", "LONGITUDE", "LATITUDE", "COLLISIONTYPE", "CLASSIFICATIONOFACCIDENT", "IMPACTLOCATION", "INITIALDIRECTIONOFTRAVELONE", "INITIALDIRECTIONOFTRAVELTWO", "INITIALIMPACTTYPE", "INTTRAFFICCONTROL", "LIGHTFORREPORT", "THRULANENO", "NORTHBOUNDDISOBEYCOUNT", "SOUTHBOUNDDISOBEYCOUNT", "PEDESTRIANINVOLVED", "CYCLISTINVOLVED", "MOTORCYCLISTINVOLVED", "ENVIRONMENTCONDITION2", "SELFREPORTED", "LASTEDITEDDATE", "CREATE_BY", "CREATE_DATE", "x", "y", "source_row", "year", "has_valid_coords", "distance_to_zone_m", "XMLIMPORTNOTES"]

df_updated = df.drop(columns=columns_to_del)
df_updated = df_updated[~df_updated.astype(str).apply(lambda x: x.str.lower()).isin(['other']).any(axis=1)]
df_updated['TRAFFICCONTROLCONDITION'] = df_updated['TRAFFICCONTROLCONDITION'].replace('unknown', 'Not Applicable')
df_updated['ACCIDENTDATE'] = pd.to_datetime(df_updated['ACCIDENTDATE'], format='mixed', errors='coerce')
df_updated['zone_id'] = df_updated['zone_id'].astype('category')
df_updated['zone_area_km2'] = df_updated['zone_area_km2'].astype('float64')

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_updated[['ACCIDENT_WEEKDAY']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ACCIDENT_WEEKDAY']), index=df_updated.index)
df_updated = pd.concat([df_updated, df_encoded], axis=1)
df_updated = df_updated.drop(columns='ACCIDENT_WEEKDAY')

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_updated[['ACCIDENTLOCATION']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ACCIDENTLOCATION']), index=df_updated.index)
df_updated = pd.concat([df_updated, df_encoded], axis=1)
df_updated = df_updated.drop(columns='ACCIDENTLOCATION')

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_updated[['LIGHT']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['LIGHT']), index=df_updated.index)
df_updated = pd.concat([df_updated, df_encoded], axis=1)
df_updated = df_updated.drop(columns='LIGHT')

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_updated[['ROADJURISDICTION']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ROADJURISDICTION']), index=df_updated.index)
df_updated = pd.concat([df_updated, df_encoded], axis=1)
df_updated = df_updated.drop(columns='ROADJURISDICTION')

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_updated[['TRAFFICCONTROL']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['TRAFFICCONTROL']), index=df_updated.index)
df_updated = pd.concat([df_updated, df_encoded], axis=1)
df_updated = df_updated.drop(columns='TRAFFICCONTROL')

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_updated[['TRAFFICCONTROLCONDITION']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['TRAFFICCONTROLCONDITION']), index=df_updated.index)
df_updated = pd.concat([df_updated, df_encoded], axis=1)
df_updated = df_updated.drop(columns='TRAFFICCONTROLCONDITION')

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_updated[['ENVIRONMENTCONDITION1']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ENVIRONMENTCONDITION1']), index=df_updated.index)
df_updated = pd.concat([df_updated, df_encoded], axis=1)
df_updated = df_updated.drop(columns='ENVIRONMENTCONDITION1')

df_updated.to_csv("Traffic_Collisions_Updated.csv")