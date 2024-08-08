import pandas as pd
from dfpl import fingerprint as fp

input_files = [
    "data/input_datasets/tox24_challenge/tox24_challenge_train.csv",
    "data/input_datasets/tox24_challenge/tox24_challenge_test.csv",
    "data/input_datasets/tox24_challenge/ttr-supplemental-tables_s4.csv"
]

for file in input_files:
    df = fp.importDataFile(file,
                           import_function=fp.importCSV,
                           fp_size=2048)
    df_fp = df[df['fp'].notnull()]
    df_fp = df_fp['fp'].apply(lambda row: [1 if el == True else 0 for el in row])
    df_fp = df_fp.apply(pd.Series)
    df_fp = df_fp.rename(columns=lambda x: 'fp_' + str(x))
    df = df.drop('fp', axis=1)
    df_final = pd.concat([df, df_fp], axis=1)
    o_file = file.replace('.csv', '_fp.csv')
    df_final.to_csv(o_file, index=False)
