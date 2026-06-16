import pandas as pd

path = 'backend/E Commerce Dataset.xlsx'
xl = pd.ExcelFile(path)
print('SHEETS', xl.sheet_names)
for sheet in xl.sheet_names:
    df = pd.read_excel(path, sheet_name=sheet)
    print('\nSHEET', sheet, 'shape=', df.shape)
    print(df.head(15).to_string())
