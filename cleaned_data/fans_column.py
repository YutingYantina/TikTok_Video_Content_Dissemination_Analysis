import pandas as pd

def fans_column(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='gbk')

    if '粉丝' in df.columns:
        df['粉丝'] = df['粉丝'].astype(str).str.replace('万', '*10000')
        df['粉丝'] = df['粉丝'].apply(lambda x: eval(x) if '*' in x else x)
        df['粉丝'] = pd.to_numeric(df['粉丝'], errors='coerce').fillna(0).astype(int)
    
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Processed CSV file saved: {csv_file}")

csv_file = '/accounts_info.csv' 
fans_column(csv_file)
