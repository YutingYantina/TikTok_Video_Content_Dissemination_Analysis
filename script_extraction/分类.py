import pandas as pd
file_path = '/Users/fengliyinghua/Desktop/intern/extract_texts_week4/alldata copy 2.csv'
df = pd.read_csv(file_path)
filtered_df = df[df['Title'].str.contains('冬虫夏草', na=False)]
row_count = len(filtered_df)
output_file_path = f'{row_count}_冬虫夏草.csv'
filtered_df.to_csv(output_file_path, index=False)
print(f"筛选后的数据已输出到 {output_file_path}")