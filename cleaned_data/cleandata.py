import pandas as pd
import os

def preprocess_and_extract_empty_likes(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='gbk')
    if 'Author' in df.columns:
        df = df.drop(columns=['Author'])
        print(f"'Author' column removed from {csv_file}")
    for column in ['likes', 'comment', 'collect', 'share']:
        if column in df.columns:
            df[column] = df[column].astype(str).str.replace('万', '*10000')
            df[column] = df[column].apply(lambda x: eval(x) if '*' in x else x)
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
    if 'share' in df.columns:
        df['share'] = df['share'].replace('分享', 0)
    if 'comment' in df.columns:
        df['comment'] = df['comment'].replace('抢首评', 0)
    if 'collect' in df.columns:
        df['collect'] = df['collect'].replace('收藏', 0)
    if 'likes' in df.columns:
        df['likes'] = df['likes'].replace('点赞', 0)
    empty_likes_df = df[df['likes'] == 0]
    title_urls = empty_likes_df['Title_URL'].tolist()
    base_name = os.path.splitext(csv_file)[0]
    output_txt_file = f"{base_name}_emptyurl.txt"
    
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for url in title_urls:
            f.write(f"{url}\n")

    print(f"Extracted {len(title_urls)} Title_URLs to {output_txt_file}")
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Processed CSV file saved: {csv_file}")

csv_files = [
    '/content/1000种中草植物.csv', '/content/NB8888NNNN.csv', '/content/一方见地.csv', 
    '/content/东北深山采药人.csv', '/content/中华本草堂（中医药）.csv', '/content/中草花.csv', 
    '/content/乌蒙本草。（山货代购）.csv', '/content/人物微纪实.csv', '/content/国学问道.csv', 
    '/content/大盛先生.csv', '/content/奇妙文化说.csv', '/content/小禾药坊.csv', 
    '/content/山里林里.csv', '/content/嵩山草本说.csv', '/content/本草中国.csv', 
    '/content/本草圣药.csv', '/content/本草新说.csv', '/content/朱鸟寻药记.csv', 
    '/content/植物之美.csv', '/content/汉昆堂滋补甄选.csv', '/content/海龙讲百草.csv', 
    '/content/独山县蒙氏草堂.csv', '/content/生态农人合作社.csv', '/content/痴百草.csv', 
    '/content/百晓参.csv', '/content/百草老农.csv','/content/神奇的草药.csv', 
    '/content/胖达.csv', '/content/艾柚动漫.csv', '/content/药材故事分享.csv', 
    '/content/识本草.csv', '/content/超哥说动漫.csv', '/content/青山眼里.csv', 
    '/content/鲁阳采药人.csv', '/content/黑苹果.csv'
]

for csv_file in csv_files:
    preprocess_and_extract_empty_likes(csv_file)

