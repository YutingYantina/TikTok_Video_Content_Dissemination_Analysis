import pandas as pd
import os

def extract_empty_likes(csv_file):
    df = pd.read_csv(csv_file)
    empty_likes_df = df[df['likes'].isnull()]
    title_urls = empty_likes_df['Title_URL'].tolist()
    base_name = os.path.splitext(csv_file)[0]
    output_txt_file = f"{base_name}_emptyurl.txt"
    with open(output_txt_file, 'w') as f:
        for url in title_urls:
            f.write(f"{url}\n")
    print(f"Extracted {len(title_urls)} Title_URLs to {output_txt_file}")


csv_files = ['1000种中草植物.csv', 'NB8888NNNN.csv', '一方见地.csv', '东北深山采药人.csv', '中华本草堂（中医药）.csv', '中草花.csv', '乌蒙本草。（山货代购）.csv', '人物微纪实.csv', '国学问道.csv', '大盛先生.csv', '奇妙文化说.csv', '小禾药坊.csv', '山里林里.csv', '嵩山草本说.csv', '本草中国.csv', '本草圣药.csv', '本草新说.csv', '朱鸟寻药记.csv', '植物之美.csv', '汉昆堂滋补甄选.csv', '海龙讲百草.csv', '独山县蒙氏草堂 .csv', '生态农人合作社 .csv', '痴百草.csv', '百晓参.csv', '百草老农.csv', '神奇的草药 .csv', '胖达.csv', '艾柚动漫.csv', '药材故事分享.csv', '识本草.csv', '超哥说动漫.csv', '青山眼里.csv', '鲁阳采药人.csv', '黑苹果.csv']
for csv_file in csv_files:
    extract_empty_likes(csv_file)
