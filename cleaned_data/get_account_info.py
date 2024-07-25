import pandas as pd
import os

def get_account_info(csv_files, account_names, output_file):
    columns = ['账号名', '关注', '粉丝', '获赞', '抖音号', 'IP属地', '年龄', '地址', '院校', '个性签名', '作品数', '账号特征']
    data = []
    for csv_file, account_name in zip(csv_files, account_names):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='gbk')
        works_count = df.shape[0] 
        total_likes = df['likes'].sum()
        data.append([account_name, '', '', total_likes, '', '', '', '', '', '', works_count, ''])
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file, encoding='utf-8')
    else:
        existing_df = pd.DataFrame(columns=columns)
    result_df = pd.DataFrame(data, columns=columns)
    updated_df = pd.concat([existing_df, result_df]).drop_duplicates(subset=['账号名']).reset_index(drop=True)
    updated_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Account information has been updated in {output_file}")
    
csv_files = [
    '/content/1000种中草植物.csv', '/content/NB8888NNNN.csv', '/content/一方见地.csv', 
    '/content/东北深山采药人.csv', '/content/中华本草堂（中医药）.csv', '/content/中草花.csv', 
    '/content/乌蒙本草。（山货代购）.csv', '/content/人物微纪实.csv', '/content/国学问道.csv', 
    '/content/大盛先生.csv', '/content/奇妙文化说.csv', '/content/小禾药坊.csv', 
    '/content/山里林里.csv', '/content/嵩山草本说.csv', '/content/本草中国.csv', 
    '/content/本草圣药.csv', '/content/本草新说.csv', '/content/朱鸟寻药记.csv', 
    '/content/植物之美.csv', '/content/汉昆堂滋补甄选.csv', '/content/海龙讲百草.csv', 
    '/content/独山县蒙氏草堂.csv', '/content/生态农人合作社.csv', '/content/痴百草.csv', 
    '/content/百晓参.csv', '/content/百草老农.csv', '/content/神奇的草药.csv', 
    '/content/胖达视界.csv', '/content/艾柚动漫.csv', '/content/药材故事分享.csv', 
    '/content/识本草.csv', '/content/超哥说动漫.csv', '/content/青山眼里.csv', 
    '/content/鲁阳采药人.csv', '/content/黑苹果视界.csv'
]
account_names = [
    '1000种中草植物', 'NB8888NNNN', '一方见地', '东北深山采药人', '中华本草堂（中医药）', '中草花',
    '乌蒙本草。（山货代购）', '人物微纪实', '国学问道', '大盛先生', '奇妙文化说', '小禾药坊',
    '山里林里', '嵩山草本说', '本草中国', '本草圣药', '本草新说', '朱鸟寻药记',
    '植物之美', '汉昆堂滋补甄选', '海龙讲百草', '独山县蒙氏草堂', '生态农人合作社', '痴百草',
    '百晓参', '百草老农', '神奇的草药', '胖达视界', '艾柚动漫', '药材故事分享',
    '识本草', '超哥说动漫', '青山眼里', '鲁阳采药人', '黑苹果视界'
]
output_file = '/accounts_info.csv'
get_account_info(csv_files, account_names, output_file)
