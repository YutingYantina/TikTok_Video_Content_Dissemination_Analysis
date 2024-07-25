import pandas as pd
import os
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
account_info_path = '/content/accounts_info.csv'
accounts_info = pd.read_csv(account_info_path)
results = []
for account_name, csv_file in zip(account_names, csv_files):
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            df_selected = df[['likes', 'comment', 'collect', 'share']].dropna()
            fans_count = accounts_info.loc[accounts_info['账号名'] == account_name, '粉丝'].values[0]
            likes_interaction_rate = df_selected['likes'].mean() / fans_count
            comment_interaction_rate = df_selected['comment'].mean() / fans_count
            collect_interaction_rate = df_selected['collect'].mean() / fans_count
            share_interaction_rate = df_selected['share'].mean() / fans_count
            results.append([
                account_name,
                likes_interaction_rate,
                comment_interaction_rate,
                collect_interaction_rate,
                share_interaction_rate
            ])
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
results_df = pd.DataFrame(results, columns=[
    '账号名', '平均点赞互动率', '平均评论互动率', '平均收藏互动率', '平均分享互动率'
])
output_path = '/content/account_interaction_rates.csv'
results_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"结果已保存到 {output_path}")
