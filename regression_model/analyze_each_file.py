import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc' #use chinese
prop = fm.FontProperties(fname=font_path)

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

def analyze_each_file(file_path):
    df = pd.read_csv(file_path)
    df_selected = df[['likes', 'comment', 'collect', 'share']].dropna()
    X = df_selected[['comment', 'collect', 'share']]
    y = df_selected['likes']
    model = LinearRegression()
    model.fit(X, y)
    coefficients = model.coef_
    intercept = model.intercept_
    print(f"文件: {os.path.basename(file_path)}")
    print(f"截距（Intercept）：{intercept}")
    print(f"系数（Coefficients）：")
    print(f"comment（评论）：{coefficients[0]}")
    print(f"collect（收藏）：{coefficients[1]}")
    print(f"share（分享）：{coefficients[2]}")
    print()
    plt.figure(figsize=(10, 6))
    plt.scatter(df_selected['comment'], y, color='blue', label='Comment')
    plt.scatter(df_selected['collect'], y, color='green', label='Collect')
    plt.scatter(df_selected['share'], y, color='red', label='Share')
    y_pred = model.predict(X)
    plt.plot(df_selected['comment'], y_pred, color='blue', linewidth=2)
    plt.plot(df_selected['collect'], y_pred, color='green', linewidth=2)
    plt.plot(df_selected['share'], y_pred, color='red', linewidth=2)
    plt.title(f"回归分析: {os.path.basename(file_path)}", fontproperties=prop)
    plt.xlabel('特征', fontproperties=prop)
    plt.ylabel('点赞', fontproperties=prop)
    plt.legend(prop=prop)
    plt.grid(True)
    plt.savefig(f"/content/regression_{os.path.basename(file_path)}.png", bbox_inches='tight')
    plt.show()


for file_path in csv_files:
    if os.path.exists(file_path):
        try:
            analyze_each_file(file_path)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")
