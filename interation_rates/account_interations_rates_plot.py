import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
noto_fonts = [f for f in font_paths if 'NotoSansCJK' in f]
font_path = noto_fonts[0]
prop = fm.FontProperties(fname=font_path)
file_path = '/content/account_interaction_rates.csv'
df = pd.read_csv(file_path)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题
def plot_interaction_rates(df):
    accounts = df['账号名']
    likes_rate = df['平均点赞互动率']
    comment_rate = df['平均评论互动率']
    collect_rate = df['平均收藏互动率']
    share_rate = df['平均分享互动率']
    plt.figure(figsize=(12, 8))
    plt.plot(accounts, likes_rate, label='平均点赞互动率', marker='o')
    plt.plot(accounts, comment_rate, label='平均评论互动率', marker='o')
    plt.plot(accounts, collect_rate, label='平均收藏互动率', marker='o')
    plt.plot(accounts, share_rate, label='平均分享互动率', marker='o')
    plt.xlabel('账号名', fontproperties=prop)
    plt.ylabel('互动率', fontproperties=prop)
    plt.title('各账号的平均互动率', fontproperties=prop)
    plt.xticks(rotation=90, fontproperties=prop)
    plt.legend(prop=prop)
    plt.tight_layout()
    plt.savefig('/content/account_interaction_rates_plot.png')
    plt.show()
plot_interaction_rates(df)