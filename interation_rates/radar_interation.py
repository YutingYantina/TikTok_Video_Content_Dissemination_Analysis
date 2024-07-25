import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
noto_fonts = [f for f in font_paths if 'NotoSansCJK' in f]
font_path = noto_fonts[0]
prop = fm.FontProperties(fname=font_path)
file_path = '/content/account_interaction_rates.csv'
df = pd.read_csv(file_path)
print(df.head())
def plot_radar_chart(df, account_name):
    categories = ['平均点赞互动率', '平均评论互动率', '平均收藏互动率', '平均分享互动率']
    values = df.loc[df['账号名'] == account_name, categories].values.flatten().tolist()
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontproperties=prop)
    plt.title(account_name, fontproperties=prop, size=20, color='blue', y=1.1)
    plt.savefig(f'/content/radar_chart_{account_name}.png')
    plt.show()
for account_name in df['账号名']:
    plot_radar_chart(df, account_name)