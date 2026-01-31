from nba_api.stats.library.parameters import LeagueID
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import nfl_data_py as nfl
from nba_api.stats.endpoints import leaguedashteamstats

def scrape_wnba_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # 寻找数据表格 (需根据具体网页结构调整)
        table = soup.find('table') 
        df = pd.read_html(str(table))[0]
        return df
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

# 注意：请遵循 robots.txt 并设置合理的请求间隔（time.sleep）

# --- 1. 获取 NFL 数据（示例：获取 2023 赛季球员/球队基本数据） ---
print("正在获取 NFL 数据...")
nfl_df = nfl.import_weekly_data([2023])
# 金融/商分同学关注：可以通过统计得分波动来评估“竞争平衡”
# 注意：数据是球员级别的，使用 'recent_team' 分组，并用 'fantasy_points' 作为得分代理
nfl_balance = nfl_df.groupby('recent_team')['fantasy_points'].std()

# --- 2. 获取 WNBA 数据（Problem D 的核心） ---
print("正在获取 WNBA 数据...")
try:
    wnba_stats = leaguedashteamstats.LeagueDashTeamStats(league_id_nullable='10')
    wnba_df = wnba_stats.get_data_frames()[0]
    wnba_success = True
except Exception as e:
    print(f"WNBA 数据获取失败：{e}")
    wnba_df = None
    wnba_success = False

# --- 3. 简单的商业分析逻辑示例：胜率与某种指标的相关性 ---
# 假设我们要看 WNBA 球队的场均得分(PTS)与胜率(W_PCT)的关系
# 这可以作为你们“球队成功模型”的基础
analysis_nfl = nfl_balance.reset_index().rename(columns={'recent_team': 'TEAM', 'fantasy_points': 'SCORE_STD'})
print("\nNFL 2023 赛季球队得分波动性：")
print(analysis_nfl.head())

if wnba_success:
    analysis = wnba_df[['TEAM_NAME', 'W_PCT', 'PTS', 'REB', 'AST']].sort_values(by='W_PCT', ascending=False)
    print("\nWNBA 2023 赛季球队表现速览：")
    print(analysis.head())
else:
    analysis = None
    print("\nWNBA 数据获取失败，跳过分析。")

# --- 4. 网页抓取额外 WNBA 数据 ---
print("\n正在抓取额外 WNBA 数据...")
wnba_url = "https://www.wnba.com/stats/team-stats/"  # 示例 URL，需根据实际页面调整
try:
    scraped_df = scrape_wnba_data(wnba_url)
    if scraped_df is not None:
        print("抓取成功，数据预览：")
        print(scraped_df.head())
    else:
        print("抓取失败：未找到数据。")
        scraped_df = None
except Exception as e:
    print(f"抓取失败：{e}")
    scraped_df = None

# --- 5. 导出数据供全队共享（金融/商分同学可以用 Excel 进一步分析） ---
analysis_nfl.to_csv('nfl_balance_analysis.csv', index=False)
if wnba_success:
    analysis.to_csv('wnba_market_analysis.csv', index=False)
if scraped_df is not None:
    scraped_df.to_csv('wnba_scraped_data.csv', index=False)
print("\n数据已保存到 CSV 文件：nfl_balance_analysis.csv" + (", wnba_market_analysis.csv" if wnba_success else "") + (", wnba_scraped_data.csv" if scraped_df is not None else ""))