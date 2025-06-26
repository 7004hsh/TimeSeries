import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. CSV 불러오기
df = pd.read_csv("aapl_lstm_enhanced_with_plot.csv")
df['date'] = pd.to_datetime(df['date'])

# 2. 이동평균 고정 설정 (20일 / 40일)
short_window = 20
long_window = 40

df = df[df['predicted_price'].notna()].reset_index(drop=True)
df['short_ma'] = df['predicted_price'].rolling(window=short_window, min_periods=1).mean()
df['long_ma'] = df['predicted_price'].rolling(window=long_window, min_periods=1).mean()
df['position'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
df['signal'] = df['position'].diff().fillna(0)

# 3. 매수/매도 시점
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]

# 4. 📈 예측 + MA + GDC 시각화
plt.figure(figsize=(16, 6))
plt.plot(df['date'], df['predicted_price'], label='예측 종가', color='black', linewidth=2)
plt.plot(df['date'], df['short_ma'], label='단기 MA (20일)', linestyle='--', color='dodgerblue', linewidth=2, alpha=0.7)
plt.plot(df['date'], df['long_ma'], label='장기 MA (60일)', linestyle='--', color='darkorange', linewidth=2, alpha=0.7)
plt.scatter(buy_signals['date'], buy_signals['predicted_price'], marker='^', color='green', label='매수 시점', s=100, zorder=5)
plt.scatter(sell_signals['date'], sell_signals['predicted_price'], marker='v', color='red', label='매도 시점', s=100, zorder=5)
plt.title(f"[예측 기반] GDC 전략 시각화 (단기 20 / 장기 60)", fontsize=14)
plt.xlabel("날짜", fontsize=12)
plt.ylabel("예측 종가 (USD)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 📊 GDC 추세 시각화 (인덱스 제거, 연도 기준만)
gdc_trend = np.where(df['signal'] == 1, 1, np.where(df['signal'] == -1, -1, 0))

plt.figure(figsize=(16, 4))
plt.plot(df['date'], gdc_trend, drawstyle='steps-post', color='purple', linewidth=2)
plt.title("GDC 추세 시각화 (+1=매수, -1=매도, 0=유지)", fontsize=13)
plt.xlabel("날짜")
plt.ylabel("신호 값")
plt.grid(True)

# 연도 구분선 표시만 (파란 인덱스 제거됨)
last_year = None
for idx, date in enumerate(df['date']):
    year = date.year
    if year != last_year:
        plt.axvline(x=date, color='gray', linestyle='--', linewidth=0.5)
        last_year = year

plt.tight_layout()
plt.show()

# 6. 🧾 매매 전략 요약표
summary = pd.DataFrame({
    "Buy Index": buy_signals.index,
    "Buy Price": buy_signals['predicted_price'].values,
})
min_len = min(len(buy_signals), len(sell_signals))
summary = summary.iloc[:min_len]
summary["Sell Index"] = sell_signals.index[:min_len]
summary["Sell Price"] = sell_signals['predicted_price'].values[:min_len]
summary["Return (%)"] = ((summary["Sell Price"] - summary["Buy Price"]) / summary["Buy Price"] * 100).round(2)
summary.reset_index(drop=True, inplace=True)

print("📋 GDC 매매 전략 요약표:")
print(summary.head(10))
print("전체 CSV 행 수:", len(df))
print("NaN 제거 후 사용 중인 데이터 수:", df[['predicted_price', 'short_ma', 'long_ma']].dropna().shape[0])
print("GDC 신호가 발생한 데이터 수:", (df['signal'] != 0).sum())
