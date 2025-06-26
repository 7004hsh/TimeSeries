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

# CSV 불러오기
df = pd.read_csv("aapl_lstm_enhanced_with_plot.csv")
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
df = df[df['predicted_price'].notna()].reset_index(drop=True)  # 이건 나중

# 이동평균 + 시그널
short_window = 20
long_window = 40
df['short_ma'] = df['predicted_price'].rolling(window=short_window, min_periods=1).mean()
df['long_ma'] = df['predicted_price'].rolling(window=long_window, min_periods=1).mean()
df['position'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
df['signal'] = df['position'].diff().fillna(0)

# 매수/매도 시점
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]
min_len = min(len(buy_signals), len(sell_signals))

buy_prices = buy_signals['predicted_price'].iloc[:min_len].values
sell_prices = sell_signals['predicted_price'].iloc[:min_len].values
buy_dates = buy_signals['date'].iloc[:min_len].dt.strftime('%Y-%m-%d').tolist()

# 수익률
returns = (sell_prices - buy_prices) / buy_prices
equity_curve = np.cumprod(1 + returns)
cumulative_return = equity_curve[-1] - 1

# 출력
print("📈 GDC 전략 누적 수익률 분석")
print(f"총 매매 횟수: {min_len}")
print(f"평균 단건 수익률: {returns.mean() * 100:.2f}%")
print(f"📊 누적 수익률: {cumulative_return * 100:.2f}%")

# 시각화
plt.figure(figsize=(14, 5))
plt.plot(equity_curve, marker='o', color='green', linewidth=2)
plt.title("📈 GDC 전략 누적 수익률 곡선 (매수 날짜 기준)", fontsize=14)
plt.xlabel("매수 날짜", fontsize=12)
plt.ylabel("누적 수익 배율 (배)", fontsize=12)
plt.xticks(ticks=np.arange(len(buy_dates)), labels=buy_dates, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
