# ✅ 전체 통합 코드: GA 최적화 포함 주가 예측 + RSI/MACD 시각화 포함
from curl_cffi import requests
import yfinance as yf
import pandas as pd
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
from pyESN import ESN
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from scipy import stats
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 날짜 설정
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(weeks=600)

# ✅ yfinance 429 우회 세션 적용 - curl_cffi 기반 Chrome user-agent 세션을 활용하여 우회 다운로드
session = requests.Session(impersonate="chrome")

# 백업 경로
raw_data_path = "aapl_raw_backup.csv"

# 백업이 있고, 형식이 올바른 경우 로드
use_backup = False
if os.path.exists(raw_data_path):
    try:
        df_test = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
        if pd.to_numeric(df_test['Close'], errors='coerce').notnull().all():
            data = df_test
            use_backup = True
            print("✅ 백업된 원본 데이터 로드 완료.")
    except Exception as e:
        print("❌ 백업 파일 오류, 재다운로드합니다:", e)

if not use_backup:
    ticker = yf.Ticker("AAPL", session=session)
    data = ticker.history(period="600wk")
    if not data.empty:
        data.to_csv(raw_data_path)
        print("📦 원본 데이터 백업 저장 완료.")
    else:
        raise RuntimeError("❌ yfinance에서 데이터를 받아오지 못했습니다.")

close_prices = data[['Close']].copy()

# ✅ 기술적 지표 추가 (RSI, MACD)
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_MACD(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def plot_gru_vs_esn(real_val_close, pred_val_gru, pred_val_esn):
    plt.figure(figsize=(12, 6))
    plt.plot(real_val_close[-50:], label='📈 실제 종가', color='black', marker='o', linewidth=2)
    plt.plot(pred_val_gru[-50:], label='🧠 GRU 예측', linestyle='--', color='orange', marker='x', linewidth=2)
    plt.plot(pred_val_esn[-50:], label='🔁 ESN 예측', linestyle=':', color='blue', marker='s', linewidth=2)

    plt.title("최근 10주 예측 비교: GRU vs ESN (RSI+MACD)", fontsize=14)
    plt.xlabel("최근 날짜 순서", fontsize=12)
    plt.ylabel("가격 (USD)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(
        0.5, -0.02,
        f"GRU MSE={mean_squared_error(real_val_close[-50:], pred_val_gru[-50:]):.4f} | "
        f"ESN MSE={mean_squared_error(real_val_close[-50:], pred_val_esn[-50:]):.4f}",
        ha="center", fontsize=10
    )
    plt.show()

def run_esn_and_plot(scaled_data, seq_len, split, scaler_close, real_val_close, pred_val_close):
    # 시계열 샘플 생성 함수 중복 정의
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    # 데이터 생성
    X_esn, _ = create_sequences(scaled_data, seq_len)
    _, y_esn = create_sequences(scaled_close, seq_len)

    # reshape for ESN
    X_esn = X_esn.reshape(X_esn.shape[0], seq_len * scaled_data.shape[1])
    X_esn_train, X_esn_val = X_esn[:split], X_esn[split:]
    y_esn_train, y_esn_val = y_esn[:split], y_esn[split:]

    esn = ESN(
        n_inputs=seq_len * scaled_data.shape[1],
        n_outputs=1,
        n_reservoir=500,
        spectral_radius=0.95,
        sparsity=0.1,
        noise=0.0001,
        random_state=42
    )
    esn.fit(X_esn_train, y_esn_train)
    pred_val_esn = esn.predict(X_esn_val)
    pred_val_esn = np.clip(pred_val_esn, 0, 0.9)  # → 최대값이 완전히 1에 수렴하는 걸 방지
    pred_val_esn_close = scaler_close.inverse_transform(pred_val_esn.reshape(-1, 1))

    plot_gru_vs_esn(real_val_close, pred_val_close, pred_val_esn_close)
    return pred_val_esn_close

close_prices['RSI'] = compute_RSI(close_prices['Close'])
close_prices['MACD'], close_prices['MACD_signal'] = compute_MACD(close_prices['Close'])

# 이상치 클리핑 함수 정의 (1% 이하, 99% 이상 값을 절단하여 왜곡 방지)
def clip_outliers(series, lower_quantile=0.01, upper_quantile=0.99):
    q_low = series.quantile(lower_quantile)
    q_high = series.quantile(upper_quantile)
    return series.clip(lower=q_low, upper=q_high)

# RSI, MACD, Signal에 클리핑 적용
close_prices['RSI'] = clip_outliers(close_prices['RSI'])
close_prices['MACD'] = clip_outliers(close_prices['MACD'])
close_prices['MACD_signal'] = clip_outliers(close_prices['MACD_signal'])

# 1. 결측치가 없는 구간만 선택 (RSI, MACD, Signal 모두 포함)
valid_df = close_prices.dropna(subset=['RSI', 'MACD', 'MACD_signal']).copy()

# 2. 스케일러 초기화
scaler_close = MinMaxScaler()
scaler_rsi = MinMaxScaler()
scaler_macd = MinMaxScaler()
scaler_signal = MinMaxScaler()

# 3. 정규화 (정상 구간만)
scaled_close = scaler_close.fit_transform(valid_df[['Close']])
scaled_rsi = scaler_rsi.fit_transform(valid_df[['RSI']])
scaled_macd = scaler_macd.fit_transform(valid_df[['MACD']])
scaled_signal = scaler_signal.fit_transform(valid_df[['MACD_signal']])

# 4. 최종 통합된 입력 데이터 구성
scaled_data = np.hstack([scaled_close, scaled_rsi, scaled_macd, scaled_signal])

# 시계열 샘플 생성 함수
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# ✅ GA 최적화된 하이퍼파라미터 불러오기
ga_param_path = "best_hyperparams_final_aapl.json"
if not os.path.exists(ga_param_path):
    raise FileNotFoundError("GA 최적화 파라미터 파일이 존재하지 않습니다.")

with open(ga_param_path, 'r') as f:
    best_params = json.load(f)

seq_len = best_params['seq_len']
gru_units_1 = best_params['gru_units_1']
gru_units_2 = best_params['gru_units_2']
dropout = best_params['dropout']
batch_size = best_params['batch_size']
learning_rate = best_params['learning_rate']

X, _ = create_sequences(scaled_data, seq_len)
_, y = create_sequences(scaled_close, seq_len)
# GRU 입력을 위한 다차원 구조로 변경
X = X.reshape((X.shape[0], X.shape[1], scaled_data.shape[1]))

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 모델 구성 및 학습
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
model = Sequential([
    GRU(gru_units_1, return_sequences=True, input_shape=(seq_len, X.shape[2])),
    Dropout(dropout),
    GRU(gru_units_2),
    Dropout(dropout),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size,
                    validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

# 예측 및 역정규화
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# 예측값은 종가(Close)만 역정규화
pred_train_close = scaler_close.inverse_transform(pred_train)
pred_val_close = scaler_close.inverse_transform(pred_val)
real_train_close = scaler_close.inverse_transform(y_train[:pred_train.shape[0]].reshape(-1, 1))
real_val_close = valid_df['Close'].values.reshape(-1, 1)[-len(pred_val):]
pred_val = scaler_close.inverse_transform(pred_val)

# 기존 예측 시각화
window = 20
if len(pred_val) >= window:
    x_range = np.arange(window).reshape(-1, 1)
    y_range = pred_val[-window:]
    reg = LinearRegression()
    reg.fit(x_range, y_range)
    slope = reg.coef_[0][0]
    if slope > 0.5:
        trend = "강한 상승 📈"
    elif slope > 0.1:
        trend = "완만한 상승 ⬆️"
    elif slope < -0.5:
        trend = "강한 하강 📉"
    elif slope < -0.1:
        trend = "완만한 하강 ⬇️"
    else:
        trend = "횡보 ➖"
else:
    slope = 0
    trend = "데이터 부족"

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].plot(np.concatenate([real_train_close, real_val_close]), label='실제 종가', linewidth=2, color='black')
axs[0].plot(np.concatenate([pred_train_close, pred_val_close]), label='예측 종가', linewidth=2, linestyle='--', color='orange')
if len(pred_val) >= window:
    trend_line = reg.predict(x_range)
    axs[0].plot(
        np.arange(len(real_train_close) + len(real_val_close) - window, len(real_train_close) + len(real_val_close)),
        trend_line, label='예측 추세선', linestyle=':', color='red')
axs[0].set_title(f"전체 예측 (600주) - RSI + MACD 포함 | 예측 기반 추세: {trend} (기울기={slope:.4f})", fontsize=13)
axs[0].set_xlabel('시간 순서', fontsize=12)
axs[0].set_ylabel('가격 (USD)', fontsize=12)
axs[0].legend()
axs[0].grid(True)

axs[1].plot(real_val_close[-50:], label='실제 종가', marker='o', linewidth=2)
axs[1].plot(pred_val_close[-50:], label='예측 종가', marker='o', linewidth=2, linestyle='--')
axs[1].set_title("최근 10주 예측", fontsize=13)
axs[1].set_xlabel("최근 날짜 순서", fontsize=12)
axs[1].set_ylabel("가격 (USD)", fontsize=12)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.figtext(0.5, -0.05, "✅ 입력 features: [Close, RSI, MACD, MACD Signal]", ha="center", fontsize=10)
plt.show()

# ✅ 비교 실험: Close-only 입력 버전으로 학습 및 시각화
X_base, _ = create_sequences(scaled_close, seq_len)
_, y_base = create_sequences(scaled_close, seq_len)
X_base = X_base.reshape((X_base.shape[0], X_base.shape[1], 1))
X_base_train, X_base_val = X_base[:split], X_base[split:]
y_base_train, y_base_val = y_base[:split], y_base[split:]

model_base = Sequential([
    GRU(gru_units_1, return_sequences=True, input_shape=(seq_len, 1)),
    Dropout(dropout),
    GRU(gru_units_2),
    Dropout(dropout),
    Dense(1)
])
model_base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
model_base.fit(X_base_train, y_base_train, epochs=100, batch_size=batch_size,
               validation_data=(X_base_val, y_base_val), callbacks=[early_stop], verbose=0)

pred_val_base = model_base.predict(X_base_val)
pred_val_base_close = scaler_close.inverse_transform(pred_val_base)

model_lstm = Sequential([
    LSTM(gru_units_1, return_sequences=True, input_shape=(seq_len, X.shape[2])),
    Dropout(dropout),
    LSTM(gru_units_2),
    Dropout(dropout),
    Dense(1)
])
model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=100,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ✅ LSTM 예측 결과
pred_val_lstm = model_lstm.predict(X_val)
pred_val_lstm_close = scaler_close.inverse_transform(pred_val_lstm)

# ✅ ESN 예측도 여기서 받아옴 (함수 실행)
pred_val_esn_close = run_esn_and_plot(scaled_data, seq_len, split, scaler_close, real_val_close, pred_val_close)

# ✅ 최종 비교 시각화 (최근 10주)
plt.figure(figsize=(12, 6))
plt.plot(real_val_close[-50:], label='📈 실제 종가', marker='o', linewidth=2)
plt.plot(pred_val_close[-50:], label='🧠 GRU 예측', linestyle='--', marker='x', linewidth=2)
plt.plot(pred_val_lstm_close[-50:], label='🔮 LSTM 예측', linestyle='-.', marker='^', linewidth=2)
plt.plot(pred_val_esn_close[-50:], label='🔁 ESN 예측', linestyle=':', marker='s', linewidth=2)
plt.title("최근 10주 예측 비교: GRU vs LSTM vs ESN", fontsize=14)
plt.xlabel("최근 날짜 순서", fontsize=12)
plt.ylabel("가격 (USD)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# 예측 결과 저장
pred_all = model.predict(X)
pred_all_close = scaler_close.inverse_transform(pred_all)
real_all = scaler_close.inverse_transform(y.reshape(-1, 1))

result_df = close_prices.copy().reset_index()
result_df.columns = ['date', 'real_price', 'RSI', 'MACD', 'MACD_signal']
result_df['predicted_price'] = np.nan
n_pred = pred_all_close.flatten().shape[0]
result_df.loc[seq_len:seq_len + n_pred - 1, 'predicted_price'] = pred_all_close.flatten()
result_df['trend'] = trend
result_df['slope'] = round(float(slope), 4)

# 슬라이딩 윈도우로 여러 개의 MSE 계산
def sliding_mse(real, pred, window=10, step=10):
    mse_list = []
    for i in range(0, len(real) - window + 1, step):
        mse = mean_squared_error(real[i:i+window], pred[i:i+window])
        mse_list.append(mse)
    return np.array(mse_list)

gru_mse = sliding_mse(real_val_close, pred_val_close, window=10, step=10)
lstm_mse = sliding_mse(real_val_close, pred_val_lstm_close, window=10, step=10)

# 1. 두 샘플의 t-검정 (두 표본 평균 차이에 대한 검정)
t_stat, p_value = stats.ttest_ind(gru_mse, lstm_mse)

# 2. p-value 출력
print(f"t-검정 통계량 (t-statistic): {t_stat}")
print(f"p-value: {p_value}")

# 3. p-value 기준으로 결과 해석
alpha = 0.05  # 유의수준 (5%)
if p_value < alpha:
    print("대립가설을 채택합니다: 두 모델의 성능에 유의미한 차이가 있습니다.")
else:
    print("귀무가설을 채택합니다: 두 모델의 성능에 유의미한 차이가 없습니다.")

result_df.to_csv("aapl_lstm_enhanced_with_plot.csv", index=False, encoding='utf-8-sig')
print("✅ 전체 예측 포함 CSV 저장 완료: 'aapl_lstm_enhanced_with_plot.csv'")
print(f"📈 예측 기반 추세: {trend} (기울기={slope:.4f})")
# 학습 및 검증 예측 결과가 각각 pred_train, pred_val에 저장된 상태
mse_train, rmse_train, mae_train = mean_squared_error(real_train_close, pred_train_close), np.sqrt(mean_squared_error(real_train_close, pred_train_close)), mean_absolute_error(real_train_close, pred_train_close)
mse_val, rmse_val, mae_val = mean_squared_error(real_val_close, pred_val_close), np.sqrt(mean_squared_error(real_val_close, pred_val_close)), mean_absolute_error(real_val_close, pred_val_close)
print(f"📊 학습 MSE: {mse_train:.4f}, 검증 MSE: {mse_val:.4f}")

