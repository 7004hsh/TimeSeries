# ✅ AAPL 전용 GA 최적화 루틴 (하이퍼파라미터 저장 전용)
from curl_cffi import requests
import yfinance as yf
import pandas as pd
import numpy as np
import random
import datetime
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

# 1. AAPL 데이터 수집 (curl_cffi로 우회)
session = requests.Session(impersonate="chrome")
ticker = yf.Ticker("AAPL", session=session)
data = ticker.history(period="600wk")
close_prices = data[['Close']].dropna().copy()

# 2. 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

# 3. 시계열 샘플 생성 함수
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# 4. 모델 학습 및 평가
def evaluate_model(params, data):
    seq_len = params['seq_len']
    X, y = create_sequences(data, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        GRU(params['gru_units_1'], return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(params['dropout']),
        GRU(params['gru_units_2']),
        Dropout(params['dropout']),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=params['batch_size'], verbose=0)
    y_pred = model.predict(X_val, verbose=0)
    return mean_squared_error(y_val, y_pred)

# 5. 파라미터 공간 설정
param_grid = {
    'seq_len': [20, 30, 40],
    'gru_units_1': [32, 64, 128],
    'gru_units_2': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'batch_size': [16, 32],
    'learning_rate': [0.0005, 0.001, 0.005]
}

def random_individual():
    return {key: random.choice(values) for key, values in param_grid.items()}

def crossover(p1, p2):
    return {k: p1[k] if random.random() < 0.5 else p2[k] for k in p1}

def mutate(ind):
    k = random.choice(list(param_grid.keys()))
    ind[k] = random.choice(param_grid[k])
    return ind

def run_ga(data, generations=5, pop_size=6, elite_size=2, mutation_rate=0.2):
    population = [random_individual() for _ in range(pop_size)]
    for gen in range(generations):
        scored = [(ind, evaluate_model(ind, data)) for ind in population]
        scored.sort(key=lambda x: x[1])
        print(f"Gen {gen+1}: Best MSE = {scored[0][1]:.4f}")
        next_gen = [s[0] for s in scored[:elite_size]]
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(scored[:elite_size+2], 2)
            child = crossover(p1[0], p2[0])
            if random.random() < mutation_rate:
                child = mutate(child)
            next_gen.append(child)
        population = next_gen
    return scored[0][0]

# 6. 실행 및 저장
best_params = run_ga(scaled_data)

with open("best_hyperparams_final_aapl.json", "w") as f:
    json.dump(best_params, f)

print("\n✅ 최적 하이퍼파라미터 저장 완료: best_hyperparams_final_aapl.json")
print(best_params)
