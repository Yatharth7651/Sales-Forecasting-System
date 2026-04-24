import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import warnings
warnings.filterwarnings("ignore")

class SaleForecaster:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.predictions = {}
        self.forecast_data = None
        self.feature_cols = []
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        df.sort_values('Date', inplace=True)
        
        # Engineering features
        for lag in [1, 3, 7]:
            df[f'Lag_{lag}'] = df['Close'].shift(lag)
        for window in [7, 14]:
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window).mean()
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def split_data(self, df, test_ratio=0.2):
        split_index = int(len(df) * (1 - test_ratio))
        exclude = ['Date','Close','Open','High','Low','Volume','Dividends','Stock Splits']
        self.feature_cols = [c for c in df.columns if c not in exclude]
        
        X = df[self.feature_cols]
        y = df['Close']
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    def train_models(self, X_train, y_train, X_train_scaled):
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        self.models['Linear Regression'] = ('scaled', lr)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = ('raw', rf)
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = ('raw', gb)
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        self.models['XGBoost'] = ('raw', xgb)

    def evaluate(self, X_test, y_test, X_test_scaled):
        # To maintain deterministic insertion order in dict, we rely on standard Python 3.7+ dict
        for name, (ftype, model) in self.models.items():
            X_input = X_test_scaled if ftype == 'scaled' else X_test
            preds = model.predict(X_input)
            mape = np.mean(np.abs((np.array(y_test) - preds) / (np.array(y_test) + 1e-10))) * 100
            self.results[name] = {
                "MAE": round(mean_absolute_error(y_test, preds), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
                "R2_Score": round(r2_score(y_test, preds), 4),
                "MAPE(%)": round(mape, 4)
            }
            self.predictions[name] = preds

    def get_best_model(self):
        best_name = max(self.results, key=lambda x: self.results[x]["R2_Score"])
        ftype, model = self.models[best_name]
        return best_name, model, ftype

    def forecast_future(self, df, days=30):
        best_name, best_model, ftype = self.get_best_model()
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
        
        future_predictions = []
        recent_closes = list(df['Close'].tail(max(14, 30)))
        
        for date in future_dates:
            rc = np.array(recent_closes)
            
            row = {
                'Lag_1': rc[-1],
                'Lag_3': rc[-3] if len(rc)>=3 else rc[-1],
                'Lag_7': rc[-7] if len(rc)>=7 else rc[-1],
                'Rolling_Mean_7': rc[-7:].mean() if len(rc)>=7 else rc.mean(),
                'Rolling_Mean_14': rc[-14:].mean() if len(rc)>=14 else rc.mean()
            }
            
            X_new = pd.DataFrame([row])[self.feature_cols]
            if ftype == 'scaled':
                X_new = self.scaler.transform(X_new)
                
            pred = best_model.predict(X_new)[0]
            future_predictions.append({"Date": date, "Predicted_Price": round(pred, 2)})
            recent_closes.append(pred)
            
        self.forecast_data = pd.DataFrame(future_predictions)
        return self.forecast_data, best_name

    def plot_forecast(self, df, ticker):
        fig = go.Figure()

        plot_df = df.tail(60).copy()
        plot_df = plot_df.sort_values('Date').reset_index(drop=True)
        plot_df['Date'] = pd.to_datetime(plot_df['Date']).dt.tz_localize(None)

        dates_list = plot_df['Date'].tolist()
        close_list = plot_df['Close'].tolist()
        open_list = plot_df['Open'].tolist()
        high_list = plot_df['High'].tolist()
        low_list = plot_df['Low'].tolist()

        fig.add_trace(go.Scatter(
            x=dates_list,
            y=close_list,
            mode='lines',
            name='Historical Trend',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=1.5),
            hoverinfo='skip'
        ))

        fig.add_trace(go.Candlestick(
            x=dates_list,
            open=open_list,
            high=high_list,
            low=low_list,
            close=close_list,
            name='Historical (OHLC)',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))

        if self.forecast_data is not None and not self.forecast_data.empty:
            forecast_dates = list(self.forecast_data['Date'])
            forecast_y = self.forecast_data['Predicted_Price'].tolist()

            if not plot_df.empty:
                last_hist_date = plot_df['Date'].iloc[-1]
                last_hist_price = plot_df['Close'].iloc[-1]
                forecast_dates.insert(0, last_hist_date)
                forecast_y.insert(0, last_hist_price)

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_y,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#f77f00', width=2),
                marker=dict(size=5)
            ))

            fig.add_vline(x=plot_df['Date'].iloc[-1], line_dash="dash", line_color="gray")

        fig.update_layout(
            title=dict(text=f"<b>{ticker} - Candlestick + Forecast</b>", x=0.01, y=0.95, font=dict(size=18, color="white")),
            template="plotly_dark", 
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Date", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title="Price", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=70, b=30),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(0,0,0,0.5)')
        )
        return pio.to_json(fig)


def run_lstm_vs_xgboost(ticker, data):
    close = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    
    SEQ = min(60, len(scaled) // 4)
    if SEQ < 10: SEQ = 10 
    
    X, y = [], []
    for i in range(SEQ, len(scaled)):
        X.append(scaled[i-SEQ:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = scaler.inverse_transform(xgb.predict(X_test).reshape(-1, 1)).flatten()

    # LSTM
    X_train_l = X_train.reshape(-1, SEQ, 1)
    X_test_l = X_test.reshape(-1, SEQ, 1)
    
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_l, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test_l, verbose=0)).flatten()

    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten().tolist()
    dates_test = data['Date'].values[SEQ + split:].tolist()
    
    xgb_pred = xgb_pred.tolist() if isinstance(xgb_pred, np.ndarray) else list(xgb_pred)
    lstm_pred = lstm_pred.tolist() if isinstance(lstm_pred, np.ndarray) else list(lstm_pred)

    # Metrics
    def calc(actual, pred):
        return {
            "MAE": round(mean_absolute_error(actual, pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(actual, pred)), 2),
            "R2": round(r2_score(actual, pred), 4),
            "MAPE": round(np.mean(np.abs((np.array(actual) - np.array(pred)) / (np.array(actual) + 1e-10))) * 100, 2)
        }
        
    xgb_m = calc(y_actual, xgb_pred)
    lstm_m = calc(y_actual, lstm_pred)

    # Chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates_test, y=y_actual, name="Actual", line=dict(color="#00b4d8", width=2)))
    fig.add_trace(go.Scatter(x=dates_test, y=xgb_pred, name="XGBoost", line=dict(color="#f77f00", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=dates_test, y=lstm_pred, name="LSTM", line=dict(color="#9b59b6", width=2, dash="dash")))

    fig.update_layout(
        template="plotly_dark", 
        height=500, 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        title=dict(
            text=f"<b>🎯 LSTM vs XGBoost — {ticker}</b>", 
            x=0.01, 
            y=0.95,
            font=dict(size=16, color="white")
        ),
        annotations=[dict(
            text="📈 Predictions vs Actual",
            xref="paper", yref="paper",
            x=0.5, y=0.98,
            showarrow=False,
            font=dict(size=15, color="#e5e7eb"),
            bgcolor="#374151",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            borderpad=6
        )],
        margin=dict(l=10, r=10, t=80, b=30),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(0,0,0,0.5)')
    )

    return pio.to_json(fig), xgb_m, lstm_m


def run_full_pipeline(ticker, data, days=30):
    forecaster = SaleForecaster()
    prepared_df = forecaster.prepare_data(data)
    
    # Train traditional models
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = forecaster.split_data(prepared_df)
    forecaster.train_models(X_train, y_train, X_train_scaled)
    forecaster.evaluate(X_test, y_test, X_test_scaled)
    
    # Forecast with Best
    forecast_df, best_name = forecaster.forecast_future(prepared_df, days)
    
    # Generate Forecast Plot
    forecast_plot_json = forecaster.plot_forecast(data, ticker)
    
    # Run LSTM vs XGBoost deep div
    comp_plot_json, xgb_m, lstm_m = run_lstm_vs_xgboost(ticker, data)
    
    return {
        "forecast_df": forecast_df.to_dict(orient="records"),
        "best_model": best_name,
        "model_results": forecaster.results,
        "forecast_plot_json": forecast_plot_json,
        "comp_plot_json": comp_plot_json,
        "xgb_metrics": xgb_m,
        "lstm_metrics": lstm_m
    }
