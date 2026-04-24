from flask import Flask, render_template, request, jsonify
from data_fetcher import fetch_stock_data, get_popular_tickers
from model import run_full_pipeline

app = Flask(__name__)

@app.route("/")
def index():
    popular_tickers = get_popular_tickers()
    return render_template("index.html", popular_tickers=popular_tickers)

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form.get("ticker", "").upper()
    period = request.form.get("period", "2y")
    try:
        days = int(request.form.get("days", 30))
    except ValueError:
        days = 30
    
    popular_tickers = get_popular_tickers()
    
    if not ticker:
         return render_template("index.html", error="Please enter a ticker symbol.", popular_tickers=popular_tickers)

    data, info, error = fetch_stock_data(ticker, period)
    
    if error:
        return render_template("index.html", error=error, popular_tickers=popular_tickers)
        
    if len(data) < 60:
        return render_template("index.html", error="Not enough data for this ticker (min 60 days required for LSTM/XGBoost).", popular_tickers=popular_tickers)
        
    try:
        results = run_full_pipeline(ticker, data, days)
        
        predictions = results["forecast_df"]
        current_price = round(data['Close'].iloc[-1], 2)
        forecast_avg = round(sum([p['Predicted_Price'] for p in predictions]) / len(predictions), 2)
        change_pct = round(((forecast_avg - current_price) / current_price) * 100, 2)
        
        import requests
        try:
            _fx = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=2).json()
            usd_to_inr = _fx['rates']['INR']
        except:
            usd_to_inr = 84.0
            
        return render_template("result.html", 
                               ticker=ticker,
                               name=info.get('name', ticker),
                               currency=info.get('currency', 'USD'),
                               current_price=current_price,
                               forecast_avg=forecast_avg,
                               change_pct=change_pct,
                               usd_to_inr=usd_to_inr,
                               results=results)
    except Exception as e:
         return render_template("index.html", error=str(e), popular_tickers=popular_tickers)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
