import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker_symbol, period="2y"):
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None, None, "No data found. Check ticker symbol."
        data.reset_index(inplace=True)
        
        info = ticker.fast_info
        try:
            company_info = {
                'name': ticker.info.get('longName', ticker_symbol),
                'currency': ticker.info.get('currency', 'USD'),
            }
        except:
             company_info = {
                'name': ticker_symbol,
                'currency': 'USD',
            }
        return data, company_info, None
    except Exception as e:
        return None, None, str(e)

def get_popular_tickers():
    return {
        'AAPL': 'Apple Inc.',         'GOOGL': 'Alphabet (Google)',
        'MSFT': 'Microsoft Corp.',    'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',         'META': 'Meta (Facebook)',
        'NVDA': 'NVIDIA Corp.',       'JPM': 'JPMorgan Chase',
        'WMT': 'Walmart Inc.',        'NFLX': 'Netflix Inc.',
        'TCS.NS': 'TCS (India)',      'RELIANCE.NS': 'Reliance (India)',
        'INFY.NS': 'Infosys (India)', 'HDFCBANK.NS': 'HDFC Bank (India)',
        'WIPRO.NS': 'Wipro (India)',  'ITC.NS': 'ITC Ltd (India)',
        'SBIN.NS': 'SBI (India)',     'TATAMOTORS.NS': 'Tata Motors (India)'
    }
