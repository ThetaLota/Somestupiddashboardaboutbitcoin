import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Market Sentiment & Forecasting", layout="wide")
st.title("📈 Algorithmic Market Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Data Parameters")
asset_choice = st.sidebar.selectbox("Select Asset to Analyze", ("FMETF.PS", "BTC-USD", "ALI.PS", "BDO.PS"))
timeframe = st.sidebar.slider("Historical Data (Days)", min_value=30, max_value=365, value=180)

st.sidebar.header("Quantitative Indicators")
show_sma = st.sidebar.checkbox("Show Simple Moving Average (SMA)", value=True)
sma_window = st.sidebar.slider("SMA Window (Days)", 5, 200, 20)

show_rsi = st.sidebar.checkbox("Show Relative Strength Index (RSI)", value=True)
rsi_window = st.sidebar.slider("RSI Window (Days)", 5, 30, 14)

# --- Data Fetching Function ---
@st.cache_data 
def load_data(ticker, days):
    asset = yf.Ticker(ticker)
    df = asset.history(period=f"{days}d")
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df

data = load_data(asset_choice, timeframe)

# --- Algorithmic Calculations ---
if show_sma:
    # Calculates the rolling average
    data[f'SMA_{sma_window}'] = data['Close'].rolling(window=sma_window).mean()

if show_rsi:
    # Calculates the RSI momentum oscillator
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

# --- Main Price Chart (Candlesticks + SMA) ---
st.subheader("Price Action & Trend Analysis")
fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name="Price")])

# Overlay SMA if toggled
if show_sma:
    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'SMA_{sma_window}'], 
                             mode='lines', name=f'SMA {sma_window}', 
                             line=dict(color='orange', width=2)))

fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- Secondary Chart (RSI Momentum) ---
if show_rsi:
    st.subheader("Momentum Analysis (RSI)")
    rsi_fig = go.Figure(go.Scatter(x=data['Date'], y=data['RSI'], 
                                   mode='lines', name='RSI', 
                                   line=dict(color='cyan', width=2)))
    
    # Add Overbought/Oversold thresholds
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    rsi_fig.update_layout(height=250, template="plotly_dark", yaxis=dict(range=[0, 100]))
    st.plotly_chart(rsi_fig, use_container_width=True)

# --- Backtesting: RSI Strategy Simulation ---
st.markdown("---")
st.subheader("Backtesting: Does the RSI Strategy Work?")
st.write("This simulates buying when RSI < 30 and selling when RSI > 70, compared to just buying and holding.")

if show_rsi:
    # 1. Generate Trading Signals
    # 1 = Buy (Long), -1 = Sell (Short), 0 = Hold
    data['Signal'] = 0
    data.loc[data['RSI'] < 30, 'Signal'] = 1  
    data.loc[data['RSI'] > 70, 'Signal'] = -1 
    
    # 2. Track Position (Forward fill signals to maintain the position until a new signal)
    data['Position'] = data['Signal'].replace(0, pd.NA).ffill().fillna(0)
    
    # 3. Calculate Returns
    data['Market_Returns'] = data['Close'].pct_change()
    # Strategy returns are based on the position of the previous day
    data['Strategy_Returns'] = data['Market_Returns'] * data['Position'].shift(1)
    
    # 4. Calculate Cumulative Returns (Starting at 1.0 or 100%)
    data['Cumulative_Market'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()
    
    # 5. Plot the Results
    backtest_fig = go.Figure()
    backtest_fig.add_trace(go.Scatter(x=data['Date'], y=data['Cumulative_Market'], 
                                      mode='lines', name='Buy & Hold (Market)', 
                                      line=dict(color='gray', width=2)))
    backtest_fig.add_trace(go.Scatter(x=data['Date'], y=data['Cumulative_Strategy'], 
                                      mode='lines', name='RSI Strategy', 
                                      line=dict(color='green', width=2)))
    
    backtest_fig.update_layout(height=400, template="plotly_dark", 
                               yaxis_title="Return Multiplier (1.0 = Breakeven)")
    st.plotly_chart(backtest_fig, use_container_width=True)
else:
    st.warning("Please enable the RSI indicator in the sidebar to run the backtest.")

import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Qualitative Engine: NLP News Sentiment ---
st.markdown("---")
st.header("📰 Qualitative Analysis (Market Sentiment)")

@st.cache_data(ttl=3600) # Caches the news for 1 hour so we don't spam the server
def get_news_sentiment(query):
    # Fetch news from Google News RSS
    url = f"https://news.google.com/rss/search?q={query}+stock+finance&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features="xml")
    
    # Extract headlines
    articles = soup.findAll("item")
    headlines = [a.title.text for a in articles[:10]] # Get the top 10 recent headlines
    
    # Analyze Sentiment using VADER
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for headline in headlines:
        score = analyzer.polarity_scores(headline)
        # We use the 'compound' score which ranges from -1 (Extremely Negative) to 1 (Extremely Positive)
        sentiment_scores.append(score['compound'])
        
    # Calculate average sentiment
    avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return headlines, sentiment_scores, avg_score

# Run the scraper for the currently selected asset
st.write(f"Scraping the latest financial news for **{asset_choice}**...")

# Remove the '.PS' for local stocks so the search query works better
search_term = asset_choice.replace(".PS", "") 
headlines, scores, avg_sentiment = get_news_sentiment(search_term)

# --- Display the Fear/Greed Sentiment Gauge ---
# Convert the -1 to 1 score into a 0 to 100 gauge
gauge_score = (avg_sentiment + 1) * 50 

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=gauge_score,
    title={'text': "Market Sentiment (Fear & Greed Index)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "white"},
        'steps': [
            {'range': [0, 40], 'color': "red"},      # Fear
            {'range': [40, 60], 'color': "gray"},    # Neutral
            {'range': [60, 100], 'color': "green"}   # Greed
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': gauge_score
        }
    }
))

fig_gauge.update_layout(height=300, template="plotly_dark")
st.plotly_chart(fig_gauge, use_container_width=True)

# --- Display the Scraped Headlines ---
with st.expander("View Scraped Headlines & Individual Scores"):
    for i in range(len(headlines)):
        # Color code the text based on sentiment
        if scores[i] > 0.1:
            st.success(f"**Score: {scores[i]:.2f}** | {headlines[i]}")
        elif scores[i] < -0.1:
            st.error(f"**Score: {scores[i]:.2f}** | {headlines[i]}")
        else:
            st.info(f"**Score: {scores[i]:.2f}** | {headlines[i]}")

