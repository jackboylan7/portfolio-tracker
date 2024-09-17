import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page config
st.set_page_config(page_title="BEZZ Advanced Multi-Portfolio Stock Tracker", layout="wide")

# Custom CSS for modern UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size: 40px !important;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: Black;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTextInput>div>div>input {
        background-color: Black;
        border-radius: 10px;
    }
    .stNumberInput>div>div>input {
        background-color: #F3F4F6;
        border-radius: 10px;
    }
    .stock-card {
        background-color: black;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<p class='big-font'>BEZZ Advanced Multi-Portfolio Stock Tracker</p>", unsafe_allow_html=True)

# Initialize session state for portfolios
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = {}

if 'current_portfolio' not in st.session_state:
    st.session_state.current_portfolio = None

# Function to fetch S&P 500 data
@st.cache_data
def get_sp500_data(start_date, end_date):
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)
    return sp500['Close']

# Sidebar for portfolio management
st.sidebar.header("Portfolio Management")

# Create new portfolio
new_portfolio_name = st.sidebar.text_input("Create a new portfolio")
if st.sidebar.button("Create Portfolio"):
    if new_portfolio_name and new_portfolio_name not in st.session_state.portfolios:
        st.session_state.portfolios[new_portfolio_name] = {}
        st.session_state.current_portfolio = new_portfolio_name
        st.sidebar.success(f"Created portfolio: {new_portfolio_name}")
    elif new_portfolio_name in st.session_state.portfolios:
        st.sidebar.error("Portfolio with this name already exists.")
    else:
        st.sidebar.error("Please enter a portfolio name.")

# Select portfolio
st.sidebar.subheader("Select Portfolio")
portfolio_options = ["All"] + list(st.session_state.portfolios.keys())
selected_portfolio = st.sidebar.selectbox("Choose a portfolio", options=portfolio_options)
if selected_portfolio != "All":
    st.session_state.current_portfolio = selected_portfolio
else:
    st.session_state.current_portfolio = "All"

# Add stocks to the selected portfolio
if st.session_state.current_portfolio and st.session_state.current_portfolio != "All":
    st.sidebar.subheader(f"Add Stocks to {st.session_state.current_portfolio}")
    new_stock = st.sidebar.text_input("Add a new stock (e.g., AAPL)")
    shares = st.sidebar.number_input("Number of shares", min_value=1, value=1)
    purchase_price = st.sidebar.number_input("Purchase price per share", min_value=0.01, value=100.00, step=0.01)

    if st.sidebar.button("Add Stock"):
        st.session_state.portfolios[st.session_state.current_portfolio][new_stock] = {'shares': shares, 'purchase_price': purchase_price}
        st.sidebar.success(f"Added {shares} shares of {new_stock} to {st.session_state.current_portfolio}")

# Display and edit current portfolio
if st.session_state.current_portfolio and st.session_state.current_portfolio != "All":
    st.sidebar.subheader(f"Stocks in {st.session_state.current_portfolio}")
    for stock, details in st.session_state.portfolios[st.session_state.current_portfolio].items():
        col1, col2, col3, col4 = st.sidebar.columns([2,1,1,1])
        col1.write(stock)
        col2.write(f"{details['shares']} shares")
        col3.write(f"${details['purchase_price']:.2f}")
        if col4.button("Remove", key=f"remove_{stock}"):
            del st.session_state.portfolios[st.session_state.current_portfolio][stock]
            st.experimental_rerun()

# Main content
if st.session_state.current_portfolio == "All":
    st.subheader("All Portfolios Overview")
    
    # Fetch S&P 500 data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    sp500_data = get_sp500_data(start_date, end_date)
    
    # Calculate performance for each portfolio
    portfolio_performances = {}
    for portfolio_name, portfolio in st.session_state.portfolios.items():
        if not portfolio:  # Skip empty portfolios
            continue
        
        portfolio_data = {}
        for stock in portfolio:
            data = yf.download(stock, start=start_date, end=end_date)
            portfolio_data[stock] = data['Close']
        
        df = pd.DataFrame(portfolio_data)
        
        # Calculate portfolio value over time
        portfolio_value = sum(df.iloc[-1] * pd.Series({stock: details['shares'] for stock, details in portfolio.items()}))
        portfolio_cost = sum(details['shares'] * details['purchase_price'] for details in portfolio.values())
        
        # Calculate portfolio returns
        portfolio_returns = (df * pd.Series({stock: details['shares'] for stock, details in portfolio.items()})).sum(axis=1)
        portfolio_returns_pct = portfolio_returns.pct_change()
        
        portfolio_performances[portfolio_name] = {
            'returns': portfolio_returns,
            'returns_pct': portfolio_returns_pct,
            'total_value': portfolio_value,
            'total_cost': portfolio_cost,
            'total_return': (portfolio_value - portfolio_cost) / portfolio_cost * 100
        }
    
    # Display summary metrics for all portfolios
    st.subheader("Portfolio Summary")
    summary_data = []
    for portfolio_name, performance in portfolio_performances.items():
        summary_data.append({
            "Portfolio": portfolio_name,
            "Total Value": f"${performance['total_value']:,.2f}",
            "Total Return": f"{performance['total_return']:.2f}%",
            "Total Cost": f"${performance['total_cost']:,.2f}"
        })
    
# Main content
if st.session_state.current_portfolio == "All":
    st.subheader("All Portfolios Overview")
    
    # Fetch S&P 500 data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    sp500_data = get_sp500_data(start_date, end_date)
    
    # Calculate performance for each portfolio
    portfolio_performances = {}
    for portfolio_name, portfolio in st.session_state.portfolios.items():
        if not portfolio:  # Skip empty portfolios
            continue
        
        portfolio_data = {}
        for stock in portfolio:
            data = yf.download(stock, start=start_date, end=end_date)
            portfolio_data[stock] = data['Close']
        
        df = pd.DataFrame(portfolio_data)
        
        # Calculate portfolio value over time
        portfolio_value = sum(df.iloc[-1] * pd.Series({stock: details['shares'] for stock, details in portfolio.items()}))
        portfolio_cost = sum(details['shares'] * details['purchase_price'] for details in portfolio.values())
        
        # Calculate portfolio returns
        portfolio_returns = (df * pd.Series({stock: details['shares'] for stock, details in portfolio.items()})).sum(axis=1)
        portfolio_returns_pct = portfolio_returns.pct_change()
        
        portfolio_performances[portfolio_name] = {
            'returns': portfolio_returns,
            'returns_pct': portfolio_returns_pct,
            'total_value': portfolio_value,
            'total_cost': portfolio_cost,
            'total_return': (portfolio_value - portfolio_cost) / portfolio_cost * 100
        }
    
    # Display summary metrics for all portfolios
    st.subheader("Portfolio Summary")
    summary_data = []
    for portfolio_name, performance in portfolio_performances.items():
        summary_data.append({
            "Portfolio": portfolio_name,
            "Total Value": f"${performance['total_value']:,.2f}",
            "Total Return": f"{performance['total_return']:.2f}%",
            "Total Cost": f"${performance['total_cost']:,.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
if 'Portfolio' in summary_df.columns:
    summary_df = summary_df.set_index('Portfolio')
else:
    st.warning("No portfolios found. Please create a portfolio and add stocks to it.")
    st.table(summary_df)

    # Create a line chart comparing all portfolio performances with S&P 500
    fig = go.Figure()
    for portfolio_name, performance in portfolio_performances.items():
        fig.add_trace(go.Scatter(x=performance['returns'].index, y=performance['returns'], name=portfolio_name, mode='lines'))

    # Add S&P 500 to the chart
    fig.add_trace(go.Scatter(x=sp500_data.index, y=sp500_data / sp500_data.iloc[0] * 100000, name='S&P 500', mode='lines', line=dict(dash='dash')))

    fig.update_layout(
        title="Portfolio Performance Comparison (Normalized)",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        legend_title="Portfolios",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate and display correlation with S&P 500
    st.subheader("Correlation with S&P 500")
    correlations = {}
    for portfolio_name, performance in portfolio_performances.items():
        correlation = performance['returns_pct'].corr(sp500_data.pct_change())
        correlations[portfolio_name] = correlation

    correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    correlation_df = correlation_df.sort_values('Correlation', ascending=False)
    st.table(correlation_df)

    # Display individual portfolio details
    for portfolio_name, performance in portfolio_performances.items():
        with st.expander(f"{portfolio_name} Details"):
            st.write(f"Total Value: ${performance['total_value']:,.2f}")
            st.write(f"Total Return: {performance['total_return']:.2f}%")
            st.write(f"Correlation with S&P 500: {correlations[portfolio_name]:.4f}")
            
            # Portfolio composition pie chart
            portfolio = st.session_state.portfolios[portfolio_name]
            portfolio_composition = {stock: details['shares'] * performance['returns'].iloc[-1] for stock, details in portfolio.items()}
            fig = px.pie(values=list(portfolio_composition.values()), names=list(portfolio_composition.keys()), title=f"{portfolio_name} Composition")
            st.plotly_chart(fig, use_container_width=True)

elif st.session_state.current_portfolio and st.session_state.portfolios[st.session_state.current_portfolio]:
    # ... (rest of the code remains the same)
    # Fetch stock data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    portfolio_data = {}
    for stock in st.session_state.portfolios[st.session_state.current_portfolio]:
        data = yf.download(stock, start=start_date, end=end_date)
        portfolio_data[stock] = data

    # Create a DataFrame with all stock prices
    df = pd.DataFrame({stock: data['Close'] for stock, data in portfolio_data.items()})

    # Calculate portfolio value and performance
    current_value = sum(df.iloc[-1] * pd.Series({stock: details['shares'] for stock, details in st.session_state.portfolios[st.session_state.current_portfolio].items()}))
    cost_basis = sum(details['shares'] * details['purchase_price'] for details in st.session_state.portfolios[st.session_state.current_portfolio].values())
    total_gain_loss = current_value - cost_basis
    total_gain_loss_percent = (total_gain_loss / cost_basis) * 100

    # Display portfolio summary
    st.subheader(f"Portfolio Summary: {st.session_state.current_portfolio}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Value", f"${current_value:,.2f}")
    col2.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}", f"{total_gain_loss_percent:.2f}%")
    col3.metric("Cost Basis", f"${cost_basis:,.2f}")

    # Create a line chart of portfolio performance
    fig = go.Figure()
    for stock in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[stock], name=stock, mode='lines'))
    fig.update_layout(
        title="Stock Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Stocks",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display individual stock details
    st.subheader("Individual Stock Details")
    for stock, details in st.session_state.portfolios[st.session_state.current_portfolio].items():
        with st.expander(f"{stock} Details"):
            col1, col2, col3, col4 = st.columns(4)
            current_price = df[stock].iloc[-1]
            stock_value = current_price * details['shares']
            gain_loss = (current_price - details['purchase_price']) * details['shares']
            gain_loss_percent = ((current_price - details['purchase_price']) / details['purchase_price']) * 100
            
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Shares", details['shares'])
            col3.metric("Value", f"${stock_value:,.2f}")
            col4.metric("Gain/Loss", f"${gain_loss:,.2f}", f"{gain_loss_percent:.2f}%")
            
            # Stock-specific charts
            hist_data = portfolio_data[stock]
            
            # Price and volume chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist_data.index,
                                         open=hist_data['Open'],
                                         high=hist_data['High'],
                                         low=hist_data['Low'],
                                         close=hist_data['Close'],
                                         name='Price'),
                          row=1, col=1)
            fig.add_trace(go.Bar(x=hist_data.index, y=hist_data['Volume'], name='Volume'), row=2, col=1)
            fig.update_layout(title=f"{stock} Price and Volume", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Predict future prices
            X = np.array(range(len(hist_data))).reshape(-1, 1)
            y = hist_data['Close'].values
            model = LinearRegression()
            model.fit(X, y)
            
            future_dates = pd.date_range(start=hist_data.index[-1] + timedelta(days=1), periods=30)
            future_X = np.array(range(len(hist_data), len(hist_data) + 30)).reshape(-1, 1)
            future_prices = model.predict(future_X)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], name='Historical'))
            fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='Predicted', line=dict(dash='dash')))
            fig.update_layout(title=f"{stock} Price Prediction (Next 30 Days)", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

    # Portfolio composition pie chart
    portfolio_composition = {stock: details['shares'] * df[stock].iloc[-1] for stock, details in st.session_state.portfolios[st.session_state.current_portfolio].items()}
    fig = px.pie(values=list(portfolio_composition.values()), names=list(portfolio_composition.keys()), title=f"{st.session_state.current_portfolio} Composition")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Create a portfolio and add stocks to get started!")

# Add a footer
st.markdown("---")
st.markdown("© 2024 BEZZ Advanced Multi-Portfolio Stock Tracker")
                              