import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
import threading
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# IB API Client
class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.historical_data = {}
        
    def error(self, reqId, errorCode, errorString):
        if errorCode == 2176 and "fractional share" in errorString.lower():
            return
        st.warning(f"Error {errorCode}: {errorString}")
        
    def nextValidId(self, orderId):
        self.connected = True
        
    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
        
    def historicalDataEnd(self, reqId, start, end):
        pass

# Initialize session state
if 'ib_app' not in st.session_state:
    st.session_state.ib_app = None
    st.session_state.connected = False
    st.session_state.volatility_data = None
    st.session_state.current_implied_vol = None

# Page config
st.set_page_config(page_title="Implied Volatility Trading Dashboard", layout="wide")

# Title
st.title("📊 Implied Volatility Trading Dashboard")

# Sidebar for connection and settings
st.sidebar.header("Interactive Brokers Connection")
host = st.sidebar.text_input("Host", value="127.0.0.1")
port = st.sidebar.number_input("Port", value=7497, step=1)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Connect", disabled=st.session_state.connected):
        try:
            st.session_state.ib_app = IBApp()
            st.session_state.ib_app.connect(host, int(port), clientId=1)
            api_thread = threading.Thread(target=st.session_state.ib_app.run, daemon=True)
            api_thread.start()
            time.sleep(2)
            if st.session_state.ib_app.connected:
                st.session_state.connected = True
                st.sidebar.success("Connected to IB")
            else:
                st.sidebar.error("Failed to connect")
        except Exception as e:
            st.sidebar.error(f"Connection error: {str(e)}")

with col2:
    if st.button("Disconnect", disabled=not st.session_state.connected):
        if st.session_state.ib_app:
            st.session_state.ib_app.disconnect()
            st.session_state.connected = False
            st.sidebar.info("Disconnected")

# Main content
st.sidebar.header("Data Query")
symbol = st.sidebar.text_input("Symbol", value="SPY")
duration = st.sidebar.text_input("Duration", value="2 Y")

if st.sidebar.button("Query IV Data", disabled=not st.session_state.connected):
    with st.spinner("Fetching data from Interactive Brokers..."):
        try:
            contract = Contract()
            contract.symbol = symbol.upper()
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            req_id = 1
            st.session_state.ib_app.historical_data = {}
            st.session_state.ib_app.reqHistoricalData(
                req_id, contract, '', duration, '1 day', 'TRADES', 1, 1, False, []
            )
            
            time.sleep(5)
            
            if req_id in st.session_state.ib_app.historical_data:
                data = st.session_state.ib_app.historical_data[req_id]
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate returns and volatility
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                # Rolling 30-day volatility
                window = 30
                df['volatility_30d'] = df['log_returns'].rolling(window=window).std() * np.sqrt(252)
                
                # Implied volatility proxy (annualized rolling volatility)
                df['implied_vol'] = df['volatility_30d']
                
                st.session_state.volatility_data = df.dropna()
                st.session_state.current_implied_vol = df['implied_vol'].iloc[-1]
                
                st.success(f"Successfully loaded {len(df)} data points for {symbol}")
            else:
                st.error("No data received from IB")
        except Exception as e:
            st.error(f"Error querying data: {str(e)}")

# Analysis section
if st.session_state.volatility_data is not None:
    st.header("Volatility Analysis")
    
    if st.button("Analyze Implied Volatility"):
        df = st.session_state.volatility_data
        
        # Prepare analysis dataframe
        forward_period = 30
        analysis_df = pd.DataFrame({
            'current_vol': df['implied_vol'][:-forward_period],
            'forward_30d_vol': df['implied_vol'].shift(-forward_period)[:-forward_period]
        }).dropna()
        
        analysis_df['vol_diff'] = analysis_df['forward_30d_vol'] - analysis_df['current_vol']
        
        # Regression 1: Forward vol vs Current vol
        slope1, intercept1, r_value1, p_value1, std_error1 = stats.linregress(
            analysis_df['current_vol'], analysis_df['forward_30d_vol']
        )
        
        # Regression 2: Vol difference vs Current vol
        slope2, intercept2, r_value2, p_value2, std_error2 = stats.linregress(
            analysis_df['current_vol'], analysis_df['vol_diff']
        )
        
        # Find intersection with y=x line
        intersection_x = intercept1 / (1 - slope1)
        
        # Regime analysis
        high_vol_regime = analysis_df['current_vol'] > intersection_x
        low_vol_regime = analysis_df['current_vol'] <= intersection_x
        
        # Regime-specific regressions
        if high_vol_regime.sum() > 10:
            slope_high, intercept_high, r_high, p_high, std_error_high = stats.linregress(
                analysis_df.loc[high_vol_regime, 'current_vol'],
                analysis_df.loc[high_vol_regime, 'vol_diff']
            )
        else:
            slope_high = intercept_high = r_high = p_high = None
            
        if low_vol_regime.sum() > 10:
            slope_low, intercept_low, r_low, p_low, std_error_low = stats.linregress(
                analysis_df.loc[low_vol_regime, 'current_vol'],
                analysis_df.loc[low_vol_regime, 'vol_diff']
            )
        else:
            slope_low = intercept_low = r_low = p_low = None
        
        # Create visualizations
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Forward vol vs Current vol
        ax1.scatter(analysis_df['current_vol'], analysis_df['forward_30d_vol'], alpha=0.6, s=20)
        x_range = np.linspace(analysis_df['current_vol'].min(), analysis_df['current_vol'].max(), 100)
        y_pred1 = slope1 * x_range + intercept1
        ax1.plot(x_range, y_pred1, 'r-', linewidth=2, label=f"Regression R² = {r_value1**2:.3f}")
        min_val = min(analysis_df['current_vol'].min(), analysis_df['forward_30d_vol'].min())
        max_val = max(analysis_df['current_vol'].max(), analysis_df['forward_30d_vol'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.7, label='y=x (No Change)')
        ax1.set_xlabel("Current Implied Volatility")
        ax1.set_ylabel("30-Day Forward Average Vol")
        ax1.set_title(f"Forward Vol vs Current Vol\ny = {slope1:.3f}x + {intercept1:.3f}, R² = {r_value1**2:.3f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Vol difference by regime
        ax2.scatter(analysis_df.loc[high_vol_regime, 'current_vol'],
                   analysis_df.loc[high_vol_regime, 'vol_diff'],
                   alpha=0.6, s=20, color='red', label='High Vol Regime')
        ax2.scatter(analysis_df.loc[low_vol_regime, 'current_vol'],
                   analysis_df.loc[low_vol_regime, 'vol_diff'],
                   alpha=0.6, s=20, color='blue', label='Low Vol Regime')
        
        if slope_high is not None:
            x_high = analysis_df.loc[high_vol_regime, 'current_vol']
            if len(x_high) > 0:
                x_range_high = np.linspace(x_high.min(), x_high.max(), 100)
                y_pred_high = slope_high * x_range_high + intercept_high
                ax2.plot(x_range_high, y_pred_high, 'r-', linewidth=2, label=f'High Vol R² = {r_high**2:.3f}')
        
        if slope_low is not None:
            x_low = analysis_df.loc[low_vol_regime, 'current_vol']
            if len(x_low) > 0:
                x_range_low = np.linspace(x_low.min(), x_low.max(), 100)
                y_pred_low = slope_low * x_range_low + intercept_low
                ax2.plot(x_range_low, y_pred_low, 'b-', linewidth=2, label=f'Low Vol R² = {r_low**2:.3f}')
        
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='No Change (y=0)')
        ax2.axvline(x=intersection_x, color='g', linestyle=':', linewidth=1, alpha=0.7,
                   label=f'Regime Split (Vol={intersection_x:.3f})')
        ax2.set_xlabel("Current Implied Volatility")
        ax2.set_ylabel("Vol Difference (Forward - Current)")
        ax2.set_title("Vol Difference vs Current Vol (Regime Analysis)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Volatility time series
        ax3.plot(df.index, df['implied_vol'], label='Implied Volatility', linewidth=1)
        vol_75th = df['implied_vol'].quantile(0.75)
        vol_25th = df['implied_vol'].quantile(0.25)
        ax3.axhline(y=vol_75th, color='red', linestyle='--', alpha=0.7, label='75th Percentile')
        ax3.axhline(y=vol_25th, color='green', linestyle='--', alpha=0.7, label='25th Percentile')
        ax3.axhline(y=df['implied_vol'].mean(), color='black', linestyle='-', alpha=0.7, label='Mean')
        
        if st.session_state.current_implied_vol is not None:
            ax3.scatter(df.index[-1], st.session_state.current_implied_vol,
                       color='red', s=100, zorder=5, label='Current')
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Implied Volatility')
        ax3.set_title('Implied Volatility Time Series with Regime Bands')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display statistics
        st.subheader("Statistical Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Regression 1: Forward Vol on Current Vol**")
            st.write(f"Slope: {slope1:.4f}")
            st.write(f"Intercept: {intercept1:.4f}")
            st.write(f"R²: {r_value1**2:.4f}")
            st.write(f"P-value: {p_value1:.4f}")
            st.write(f"Intersection with y=x: {intersection_x:.4f}")
        
        with col2:
            st.write("**Regression 2: Vol Difference on Current Vol**")
            st.write(f"Slope: {slope2:.4f}")
            st.write(f"Intercept: {intercept2:.4f}")
            st.write(f"R²: {r_value2**2:.4f}")
            st.write(f"P-value: {p_value2:.4f}")
        
        st.subheader("Regime Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**HIGH VOL Regime (Vol > {intersection_x:.3f})**")
            if slope_high is not None:
                st.write(f"Slope: {slope_high:.4f}")
                st.write(f"Intercept: {intercept_high:.4f}")
                st.write(f"R²: {r_high**2:.4f}")
                st.write(f"P-value: {p_high:.4f}")
                st.write(f"Data points: {high_vol_regime.sum()}")
            else:
                st.write("Insufficient data for regression")
        
        with col2:
            st.write(f"**LOW VOL Regime (Vol ≤ {intersection_x:.3f})**")
            if slope_low is not None:
                st.write(f"Slope: {slope_low:.4f}")
                st.write(f"Intercept: {intercept_low:.4f}")
                st.write(f"R²: {r_low**2:.4f}")
                st.write(f"P-value: {p_low:.4f}")
                st.write(f"Data points: {low_vol_regime.sum()}")
            else:
                st.write("Insufficient data for regression")
        
        # Trading insights
        st.subheader("Trading Insights")
        if slope1 < 1:
            st.info("📉 Forward volatility tends to mean-revert (slope < 1)")
        else:
            st.info("📈 Forward volatility tends to trend (slope > 1)")
            
        if slope2 < 0:
            st.info("🔄 High current volatility predicts lower future volatility (mean reversion)")
        else:
            st.info("⚡ High current volatility predicts higher future volatility (momentum)")
    
    # Display current data summary
    st.subheader("Current Data Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Implied Vol", f"{st.session_state.current_implied_vol:.4f}" if st.session_state.current_implied_vol else "N/A")
    with col2:
        st.metric("Data Points", len(st.session_state.volatility_data))
    with col3:
        st.metric("Symbol", symbol)
    
    # Show data table
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state.volatility_data.tail(50))
