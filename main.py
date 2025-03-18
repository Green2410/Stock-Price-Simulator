import streamlit as st
import yaml
import logging
import matplotlib.pyplot as plt
from Stock_simulator.get_data import StockParameters
from Stock_simulator.simulator import StockPriceSimulator
import warnings
import plotly.io as pio

# Set default Plotly template
pio.templates.default = "plotly_white"

# Suppress warnings
warnings.filterwarnings("ignore")

def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Set page config
    st.set_page_config(
        page_title="Monte Carlo Stock Price Simulation",
        page_icon="📈",
        layout="wide",
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-text {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with custom styling
    st.markdown('<div class="main-header">Monte Carlo Stock Price Simulation</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="info-text">This tool simulates potential future stock prices using Monte Carlo methods based on historical volatility and returns.</p>', unsafe_allow_html=True)
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # User inputs
        ticker = st.text_input("Enter Stock Ticker:", "AAPL")
        trading_days = st.slider("Number of Trading Days:", min_value=1, max_value=252, value=252, 
                                help="Number of trading days to simulate (252 = 1 year)")
    
    with col2:
        num_simulations = st.slider("Number of Simulations:", min_value=100, max_value=5000, value=1000, step=100,
                                  help="More simulations = more accurate results but slower performance")
        target_price = st.number_input("Target Price ($):", min_value=1.0, value=200.0, step=1.0,
                                      help="Set a price target to calculate probability of reaching")
    
    # Run simulation button
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)
    
    if run_button:
        # Show spinner during calculation
        with st.spinner("Running Monte Carlo simulation... This may take a moment."):
            try:
                params = StockParameters(ticker)
                params.fetch_parameters()
                
                # Display initial parameters
                st.markdown('<div class="subheader">Stock Parameters</div>', unsafe_allow_html=True)
                
                # Create three columns for displaying parameters
                param_col1, param_col2, param_col3 = st.columns(3)
                
                with param_col1:
                    st.metric("Current Price", f"${params.initial_price:.2f}")
                
                with param_col2:
                    st.metric("Annual Volatility", f"{params.volatility:.2%}")
                
                with param_col3:
                    st.metric("Expected Annual Return", f"{params.drift:.2%}")
                
                simulator = StockPriceSimulator(
                    initial_price=params.initial_price,
                    drift=params.drift,
                    volatility=params.volatility,
                    days=trading_days,
                    num_simulations=num_simulations
                )
                
                simulator.run_simulation()
                
                # Plot simulations using Plotly
                st.markdown('<div class="subheader">Price Simulation Paths</div>', unsafe_allow_html=True)
                fig1 = simulator.plot_simulations_plotly(num_paths_to_plot=100, target_price=target_price)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Calculate statistics
                stats = simulator.calculate_statistics(target_price=target_price)
                
                st.markdown('<div class="subheader">Simulation Results</div>', unsafe_allow_html=True)
                
                # Arrange metrics in a grid
                st.markdown('<div class="stat-container">', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Final Price", f"${stats.get('mean', 0):.2f}")
                    st.metric("Minimum Final Price", f"${stats.get('min', 0):.2f}")
                
                with col2:
                    st.metric("Median Final Price", f"${stats.get('median', 0):.2f}")
                    st.metric("Maximum Final Price", f"${stats.get('max', 0):.2f}")
                
                with col3:
                    st.metric("Standard Deviation", f"${stats.get('std_dev', 0):.2f}")
                    st.metric("5th Percentile", f"${stats.get('percentile_5', 0):.2f}")
                
                with col4:
                    st.metric("Probability of Target", f"{stats.get('probability_above_target', 0) * 100:.1f}%")
                    st.metric("95th Percentile", f"${stats.get('percentile_95', 0):.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Plot distribution using Plotly
                st.markdown('<div class="subheader">Final Price Distribution</div>', unsafe_allow_html=True)
                fig2 = simulator.plot_distribution_plotly(target_price=target_price)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Add explanatory text
                with st.expander("How to interpret these results"):
                    st.markdown("""
                    ### Understanding the Results
                    
                    - **Price Paths Graph**: Each light blue line represents one possible future price path. The red line shows the average of all simulations.
                    
                    - **Distribution Graph**: Shows the probability distribution of the final stock price after the simulation period.
                    
                    - **Probability of Target**: The percentage of simulations where the final price was at or above your target price.
                    
                    - **Percentiles**: The 5th percentile means that in 5% of simulations, the price was below this value. The 95th percentile means that in 95% of simulations, the price was below this value.
                    
                    
                    Remember that these simulations are based on historical data and assume that future price movements will follow similar patterns. Market conditions can change, and past performance is not indicative of future results.
                    """)
            
            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
                st.error("Please check that the stock ticker is valid and try again.")

if __name__ == '__main__':
    main()