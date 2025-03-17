from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff

class StockPriceSimulator:
    def __init__(self, initial_price, drift, volatility, days, num_simulations=1000, dt=1/252):
        """
        Initializing the simulator
        
        Params:
        -----------
        initial_price : float
            The starting price of the stock
        drift : float
            Annual expected return (mu)
        volatility : float
            Annual volatility (sigma)
        days : int
            Number of trading days to simulate
        num_simulations : int
            Number of paths to simulate
        dt : float
            Time step in years (default: 1/252 for daily steps)
        """
        self.S0 = initial_price
        self.mu = drift
        self.sigma = volatility
        self.days = days
        self.num_simulations = num_simulations
        self.dt = dt
        self.simulation_results = None
    
    def run_simulation(self):
        """
        Monte Carlo simulation using Geometric Brownian Motion.
        Assumes underlying distribution is normal
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing all simulated price paths
        """
        # Generate time steps
        time_steps = np.arange(0, self.days + 1)
        
        # Initialize array to hold all simulations
        prices = np.zeros((self.days + 1, self.num_simulations))
        prices[0] = self.S0
        
        # Generate random returns for all simulations at once
        random_returns = np.random.normal(0, 1, size=(self.days, self.num_simulations))
        
        # Calculate each day's price
        for t in range(1, self.days + 1):
            # Drift component
            drift_component = (self.mu - 0.5 * self.sigma**2) * self.dt
            
            # Diffusion component (random shock)
            diffusion_component = self.sigma * np.sqrt(self.dt) * random_returns[t-1]
            
            # Calculate price using GBM
            prices[t] = prices[t-1] * np.exp(drift_component + diffusion_component)
        
        # Convert to DataFrame for easier analysis
        self.simulation_results = pd.DataFrame(prices, index=time_steps)
        
        return self.simulation_results
    
    def plot_simulations_plotly(self, num_paths_to_plot=100, target_price=None):
        """
        Plot the simulated price paths using Plotly.
        
        Parameters:
        -----------
        num_paths_to_plot : int
            Number of paths to display on the plot (default: 100)
        target_price : float
            Target price to highlight on the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive Plotly figure
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        # Create figure
        fig = go.Figure()
        
        # If we have too many simulations, only plot a subset
        paths_to_plot = min(num_paths_to_plot, self.num_simulations)
        
        # Get time steps for x-axis
        time_steps = self.simulation_results.index.tolist()
        
        # Add individual simulation paths
        for i in range(paths_to_plot):
            path_prices = self.simulation_results.iloc[:, i].tolist()
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=path_prices,
                    mode='lines',
                    line=dict(color='rgba(135, 206, 235, 1.0)', width=0.5),
                    hoverinfo='skip',
                    showlegend=False
                )
            )
        
        # Add mean path
        mean_path = self.simulation_results.mean(axis=1).tolist()
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=mean_path,
                mode='lines',
                line=dict(color='red', width=3),
                name='Average Path',
                hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            )
        )
        
        # Add initial price marker
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[self.S0],
                mode='markers',
                marker=dict(color='orange', size=10, symbol='circle'),
                name='Initial Price',
                hovertemplate='Starting Price: $%{y:.2f}<extra></extra>'
            )
        )
        
        # Add target price line if provided
        if target_price is not None:
            fig.add_trace(
                go.Scatter(
                    x=[0, self.days],
                    y=[target_price, target_price],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name=f'Target Price (${target_price:.2f})',
                    hoverinfo='name'
                )
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Stock Price Simulation - {self.num_simulations} Paths',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            xaxis_title='Trading Days',
            yaxis_title='Stock Price ($)',
            hovermode='closest',
            template='plotly_white',
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01,
                bgcolor='#0E1117'
            ),
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='linear'
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
        
        return fig
    
    def calculate_statistics(self, target_price=None):
        """
        Calculate summary statistics for the simulation.
        
        Params:
        -----------
        target_price : float
            Price to calculate probability of exceeding
            
        Returns:
        --------
        dict
            Dictionary containing summary statistics
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        # Get final prices
        final_prices = self.simulation_results.iloc[-1, :]
        
        # Calculate statistics
        stats = {
            'mean': final_prices.mean(),
            'median': final_prices.median(),
            'std_dev': final_prices.std(),
            'min': final_prices.min(),
            'max': final_prices.max(),
            'percentile_5': final_prices.quantile(0.05),
            'percentile_95': final_prices.quantile(0.95),
        }
        
        # Calculate probability of exceeding target price
        if target_price is not None:
            prob_above_target = (final_prices >= target_price).mean()
            stats['probability_above_target'] = prob_above_target
        
        return stats
    
    def plot_distribution_plotly(self, target_price=None):
        """
        Plot the distribution of final stock prices using Plotly.
        
        Parameters:
        -----------
        target_price : float, optional
            Target price to highlight on the plot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive Plotly figure
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        final_prices = self.simulation_results.iloc[-1, :].tolist()
        
        # Create histogram figure
        fig = go.Figure()
        
        # Add histogram with KDE
        fig.add_trace(
            go.Histogram(
                x=final_prices,
                nbinsx=50,
                histnorm='probability density',
                name='Price Distribution',
                marker=dict(
                    color='rgba(30, 144, 255, 0.6)',
                    line=dict(color='rgba(30, 144, 255, 1)', width=1)
                ),
            )
        )
        
        # Add KDE
        import numpy as np
        from scipy import stats
        
        kde_x = np.linspace(min(final_prices) * 0.9, max(final_prices) * 1.1, 1000)
        kde = stats.gaussian_kde(final_prices)
        kde_y = kde(kde_x)
        
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                line=dict(color='rgba(0, 0, 150, 0.8)', width=2),
                name='Density',
                hovertemplate='Price: $%{x:.2f}<br>Density: %{y:.6f}<extra></extra>'
            )
        )
        
        # Add initial price line
        fig.add_trace(
            go.Scatter(
                x=[self.S0, self.S0],
                y=[0, max(kde_y) * 1.1],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name=f'Initial Price (${self.S0:.2f})',
                hovertemplate='Initial Price: $%{x:.2f}<extra></extra>'
            )
        )
        
        # Add mean final price line
        mean_price = np.mean(final_prices)
        fig.add_trace(
            go.Scatter(
                x=[mean_price, mean_price],
                y=[0, max(kde_y) * 1.1],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'Mean Final Price (${mean_price:.2f})',
                hovertemplate='Mean Final Price: $%{x:.2f}<extra></extra>'
            )
        )
        
        # Add target price line and probability annotation if provided
        if target_price is not None:
            prob_above = (np.array(final_prices) >= target_price).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=[target_price, target_price],
                    y=[0, max(kde_y) * 1.1],
                    mode='lines',
                    line=dict(color='green', width=2),
                    name=f'Target Price (${target_price:.2f})',
                    hovertemplate='Target Price: $%{x:.2f}<extra></extra>'
                )
            )
            
            # Add annotations for probability
            fig.add_annotation(
                x=target_price,
                y=max(kde_y) * 0.95,
                text=f'P(Price ≥ ${target_price:.2f}) = {prob_above:.1%}',
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green',
                ax=40,
                ay=-40,
                bgcolor='white',
                bordercolor='green',
                borderwidth=2,
                borderpad=4,
                font=dict(color='green')
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Distribution of Stock Prices After {self.days} Trading Days',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            xaxis_title='Stock Price ($)',
            yaxis_title='Probability Density',
            template='plotly_white',
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01,
                bgcolor='#0E1117'
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            hovermode='closest',
            bargap=0.05
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
        
        return fig
    
    # Keep the original matplotlib methods for compatibility
    def plot_simulations(self, num_paths_to_plot=100, target_price=None):
        """
        Plot the simulated price paths using Matplotlib (kept for compatibility).
        
        Parameters:
        -----------
        num_paths_to_plot : int
            Number of paths to display on the plot (default: 100)
        target_price : float
            Target price to highlight on the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the plot
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # If we have too many simulations, only plot a subset
        paths_to_plot = min(num_paths_to_plot, self.num_simulations)
        
        # Plot individual paths
        for i in range(paths_to_plot):
            ax.plot(self.simulation_results.index, self.simulation_results.iloc[:, i], 
                     'b-', alpha=0.1, linewidth=0.5)
        
        # Plot mean path
        mean_path = self.simulation_results.mean(axis=1)
        ax.plot(self.simulation_results.index, mean_path, 'r-', linewidth=2, label='Mean Path')
        
        # Add target price if provided
        if target_price is not None:
            ax.axhline(y=target_price, color='g', linestyle='--', label=f'Target Price (${target_price})')
        
        ax.set_title(f'Stock Price Simulation - {self.num_simulations} Paths')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Stock Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_distribution(self, target_price=None):
        """
        Plot the distribution of final stock prices using Matplotlib (kept for compatibility).
        
        Parameters:
        -----------
        target_price : float, optional
            Target price to highlight on the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the plot
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        final_prices = self.simulation_results.iloc[-1, :]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram with KDE
        sns.histplot(final_prices, kde=True, stat="density", color='skyblue', ax=ax)
        
        # Add vertical line for initial price
        ax.axvline(x=self.S0, color='black', linestyle='--', 
                   linewidth=1.5, label=f'Initial Price (${self.S0:.2f})')
        
        # Add vertical line for mean final price
        mean_price = final_prices.mean()
        ax.axvline(x=mean_price, color='red', linestyle='-', 
                   linewidth=1.5, label=f'Mean Final Price (${mean_price:.2f})')
        
        # Add target price if provided
        if target_price is not None:
            ax.axvline(x=target_price, color='green', linestyle='-', 
                       linewidth=1.5, label=f'Target Price (${target_price:.2f})')
            
            # Calculate probability of exceeding target
            prob_above = (final_prices >= target_price).mean()
            ax.text(0.95, 0.95, f'P(Price ≥ ${target_price:.2f}) = {prob_above:.1%}',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(f'Distribution of Stock Prices After {self.days} Trading Days')
        ax.set_xlabel('Stock Price ($)')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig