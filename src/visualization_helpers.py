"""
Visualization Helpers for Game Theory Policy Analysis (FIXED)
============================================================

Professional visualization utilities for creating publication-ready
charts, interactive dashboards, and stakeholder presentations.

Author: Rosalina Torres
Institution: Northeastern University - Data Analytics Engineering
Date: January 2025
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class PolicyVisualizationSuite:
    """
    Professional visualization suite for policy analysis
    
    Creates publication-ready charts for academic papers,
    stakeholder presentations, and interactive dashboards.
    """
    
    def __init__(self, style: str = 'professional'):
        """
        Initialize visualization suite
        
        Args:
            style: Visualization style ('professional', 'academic', 'executive')
        """
        self.style = style
        self.color_palette = self._get_color_palette()
        self._setup_matplotlib_style()
        
    def _get_color_palette(self) -> Dict[str, str]:
        """Define professional color palette"""
        return {
            'stem_green': '#22c55e',
            'tariff_red': '#ef4444',
            'innovation_blue': '#3b82f6',
            'cooperation_teal': '#14b8a6',
            'warning_orange': '#f97316',
            'neutral_gray': '#6b7280',
            'success_emerald': '#10b981',
            'danger_rose': '#e11d48',
            'primary_indigo': '#4f46e5',
            'accent_purple': '#8b5cf6'
        }
    
    def _setup_matplotlib_style(self) -> None:
        """Configure matplotlib for professional output"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
        
        # Custom styling
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def create_payoff_matrix_heatmap(self, payoff_matrix: np.ndarray, 
                                   strategy_names: Dict[str, List[str]],
                                   title: str = "Game Theory Payoff Matrix",
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive heatmap of game theory payoff matrix
        
        Args:
            payoff_matrix: 3D array with payoffs for each strategy combination
            strategy_names: Dictionary mapping players to strategy names
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        # Calculate total welfare for coloring
        welfare_matrix = np.sum(payoff_matrix, axis=2)
        
        # Create annotations with both payoffs
        annotations = []
        for i in range(payoff_matrix.shape[0]):
            for j in range(payoff_matrix.shape[1]):
                payoffs = payoff_matrix[i, j]
                annotations.append(f"({payoffs[0]:.1f}, {payoffs[1]:.1f})")
        
        annotations_matrix = np.array(annotations).reshape(payoff_matrix.shape[:2])
        
        # FIXED: Correct colorbar configuration for Plotly
        fig = go.Figure(data=go.Heatmap(
            z=welfare_matrix,
            text=annotations_matrix,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale=[
                [0, self.color_palette['danger_rose']],
                [0.5, self.color_palette['warning_orange']], 
                [1, self.color_palette['success_emerald']]
            ],
            colorbar=dict(
                title=dict(text="Total Welfare", side="right"),  # FIXED: Correct syntax
                x=1.02,
                len=0.7
            )
        ))
        
        # Get player names and strategies
        players = list(strategy_names.keys())
        player1_strategies = strategy_names[players[0]]
        player2_strategies = strategy_names[players[1]]
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            xaxis=dict(
                title=dict(
                    text=f"{players[1]} Strategy",
                    font={'size': 14}
                ),
                tickmode='array',
                tickvals=list(range(len(player2_strategies))),
                ticktext=player2_strategies
            ),
            yaxis=dict(
                title=dict(
                    text=f"{players[0]} Strategy",
                    font={'size': 14}
                ), 
                tickmode='array',
                tickvals=list(range(len(player1_strategies))),
                ticktext=player1_strategies
            ),
            width=600,
            height=500,
            font=dict(family="Arial", size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            # Only try to write PNG if kaleido is installed
            try:
                fig.write_image(save_path.replace('.html', '.png'))
            except Exception as e:
                print(f"Note: Could not save PNG (install kaleido for image export): {e}")
        
        return fig
    
    def create_policy_comparison_dashboard(self, predictions_df: pd.DataFrame,
                                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive policy comparison dashboard
        
        Args:
            predictions_df: DataFrame with policy scenario predictions
            save_path: Optional path to save the dashboard
            
        Returns:
            Plotly figure with multiple subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Policy Scenario Outcomes',
                'Confidence Intervals',
                'Policy Type Performance',
                'Risk-Return Analysis'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Policy scenario outcomes
        colors = [self.color_palette['stem_green'] if 'STEM' in policy 
                 else self.color_palette['tariff_red'] if 'Protectionist' in policy
                 else self.color_palette['warning_orange']
                 for policy in predictions_df['policy_type']]
        
        fig.add_trace(
            go.Bar(
                x=predictions_df['scenario_name'],
                y=predictions_df['predicted_outcome'],
                marker_color=colors,
                name='Predicted Outcome',
                text=predictions_df['predicted_outcome'].round(2),
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Comprehensive Policy Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=800,
            showlegend=True,
            font=dict(family="Arial", size=11)
        )
        
        if save_path:
            fig.write_html(save_path)
            try:
                fig.write_image(save_path.replace('.html', '.png'))
            except Exception as e:
                print(f"Note: Could not save PNG (install kaleido for image export): {e}")
        
        return fig
    
    def create_executive_summary_visual(self, key_metrics: Dict,
                                      save_path: Optional[str] = None) -> go.Figure:
        """
        Create executive summary visualization with key metrics
        
        Args:
            key_metrics: Dictionary with executive-level metrics
            save_path: Optional save path
            
        Returns:
            Plotly figure with key performance indicators
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'STEM ROI Advantage', 'Employment Growth', 'Innovation Impact',
                'Economic Welfare', 'International Ranking', 'Policy Success Rate'
            ),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # 1. STEM ROI Advantage
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=320,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "STEM ROI (%)"},
                delta={'reference': -28, 'position': "top"},
                gauge={
                    'axis': {'range': [None, 400]},
                    'bar': {'color': self.color_palette['stem_green']},
                    'steps': [
                        {'range': [0, 100], 'color': "lightgray"},
                        {'range': [100, 200], 'color': "yellow"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 300
                    }
                }
            ),
            row=1, col=1
        )
        
        fig.update_layout(
            title={
                'text': 'Executive Dashboard: STEM Investment Strategy Performance',
                'x': 0.5,
                'font': {'size': 20}
            },
            height=600,
            font=dict(family="Arial", size=11)
        )
        
        if save_path:
            fig.write_html(save_path)
            try:
                fig.write_image(save_path.replace('.html', '.png'))
            except Exception as e:
                print(f"Note: Could not save PNG (install kaleido for image export): {e}")
        
        return fig

def main():
    """
    Execute the master visualization pipeline (FIXED VERSION)
    """
    print("🎨 VISUALIZATION SUITE DEMO (FIXED)")
    print("=" * 50)
    
    # Initialize visualization suite
    viz_suite = PolicyVisualizationSuite(style='professional')
    
    # Demo payoff matrix visualization
    print("1️⃣ Creating payoff matrix heatmap...")
    payoff_matrix = np.array([[(8, 7), (6, 4)], [(4, 6), (2, 2)]])
    strategy_names = {
        'United States': ['STEM Investment', 'Manufacturing Protection'],
        'Global Economy': ['Innovation', 'Traditional']
    }
    
    heatmap_fig = viz_suite.create_payoff_matrix_heatmap(
        payoff_matrix, strategy_names,
        title="Innovation Investment Game",
        save_path='payoff_matrix_demo.html'
    )
    
    # Demo policy comparison
    print("2️⃣ Creating policy comparison dashboard...")
    sample_predictions = pd.DataFrame({
        'scenario_name': ['STEM Strategy', 'Current Policy', 'Massachusetts Model'],
        'predicted_outcome': [8.2, 1.1, 8.5],
        'confidence_lower': [7.8, 0.8, 8.1],
        'confidence_upper': [8.6, 1.4, 8.9],
        'policy_type': ['STEM', 'Protectionist', 'STEM']
    })
    
    dashboard_fig = viz_suite.create_policy_comparison_dashboard(
        sample_predictions,
        save_path='policy_dashboard_demo.html'
    )
    
    # Demo executive summary
    print("3️⃣ Creating executive summary visual...")
    key_metrics = {
        'stem_roi': 320,
        'model_accuracy': 94.3,
        'stem_advantage': 2.6
    }
    
    executive_fig = viz_suite.create_executive_summary_visual(
        key_metrics,
        save_path='executive_summary_demo.html'
    )
    
    print("\n✅ VISUALIZATION DEMOS COMPLETE!")
    print("📁 Generated files:")
    print("  • payoff_matrix_demo.html")
    print("  • policy_dashboard_demo.html") 
    print("  • executive_summary_demo.html")
    print("  • PNG files (if kaleido is installed)")
    
    return viz_suite

if __name__ == "__main__":
    # Run visualization demo
    visualization_suite = main()
    
    print("\n🎨 VISUALIZATION SUITE READY!")
    print("💡 Use visualization_suite.create_*() methods for custom charts")
    print("📊 All methods generate HTML (and PNG if kaleido installed)")
