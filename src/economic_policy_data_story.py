"""
GOVERNMENT ECONOMIC POLICY DATA STORYTELLING PROJECT
====================================================
A comprehensive data story analyzing Trump's economic policies and their catastrophic impacts

This project demonstrates advanced data science skills for ML/AI engineering portfolio:
- Economic data analysis
- Policy impact modeling
- Predictive analytics
- Interactive visualizations
- Statistical storytelling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up professional styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

class EconomicPolicyDataStory:
    """
    Comprehensive data storytelling framework for economic policy analysis
    
    This class creates a complete narrative from historical context through 
    current impacts to future predictions using advanced data science techniques.
    """
    
    def __init__(self):
        """Initialize the data storytelling engine"""
        self.setup_data()
        self.story_structure = {
            'Act 1': 'Historical Context & Policy Setup',
            'Act 2': 'Policy Implementation & Immediate Impact', 
            'Act 3': 'Economic Catastrophe Unfolds',
            'Act 4': 'Predictive Modeling & Future Scenarios'
        }
        
    def setup_data(self):
        """Create comprehensive datasets for storytelling"""
        
        # Timeline data - The complete story arc
        self.timeline_data = pd.DataFrame({
            'date': pd.date_range('2024-11-01', '2025-07-19', freq='D'),
            'days_since_election': range(0, (pd.Timestamp('2025-07-19') - pd.Timestamp('2024-11-01')).days + 1)
        })
        
        # Add major events
        self.events = [
            {'date': '2024-11-05', 'event': 'Election Day - Trump Victory', 'impact_score': 2},
            {'date': '2025-01-20', 'event': 'Inauguration', 'impact_score': 3},
            {'date': '2025-02-04', 'event': 'First China Tariffs (10%)', 'impact_score': 5},
            {'date': '2025-03-04', 'event': 'USMCA Tariffs (25%)', 'impact_score': 6},
            {'date': '2025-04-02', 'event': 'Liberation Day - Universal Tariffs', 'impact_score': 10},
            {'date': '2025-04-05', 'event': 'Universal Tariffs Take Effect', 'impact_score': 9},
            {'date': '2025-04-07', 'event': 'Market Crash Continues', 'impact_score': 9},
            {'date': '2025-04-09', 'event': 'Country-Specific Tariffs', 'impact_score': 10}
        ]
        
        # Economic indicators - The quantitative backbone
        self.economic_data = pd.DataFrame({
            'metric': [
                'Stock Market Value Lost', 'Average Tariff Rate', 'Household Cost Increase',
                'Recession Probability', 'GDP Impact Projection', 'Inflation Impact',
                'Countries with Tariffs', 'Global Equity Loss', 'Oil Price Decline',
                'Currency Volatility Index'
            ],
            'pre_trump_2024': [0, 2.5, 0, 22, 0, 2.1, 0, 0, 0, 1.0],
            'trump_peak_disaster': [6800, 15.6, 1300, 60, -8.0, 2.3, 195, 10000, -7, 3.2],
            'unit': ['Billions $', '%', '$ per family', '% probability', '% decline', 
                    '% annual', 'countries', 'Billions $', '% decline', 'index'],
            'severity_score': [10, 9, 8, 9, 10, 6, 10, 10, 7, 8]
        })
        
        # Industry impact data - Sectoral analysis
        self.industry_data = pd.DataFrame({
            'industry': [
                'Technology', 'Automotive', 'Retail', 'Manufacturing', 
                'Agriculture', 'Energy', 'Pharmaceuticals', 'Textiles',
                'Aerospace', 'Chemicals', 'Food Processing', 'Electronics'
            ],
            'import_dependence': [85, 70, 90, 60, 30, 40, 75, 95, 80, 65, 45, 92],
            'tariff_exposure': [9, 8, 10, 7, 6, 5, 8, 10, 9, 7, 6, 10],
            'price_increase': [12, 15, 18, 10, 8, 6, 11, 20, 14, 9, 7, 16],
            'job_risk_score': [8, 9, 7, 8, 5, 4, 6, 9, 8, 7, 5, 9],
            'market_cap_billions': [3200, 850, 1200, 950, 420, 1800, 1100, 180, 650, 480, 350, 1400]
        })
        
        # International retaliation - Global impact
        self.retaliation_data = pd.DataFrame({
            'country': [
                'China', 'European Union', 'Canada', 'Mexico', 'Japan', 'India',
                'South Korea', 'Vietnam', 'Brazil', 'Germany', 'UK', 'Australia'
            ],
            'us_tariff_on_them': [104, 25, 25, 25, 50, 35, 40, 40, 30, 45, 20, 15],
            'their_retaliation': [34, 28, 30, 22, 45, 40, 38, 35, 25, 28, 18, 12],
            'trade_volume_2024': [690, 560, 430, 380, 320, 120, 180, 140, 95, 230, 140, 85],
            'economic_damage_score': [10, 9, 8, 8, 9, 7, 8, 7, 6, 9, 6, 5]
        })
        
        # Predictive scenarios - Future modeling
        self.scenarios = {
            'Best Case': {
                'probability': 0.15,
                'description': 'Quick negotiated resolution',
                'gdp_impact': -2.0,
                'market_recovery': 0.8,
                'timeline_months': 6
            },
            'Moderate Case': {
                'probability': 0.35,
                'description': 'Partial rollback after damage',
                'gdp_impact': -4.5,
                'market_recovery': 0.6,
                'timeline_months': 18
            },
            'Disaster Case': {
                'probability': 0.50,
                'description': 'Full trade war recession',
                'gdp_impact': -8.0,
                'market_recovery': 0.3,
                'timeline_months': 36
            }
        }

if __name__ == "__main__":
    story_engine = EconomicPolicyDataStory()
    print("🎭 Economic Policy Data Story Engine Ready!")
    print("📊 Advanced data storytelling for ML/AI portfolio demonstration")
