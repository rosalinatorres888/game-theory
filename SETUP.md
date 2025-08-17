# Setup Instructions for Game Theory Policy Analysis

## Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/rosalinatorres888/game-theory.git
cd game-theory-policy-analysis

# Create virtual environment (recommended)
python -m venv game_theory_env
source game_theory_env/bin/activate  # On Windows: game_theory_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Analysis
```bash
# Execute complete analysis pipeline
cd src
python main_analysis.py

# Run individual components
python game_theory_model.py          # Game theory analysis
python policy_impact_predictor.py    # ML validation
python data_utilities.py             # Data processing
python visualization_helpers.py      # Generate charts
```

### 3. View Results
```bash
# Check outputs folder
ls ../outputs/

# Open interactive presentations
open ../outputs/stakeholder-presentation.html
open ../outputs/payoff_matrix_demo.html
open ../outputs/policy_dashboard_demo.html
```

## Development Setup

### For Data Scientists
```bash
# Launch Jupyter for exploration
jupyter notebook

# Open key notebooks
# - Game Theory Analysis.ipynb
# - Data Processing Pipeline.ipynb
# - Visualization Development.ipynb
```

### For Web Development
```bash
# Serve interactive demos locally
cd outputs
python -m http.server 8000
# Visit http://localhost:8000
```

## Project Structure

```
src/          # Core analysis modules
├── main_analysis.py              # Master pipeline
├── game_theory_model.py          # Nash equilibrium analysis
├── nash_equilibrium_solver.py    # Mathematical solver
├── policy_impact_predictor.py    # ML validation
├── data_utilities.py             # Data processing
├── visualization_helpers.py      # Professional charts
└── economic_policy_data_story.py # Historical context

data/         # Datasets
├── game_theory_policy_analysis.csv    # Core scenarios
├── clean_bls_employment_data.csv       # BLS projections
└── clean_international_data.csv        # Global benchmarks

docs/         # Documentation
└── methodology.md                      # Technical methodology

outputs/      # Generated results
├── stakeholder-presentation.html       # Executive presentation
├── payoff_matrix_demo.html            # Interactive heatmaps
├── policy_dashboard_demo.html         # Comparative analysis
└── executive_summary_*.md             # Generated reports

blog/         # Accessible content
└── stem-investment-blog-post.md       # Non-technical blog post
```

## Key Features

### 🎮 Game Theory Analysis
- **Nash Equilibrium Calculation** for policy scenarios
- **Dominant Strategy Identification** (STEM investment wins)
- **Welfare Analysis** showing 100% vs 33% efficiency
- **Interactive Payoff Matrices** with hover details

### 🤖 Machine Learning Validation
- **Random Forest & Gradient Boosting** ensemble models
- **94.3% prediction accuracy** on policy outcomes
- **Cross-validation** with temporal and geographic splits
- **Feature importance analysis** identifying key drivers

### 📊 Professional Visualizations
- **Interactive Plotly dashboards** for stakeholder presentations
- **Publication-ready matplotlib** charts for academic use
- **Executive KPI indicators** with gauge charts
- **Comparative analysis** across multiple policy scenarios

### 📈 Real Data Integration
- **Bureau of Labor Statistics** employment projections
- **NAEP education performance** by state
- **International benchmarks** from 15 countries
- **Current economic indicators** validating predictions

## Dependencies

### Core Requirements
- **Python 3.9+** for modern language features
- **pandas, numpy** for data manipulation
- **scikit-learn** for machine learning
- **plotly, matplotlib** for visualizations

### Optional Enhancement
- **kaleido** for PNG export from Plotly
- **streamlit** for web app deployment
- **jupyter** for interactive development

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the right directory
cd src/
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Missing Data:**
```bash
# Data utilities will generate sample data if files missing
python data_utilities.py
```

**Visualization Errors:**
```bash
# Install kaleido for image export
pip install kaleido

# Or use HTML-only mode
# Set save_png=False in visualization calls
```

## Contact

**Rosalina Torres**  
Data Analytics Engineering Student  
Northeastern University

- 📧 torres.ros@northeastern.edu
- 💼 [LinkedIn](https://www.linkedin.com/in/rosalina2)
- 🌐 [Portfolio](https://rosalina.sites.northeastern.edu)
- 📚 [GitHub](https://github.com/rosalinatorres888)

---

*This project demonstrates the intersection of economic theory, data science, and real-world policy analysis—perfect for ML/AI engineering roles requiring analytical thinking and practical application skills.*
