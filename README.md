# Retirement Planner (Streamlit)

A comprehensive retirement planning application that combines historical market data with Monte Carlo projections to simulate retirement scenarios.

## Features

### Simulation Approach
- **Unified Simulation**: Uses historical market data for the accumulation phase (pre-retirement) and Monte Carlo projections for the retirement phase
- **Historical Data**: Real market returns from Yahoo Finance for pre-retirement years
- **Monte Carlo**: Statistical projections calibrated to historical returns for retirement years

### Dynamic Withdrawal System
- **CPI-Adjusted Spending**: Adjusts retirement spending based on historical Consumer Price Index (CPI) data
- **Expense Categories**: Break down spending into 8 categories:
  - Food and beverages
  - Housing
  - Apparel
  - Transportation
  - Medical care
  - Recreation
  - Education and communication
  - Other goods and services
- **Interactive Pie Chart**: Visualize and adjust expense category percentages with drag-and-drop interface
- **Two Input Modes**:
  - Total annual expense + category percentages
  - Dollar amounts per category

### Additional Features
- **Daily/Monthly** return frequency toggle
- **Smart data fetching** with historical backfill (SPY→^GSPC, VTI→VTSMX, etc.)
- **Education & Wage Tracking**: Input education level and current wage for future wage growth calculations
- **Interactive charts** for portfolio paths and terminal wealth distribution
- **Modular architecture** with proper separation of concerns
- **Type safety** and comprehensive error handling

## Architecture

```
app/
├── config.py              # Configuration management
├── schemas.py             # Data models and type definitions
├── utils.py               # Utility functions
├── main.py                # Main application entry point
├── services/              # Business logic services
│   ├── data_service.py    # Market data fetching & CPI data loading
│   ├── portfolio_service.py # Portfolio management & category spending
│   └── simulation_service.py # Unified simulation engine
└── components/            # UI components
    ├── sidebar.py         # Input forms (with expense categories & pie chart)
    ├── charts.py          # Visualization
    └── results.py          # Results display
data/
└── CPI.csv                # Historical CPI index data
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
PYTHONPATH="." streamlit run app/main.py --logger.level=debug
```

## Configuration

Edit `app/config.py` to customize:

- Default tickers and weights
- Historical ticker mappings
- UI defaults
- Simulation parameters

## Data Sources

### Market Data
The app automatically maps modern ETFs to historical equivalents:

- **SPY** → **^GSPC** (S&P 500 index, 1950s+)
- **VTI** → **VTSMX** (Vanguard Total Stock Market, 1992+)
- **BND** → **VBMFX** (Vanguard Total Bond Market, 1986+)

### Inflation Data
- **CPI Data**: Historical US Consumer Price Index data from BLS (Bureau of Labor Statistics)
- **Location**: `data/CPI.csv`
- **Usage**: Calculates historical annual inflation rates for dynamic withdrawal adjustments
- **Source**: https://www.bls.gov/cpi/

### Wage Data (Future)
- **Median Usual Weekly Earnings**: Historical wage data by education level
- **Source**: https://www.bls.gov/cps/earnings.htm
- **Status**: Structure in place, data loading to be implemented

## Usage

### Basic Simulation
1. Enter your current age, retirement age, and planning horizon
2. Set your portfolio allocation (tickers and weights)
3. Enter savings and spending amounts
4. Click "Run Simulation" to see results

### Dynamic Withdrawal Mode
1. Enable "Use dynamic withdrawal (CPI-adjusted)" checkbox
2. Choose input mode:
   - **Total + Percentages**: Enter total annual spending and adjust category percentages using the interactive pie chart or sliders
   - **Dollar amounts per category**: Enter specific dollar amounts for each expense category
3. Optionally enter current wage and education level for future wage growth calculations
4. Enable/disable CPI-based inflation adjustment

## Notes

- Uses Adjusted Close for all market data sources
- Annual rebalancing with configurable daily pacing
- Handles data limitations gracefully with proportional scaling
- CPI inflation rates are calculated as year-over-year changes: `(CPI_t / CPI_t-1) - 1`
- Monte Carlo simulations use historical average inflation rate
- Historical simulations use actual year-by-year CPI inflation rates when available
