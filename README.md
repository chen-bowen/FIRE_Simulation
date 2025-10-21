# Retirement Planner (Streamlit)

A modular retirement planning app with two simulation modes:

- **Historical backtests** using Yahoo Finance data
- **Monte Carlo simulation** calibrated to historical returns

## Features

- **Daily/Monthly** return frequency toggle
- **Smart data fetching** with historical backfill (SPY→^GSPC, VTI→VTSMX, etc.)
- **Flexible inputs** for savings, spending, allocation, inflation
- **Interactive charts** for portfolio paths and terminal wealth distribution
- **Modular architecture** with proper separation of concerns
- **Type safety** and comprehensive error handling

## Architecture

```
app/
├── config.py              # Configuration management
├── models.py              # Data models and type definitions
├── utils.py               # Utility functions
├── main.py               # Main application entry point
├── services/             # Business logic services
│   ├── data_service.py   # Market data fetching
│   ├── portfolio_service.py # Portfolio management
│   └── simulation_service.py # Simulation engines
└── components/           # UI components
    ├── sidebar.py       # Input forms
    ├── charts.py        # Visualization
    └── results.py       # Results display
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

The app automatically maps modern ETFs to historical equivalents:

- **SPY** → **^GSPC** (S&P 500 index, 1950s+)
- **VTI** → **VTSMX** (Vanguard Total Stock Market, 1992+)
- **BND** → **VBMFX** (Vanguard Total Bond Market, 1986+)

## Notes

- Uses Adjusted Close for all data sources
- Annual rebalancing with configurable daily pacing
- Handles data limitations gracefully with proportional scaling
