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

### Visualization Features

- **Portfolio Quantiles Chart**: Interactive chart showing portfolio value projections over time
  - Displays percentile bands (P10, P25, Median, P75, P90) with shaded confidence intervals
  - Quantile lines are visible on the chart but hidden from the legend for a cleaner interface
  - Retirement threshold line with label positioned in the top-left corner
  - "Can Retire" marker shows when the median portfolio value first reaches the retirement threshold
  - Age and planned retirement markers for easy reference
- **Interactive charts** for portfolio paths, spending analysis, and terminal wealth distribution
- **Multiple chart views**: Toggle between portfolio quantiles, spending quantiles, and spending vs returns breakdown

### Additional Features

- **Daily/Monthly** return frequency toggle
- **Smart data fetching** with historical backfill (SPY→^GSPC, VTI→VTSMX, etc.)
- **Education & Wage Tracking**: Input education level and current wage for future wage growth calculations
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

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### Verify All Packages Are Installed

```bash
# Check that all required packages are installed
pip check

# Or verify specific packages
python -c "import streamlit, yfinance, numpy, pandas, plotly; print('All packages installed successfully!')"
```

## Run

### Local Development

```bash
# Run the app directly
streamlit run app/main.py

# With debug logging
streamlit run app/main.py --logger.level=debug

# With custom port
streamlit run app/main.py --server.port=8502
```

The app will be available at `http://localhost:8501` (or your specified port)

**Note:** The `app/main.py` file includes path setup code that ensures imports work correctly whether run locally or on Streamlit Cloud.

## Deployment

### Streamlit Cloud (Recommended)

1. **Push your code to GitHub**

   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the main file path: `app/main.py`
   - Click "Deploy"

3. **Important Notes for Streamlit Cloud**
   - Ensure `requirements.txt` is in the root directory (✓ already present)
   - The main file path should be set to `app/main.py`
   - The app will automatically install dependencies from `requirements.txt`
   - Data files in `data/` directory will be included automatically
   - The `app/main.py` file includes path setup code to handle imports correctly
   - No additional configuration needed

### Other Deployment Options

#### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t retirement-planner .
docker run -p 8501:8501 retirement-planner
```

#### Traditional Server Deployment

1. **Install dependencies on server**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run with process manager (e.g., systemd, supervisor)**

   ```bash
   streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0
   ```

3. **Use reverse proxy (nginx) for production**
   - Configure nginx to proxy requests to `localhost:8501`
   - Enable SSL/TLS certificates

### Deployment Checklist

- [x] All dependencies listed in `requirements.txt`
- [x] Data files present in `data/` directory
- [x] Main entry point (`app/main.py`) is configured correctly
- [x] Path setup code is present in `app/main.py` for proper imports
- [ ] Environment variables configured (if needed)
- [ ] Port 8501 is accessible (or configured port)
- [ ] Internet access for Yahoo Finance API calls

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

### Retirement Phase Only (Already Retired)

The app supports users who are already retired:
- Set **Retirement age** to be less than or equal to **Current age**
- The simulation will skip the accumulation phase and start directly with retirement withdrawals
- Annual savings contributions are automatically disabled (not applicable for already-retired users)
- All simulation periods will be treated as retirement phase with withdrawals

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
- Hybrid simulation: Uses actual year-by-year CPI rates for accumulation phase (when available), historical averages for retirement phase projections
