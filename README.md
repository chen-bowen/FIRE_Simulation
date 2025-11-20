# Financial Independence, Retire Early (FIRE)

A comprehensive Financial Independence, Retire Early (FIRE) planning application that combines historical market data with Monte Carlo projections to simulate early retirement scenarios. Features category-specific CPI-adjusted spending, wage-based savings, and detailed financial planning tools to help you achieve financial independence.

## Features

### Simulation Approach

- **Hybrid Simulation**: Uses historical market data for the accumulation phase (pre-retirement) and Monte Carlo projections for the retirement phase
- **Historical Data**: Real market returns from Yahoo Finance for pre-retirement years
- **Monte Carlo**: Statistical projections calibrated to historical returns for retirement years
- **Multiple Asset Classes**: Support for stocks, bonds, international stocks, real estate, commodities, crypto, and cash

### Dynamic Withdrawal System (Detailed Plan Mode)

- **Category-Specific CPI Adjustment**: Each expense category uses its own historical inflation rate from BLS CPI data
  - Medical care typically inflates faster (~4-5% annually)
  - Technology/communication may inflate slower
  - Housing, food, transportation have their own rates
- **8 Expense Categories**:
  - Food and beverages
  - Housing
  - Apparel
  - Transportation
  - Medical care
  - Recreation
  - Education and communication
  - Other goods and services
- **Interactive Category Management**:
  - Visualize and adjust expense category percentages with interactive pie chart
  - Preset templates based on typical US household spending
  - Education-level based presets for more accurate spending patterns
  - Fine-tune individual category percentages with sliders
- **Wage-Based Retirement Spending**: Calculate retirement spending as a percentage of pre-retirement spending (replacement ratio)

### Wage-Based Savings & Spending

- **Wage Growth Projections**: Uses historical wage data by education level to project future income
- **Savings Rate Options**:
  - Constant savings rate (% of income)
  - Age-based savings rate profiles (e.g., increase savings as you age)
- **Education Levels Supported**:
  - Less than high school
  - High school
  - Some college
  - Bachelor's degree
  - Master's degree
  - Professional degree
  - Doctorate
- **Pre-Retirement Spending Tracking**: Automatically tracks spending during accumulation phase for replacement ratio calculations

### Visualization Features

- **Portfolio Performance Tab**:
  - Interactive portfolio quantiles chart showing value projections over time
  - Displays percentile bands (P10, P25, Median, P75, P90) with shaded confidence intervals
  - Retirement threshold line and "Can Retire" markers
  - Age and planned retirement markers for easy reference
- **Savings & Returns Tab**:
  - Detailed breakdown of savings contributions vs. investment returns over time
  - Shows how wage growth affects contributions
  - Available when using "Detailed Plan" mode with wage-based savings
- **Terminal Wealth Histogram**: Distribution of final portfolio values across all simulation paths
- **Pre-Simulation Summary**: Quick overview of timeline, financial inputs, and portfolio allocation

### Additional Features

- **Daily/Monthly** return frequency toggle
- **Smart data fetching** with historical backfill (SPY→^GSPC, VTI→VTSMX, BND→VBMFX, etc.)
- **Portfolio Presets**: Quick selection of Conservative (30/70), Moderate (60/40), or Aggressive (90/10) allocations
- **Custom Portfolio Allocation**: Fine-tune asset class weights with proportional sliders
- **Input Validation**: Comprehensive validation with helpful error messages
- **Results Caching**: Simulation results are cached and automatically invalidated when inputs change
- **Modular Architecture**: Clean separation of concerns for maintainability

## Architecture

The application follows a modular architecture with clear separation between UI components, business logic, and data services:

```
app/
├── config.py                    # Configuration management
├── schemas.py                   # Data models and type definitions
├── utils.py                     # Utility functions (validation, formatting)
├── main.py                      # Lightweight orchestrator (~130 lines)
├── services/                    # Business logic services
│   ├── data_service.py         # Market data fetching & CPI/wage data loading
│   ├── portfolio_service.py    # Portfolio management & category spending calculations
│   ├── simulation_service.py   # Core simulation engine
│   └── simulation_controller.py # Simulation orchestration & caching
└── components/                  # UI components
    ├── sidebar.py              # Input forms (with expense categories & pie charts)
    ├── charts.py               # Visualization components
    ├── results.py              # Results display & validation messages
    └── summary.py             # Pre-simulation summary cards & input summary
data/
├── CPI/                        # Category-specific CPI data
│   ├── All_CPI.csv
│   ├── Food_CPI.csv
│   ├── Housing CPI.csv
│   ├── Medical_CPI.csv
│   └── ... (8 categories)
└── Income/                     # Wage data by education level
    ├── All_educational_levels_income.csv
    ├── Bachelors_Income.csv
    ├── Master's_Income.csv
    └── ... (7 education levels)
```

### Key Design Principles

- **Separation of Concerns**: UI components, business logic, and data access are cleanly separated
- **Single Responsibility**: Each component/service has a focused purpose
- **Type Safety**: Comprehensive type hints and data validation
- **Error Handling**: Graceful handling of edge cases and data limitations
- **Maintainability**: Refactored codebase with `main.py` reduced from 541 to 131 lines

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
- UI defaults (ages, balances, spending)
- Simulation parameters (Monte Carlo paths, random seed)
- Crypto simulation parameters (volatility, extreme events)

## Data Sources

### Market Data

The app automatically maps modern ETFs to historical equivalents for longer data history:

- **SPY** → **^GSPC** (S&P 500 index, 1950s+)
- **VTI** → **VTSMX** (Vanguard Total Stock Market, 1992+)
- **BND** → **VBMFX** (Vanguard Total Bond Market, 1986+)
- **QQQ** → **^IXIC** (NASDAQ Composite)
- **EFA** → **^EFA** (International stocks)

### Inflation Data (CPI)

- **Source**: Bureau of Labor Statistics (BLS)
- **Location**: `data/CPI/` directory with category-specific CSV files
- **Categories**: 8 expense categories with historical CPI data
- **Usage**:
  - Calculates category-specific inflation rates for dynamic withdrawal adjustments
  - Provides more realistic spending projections than a single inflation rate
- **Update Frequency**: Data files can be updated from BLS website

### Wage Data

- **Source**: Bureau of Labor Statistics (BLS) - Median Usual Weekly Earnings
- **Location**: `data/Income/` directory with education-level specific CSV files
- **Education Levels**: 7 levels from "Less than high school" to "Doctorate"
- **Usage**:
  - Projects future wage growth based on education level and age
  - Calculates wage-based savings contributions
  - Estimates retirement spending using replacement ratio
- **Update Frequency**: Data files can be updated from BLS website

## Usage

### Basic Simulation (Simple Amount Mode)

1. Enter your current age, retirement age, and planning horizon
2. Set your portfolio allocation (use presets or customize)
3. Enter current savings and annual contribution amount
4. Enter annual retirement spending
5. Click "Run Simulation" to see results

### Detailed Plan Mode (Dynamic Withdrawal)

1. Select "Detailed plan" in the Retirement Spending section
2. **Pre-retirement Savings**:
   - Choose savings rate style (constant or age-based profile)
   - Enter current annual wage (or let it estimate from education level)
   - Select your education level
   - Adjust savings rate(s)
3. **Retirement Spending Adjustment**:
   - Set replacement ratio (% of pre-retirement spending)
   - View estimated retirement spending
4. **Spending Categories**:
   - Choose a category template (Typical US Household, Conservative, or education-based)
   - Fine-tune category percentages if needed
   - View interactive pie chart
5. Click "Run Simulation" to see results with category-specific inflation adjustments

### Retirement Phase Only (Already Retired)

The app supports users who are already retired:

- Set **Retirement age** to be less than or equal to **Current age**
- The simulation will skip the accumulation phase and start directly with retirement withdrawals
- Annual savings contributions are automatically disabled
- All simulation periods will be treated as retirement phase with withdrawals
- Dynamic withdrawal with CPI adjustments is still available

### Portfolio Allocation

- **Preset Mode**: Quick selection of Conservative (30/70), Moderate (60/40), or Aggressive (90/10)
- **Custom Mode**:
  - Select asset classes to include
  - Adjust weights with proportional sliders (others adjust automatically to maintain 100%)
  - Supports multiple asset classes: US Stocks, International Stocks, Bonds, Cash, Crypto, Real Estate, Commodities

### Advanced Settings

Expand the "Advanced Settings" section to:

- Change frequency (daily/monthly)
- Adjust date range for historical data
- Override tickers/weights manually
- Configure Monte Carlo paths and random seed
- Adjust inflation rate (for Simple Amount mode)

## Technical Details

### Simulation Methodology

- **Accumulation Phase**: Uses actual historical returns year-by-year from the start date
- **Retirement Phase**: Uses Monte Carlo simulation with returns calibrated to historical statistics
- **Rebalancing**: Annual rebalancing to maintain target portfolio weights
- **Frequency**: Supports both daily and monthly simulation frequencies

### Dynamic Withdrawal Calculation

For each period in retirement:

1. Calculate years into retirement: `years_into_retirement = periods_into_retirement / periods_per_year`
2. For each expense category:
   - Get category-specific inflation rate (from CPI data) or use general rate
   - Apply cumulative inflation: `category_amount × (1 + category_rate)^years_into_retirement`
3. Sum all categories to get total annual withdrawal
4. Convert to per-period withdrawal based on frequency

### Data Limitations & Handling

- **Historical Data Availability**: Some assets (e.g., crypto) have limited history
- **Monte Carlo Extension**: When historical data runs out, the simulation extends using Monte Carlo projections
- **Crypto Assets**: Special handling with volatility dampening and extreme event modeling
- **Data Validation**: Automatic alignment of portfolio weights with available data

## Notes

- Uses Adjusted Close for all market data sources
- Annual rebalancing with configurable daily pacing (pro-rata or monthly-boundary)
- Handles data limitations gracefully with proportional scaling
- CPI inflation rates are calculated as year-over-year changes: `(CPI_t / CPI_t-1) - 1`
- Category-specific inflation rates use historical averages from BLS data
- Wage growth projections use historical trends by education level and age
- Results are cached and automatically invalidated when inputs change (detected via input hash)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
