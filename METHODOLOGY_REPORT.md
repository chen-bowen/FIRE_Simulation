# Methodology Report: Retirement Planning Application

## Executive Summary

This report documents the methodologies employed in the Retirement Planning Application, which combines historical market data with Monte Carlo projections to model retirement scenarios. The application uses a hybrid simulation approach: historical data for the accumulation phase and Monte Carlo for the retirement phase.

---

## 1. Data Acquisition and Processing

### 1.1 Market Data

**Source**: Yahoo Finance via `yfinance` library

**Key Features**:

- **Historical Ticker Mapping**: Modern ETFs (SPY, VTI, BND) mapped to historical equivalents (^GSPC, VTSMX, VBMFX) to extend data to 1950s-1990s
- **Return Calculation**: Daily uses percentage change, monthly resamples to month-end then calculates percentage change
- **Caching**: LRU cache for performance optimization
- **Error Handling**: Graceful fallback with proportional weight scaling when data unavailable

### 1.2 Inflation Data

**Source**: Bureau of Labor Statistics (BLS) CPI data from CSV files

**Methodology**:

- Overall CPI: Historical annual values from CSV files
- Category-Specific CPI: Eight expense categories (Food, Housing, Apparel, Transportation, Medical, Recreation, Education, Other)
- Inflation calculation: Year-over-year percentage change from CPI values
- Average rates computed for projections when year-by-year data unavailable

### 1.3 Wage Data

**Source**: BLS wage data by education level from CSV files

**Methodology**:

- Six education levels from "Less than high school" to "Doctorate"
- Wage growth: Year-over-year percentage change
- Projection: Compound growth based on historical growth rates
- Conversion: Weekly to annual using 52-week multiplier

---

## 2. Portfolio Management

### 2.1 Portfolio State

Tracks three components via PortfolioState dataclass:

- **Balance**: Total portfolio value
- **Weights**: Target allocation percentages (sum to 1.0)
- **Asset Values**: Actual dollar amounts per asset

### 2.2 Rebalancing

**Rationale**: Maintains target risk-return profile by resetting allocations. Without rebalancing, portfolios drift as higher-performing assets grow proportionally larger, increasing risk beyond intended allocation.

**Methodology**:

- **Frequency**: Annual rebalancing at end of each year
- **Detection**: Year change detection for historical data, period-based for projections
- **Process**: At rebalancing points, all assets reset to target weights based on total balance
- Between rebalancing, assets drift naturally based on individual returns

### 2.3 Contributions and Withdrawals

**Methodology**:

- Net flow: Contribution minus withdrawal at start of period
- Allocation: Net flows allocated proportionally to existing asset values
- Period conversion: Daily uses 252 periods per year, monthly uses 12 periods per year
- **Pacing**: Even distribution across all periods (default)

### 2.4 Wage-Based Contributions

**Methodology**:

- Projects annual wage based on current age/year, education level, and historical growth rates
- Calculation: Annual contribution equals annual wage multiplied by savings rate
- Pre-calculates wages for all accumulation years to avoid repeated calculations

---

## 3. Dynamic Withdrawal System

### 3.1 Expense Categories

**Methodology**: Eight categories with two input modes:

1. Percentage mode: Total annual expense + category percentages (must sum to 100%)
2. Dollar amount mode: Dollar amounts per category

**Category-Specific Inflation**: Uses category CPI if available, otherwise falls back to overall CPI or user-specified rate.

### 3.2 Inflation Adjustment

**Formula**: Cumulative inflation adjustment using compound growth: spending increases by inflation rate each year.

**Methodology**: Uses actual year-by-year CPI rates per category when available (historical accumulation phase), falls back to historical averages for projections (retirement phase).

---

## 4. Simulation Methodology

### 4.1 Hybrid Simulation (Default Approach)

**Methodology**: Combines historical data for accumulation phase with Monte Carlo projections for retirement phase.

**Accumulation Phase**:

- Uses available historical market data (rolling window backtesting)
- Extracts contiguous blocks of historical returns matching accumulation horizon
- If insufficient historical data: Extends with Monte Carlo projections calibrated from available data
- Steps through applying contributions, asset returns, and annual rebalancing

**Retirement Phase**:

- Always uses Monte Carlo projections
- Calibrated to historical data statistics (mean log-returns and covariance matrix)
- Uses Cholesky decomposition for correlated asset returns
- Path generation similar to accumulation but with generated returns instead of historical

**Calibration Process**:

1. Log-return transformation of historical arithmetic returns
2. Parameter estimation: Mean log-returns and covariance matrix from historical data
3. Cholesky decomposition for correlated sampling (with numerical stability adjustments)

**Path Generation**:

- For accumulation: Randomly selects historical return blocks (or generates Monte Carlo if insufficient data)
- For retirement: Generates Monte Carlo returns using calibrated parameters
- Combines into full path, steps through with contributions/withdrawals and rebalancing

**Success Criteria**: Portfolio balance remains positive throughout retirement phase (not just terminal balance).

**Statistics**: Success rate, percentiles (P10, P50/P90), sample paths for visualization (max 100).

**Crypto Asset Handling** (if applicable): Return capping for normal volatility, rare extreme events injection, volatility dampening when projecting beyond available data.

---

## 5. Statistical Analysis

### 5.1 Success Rate

**Definition**: Portfolio balance positive throughout retirement phase (not just terminal balance). A path is successful if balance remains positive during all retirement periods and terminal balance is positive.

**Calculation**: Success rate equals number of successful paths divided by total paths.

### 5.2 Percentiles

Calculated across all paths at each time period:

- P10 (10th percentile): 10% of paths fall below this value
- P50 (median): 50% of paths fall below this value
- P90 (90th percentile): 90% of paths fall below this value

### 5.3 Terminal Wealth Distribution

Terminal balances analyzed with mean, median, percentiles, and histogram visualization.

---

## 6. Visualization

**Tools**: Plotly for interactive visualizations

**Key Visualizations**:

1. **Portfolio Charts**: Shaded confidence bands (P5-P95, P10-P90, P25-P75), median path, age/year-based x-axis
2. **Allocation Explorer**: Interactive slider showing year-by-year allocation accounting for rebalancing, drift, contributions/withdrawals
3. **Savings Breakdown**: Dual-axis chart showing cumulative portfolio balance, contributions, returns (primary), and annual contributions/returns (secondary)

---

## 7. Validation and Error Handling

**Input Validation**: Age constraints, non-negative financial values, weights sum to 1.0, category percentages sum to 100%.

**Error Handling**: Custom exceptions (DataError, SimulationError, ValidationError) with graceful degradation:

- Missing category CPI → overall CPI
- Missing wage data → fixed contributions
- Insufficient historical data → bootstrap or Monte Carlo extension

---

## 8. Performance Optimizations

**Caching**: LRU cache for market data, in-memory for CPI/wage data, session state for simulation results.

**Efficiency**: Vectorized NumPy operations, pre-calculated annual wages, limited sample paths (100 max) for visualization.

---

## 9. Technical Architecture

**Modular Design**: Separation of concerns across services:

- **DataService**: Data acquisition and processing
- **PortfolioService**: Portfolio mathematics and state management
- **SimulationService**: Hybrid simulation engine (historical accumulation, Monte Carlo retirement)
- **Components**: UI components (sidebar, charts, results)

**Type Safety**: Dataclasses for all data structures (SimulationParams, SimulationResult, PortfolioState, WithdrawalParams, ExpenseCategory).

**Configuration**: Centralized configuration with defaults, ticker mappings, and simulation parameters.

---

## 10. Limitations and Assumptions

### Key Assumptions

- **Market**: Historical returns representative of future; log-returns normally distributed; correlations stable
- **Portfolio**: Annual rebalancing (no costs), perfect execution, no taxes
- **Inflation**: Historical CPI trends continue; category rates stable; no deflation
- **Wage**: Historical growth continues; education level determines trajectory; no career changes

### Known Limitations

- **Data**: Limited historical data for some assets; CPI/wage data may be incomplete
- **Model**: No taxes or transaction costs; simplified rebalancing (annual only); no Social Security/pensions
- **Simulation**: Monte Carlo assumes normality; may miss extreme tail events; historical limited by data

---

## 11. Future Enhancements

**Potential**: Tax modeling, transaction costs, Social Security, healthcare costs, longevity risk, regime detection, stress testing.

**Data**: Additional asset classes, international data, alternative data, real-time updates.

---

## 12. Conclusion

The application employs a hybrid approach combining historical backtesting and Monte Carlo simulation. Key strengths include robust data handling, sophisticated portfolio management, category-specific inflation adjustments, wage-based contributions, and comprehensive visualization. The modular architecture supports future enhancements.

---

## References

- Yahoo Finance API (via yfinance library)
- Bureau of Labor Statistics (BLS) CPI and wage data
- Modern Portfolio Theory
- Monte Carlo simulation techniques

---

_Report Generated: 2024_
_Application Version: 1.0_
