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
- **Caching**: LRU cache (maxsize=64) for performance optimization to avoid redundant API calls
- **Error Handling**: Graceful fallback with proportional weight scaling when data unavailable

**Key Decision**: Historical ticker mapping extends data coverage significantly. Rather than limiting simulations to recent ETF data (often 2000s+), mapping modern ETFs to their historical index equivalents provides 50+ years of market history, enabling more robust backtesting and parameter calibration.

### 1.2 Inflation Data

**Source**: Bureau of Labor Statistics (BLS) CPI data from CSV files

**Methodology**:

- Overall CPI: Historical annual values from CSV files
- Category-Specific CPI: Eight expense categories (Food, Housing, Apparel, Transportation, Medical, Recreation, Education, Other)
- Inflation calculation: Year-over-year percentage change from CPI values: `(CPI_t / CPI_t-1) - 1`
- Average rates computed for projections when year-by-year data unavailable

**Key Decision**: Category-specific inflation rates capture real-world spending dynamics. Medical care inflation often exceeds overall inflation, while apparel inflation is typically lower. Using category-specific rates provides more realistic retirement spending projections than a single overall rate. When category data is unavailable, the system falls back to overall CPI to ensure robustness.

### 1.3 Wage Data

**Source**: BLS wage data by education level from CSV files

**Methodology**:

- Six education levels from "Less than high school" to "Doctorate"
- Wage growth: Year-over-year percentage change calculated from historical data
- Projection: Compound growth based on historical growth rates: `wage_t = wage_0 * (1 + growth_rate)^t`
- Conversion: Weekly to annual using 52-week multiplier

**Key Decision**: Education-based wage projections enable realistic contribution modeling. Instead of assuming fixed contributions, the system projects wage growth based on education level and age. This allows users to model savings rates (e.g., "save 15% of income") rather than fixed dollar amounts, reflecting how people actually save. Wages are pre-calculated for all accumulation years and cached to avoid repeated calculations during simulation.

---

## 2. Portfolio Management

### 2.1 Portfolio State

Tracks three components via PortfolioState dataclass:

- **Balance**: Total portfolio value
- **Weights**: Target allocation percentages (sum to 1.0)
- **Asset Values**: Actual dollar amounts per asset

**Key Decision**: Separating target weights from actual asset values enables accurate tracking of portfolio drift. During periods between rebalancing, asset values drift from target weights due to differential returns. The state tracks both to accurately model rebalancing behavior.

### 2.2 Rebalancing

**Rationale**: Maintains target risk-return profile by resetting allocations. Without rebalancing, portfolios drift as higher-performing assets grow proportionally larger, increasing risk beyond intended allocation.

**Methodology**:

- **Frequency**: Annual rebalancing at end of each year
- **Detection**:
  - Historical data: Year change detection (`period_index.year != prev_index.year`)
  - Monte Carlo: Period-based detection (`(period_number + 1) % periods_per_year == 0`)
- **Process**: At rebalancing points, all assets reset to target weights based on total balance: `new_asset_values = total_balance * target_weights`
- Between rebalancing, assets drift naturally based on individual returns

**Key Decisions**:

1. **Annual frequency**: Annual rebalancing balances realism (most investors rebalance yearly) with computational efficiency. Daily or monthly rebalancing would add complexity without meaningful improvement for long-horizon retirement planning.

2. **End-of-year rebalancing**: Rebalancing occurs at the end of each year (not the start) to allow natural drift during the year. This models typical investor behavior where portfolio review happens annually.

3. **Complete rebalancing**: Assets are reset to exact target weights, not partially rebalanced. This simplifies the model and aligns with target-date fund behavior. In practice, investors may rebalance partially or with thresholds, but complete annual rebalancing captures the essential risk-management effect.

### 2.3 Contributions and Withdrawals

**Methodology**:

- Net flow: Contribution minus withdrawal at start of period: `net_flow = contrib - spend`
- Allocation: Net flows allocated proportionally to existing asset values: `asset_values *= (1 + net_flow / balance)`
- Period conversion: Daily uses 252 periods per year, monthly uses 12 periods per year
- **Pacing**: Even distribution across all periods (default "pro-rata" mode)

**Key Decision**: Proportional allocation of contributions/withdrawals maintains portfolio composition between rebalancing events. When money flows in or out, it's allocated proportionally to existing asset holdings, preserving the current allocation until the next rebalancing. This models how real investors typically invest new money or withdraw funds.

### 2.4 Wage-Based Contributions

**Methodology**:

- Projects annual wage based on current age/year, education level, and historical growth rates
- Calculation: Annual contribution equals annual wage multiplied by savings rate: `annual_contrib = annual_wage * savings_rate`
- Pre-calculates wages for all accumulation years to avoid repeated calculations during simulation

**Key Decision**: Wage-based contributions provide more realistic modeling than fixed dollar amounts. Users can specify "save 15% of income" rather than "$10,000 per year," which better reflects how people actually save. The system caches annual wages for each year during accumulation to optimize performance during multi-path simulations (1000+ paths).

---

## 3. Dynamic Withdrawal System

### 3.1 Expense Categories

**Methodology**: Eight categories with two input modes:

1. Percentage mode: Total annual expense + category percentages (must sum to 100%)
2. Dollar amount mode: Dollar amounts per category

**Category-Specific Inflation**: Uses category CPI if available, otherwise falls back to overall CPI or user-specified rate.

**Key Decisions**:

1. **Eight categories**: Aligns with BLS CPI categories, enabling category-specific inflation rates from official government data. Categories match real-world spending patterns and available inflation data.

2. **Education-based presets**: Higher education levels automatically adjust expense distributions (e.g., higher education/communication spending, lower basic needs). This captures demographic spending patterns without requiring users to manually adjust all categories.

3. **Dual input modes**: Users can specify either total spending with percentages or direct dollar amounts per category, providing flexibility for different use cases.

### 3.2 Inflation Adjustment

**Formula**: Cumulative inflation adjustment using compound growth: `spending_t = spending_0 * (1 + inflation_rate)^t`

**Methodology**: Inflation adjustments apply only during the retirement phase (no spending occurs during accumulation). Uses historical average category-specific CPI rates when available for dynamic withdrawals, otherwise falls back to overall historical average inflation rate. The same average rate is applied consistently across all retirement periods.

**Key Decisions**:

1. **Retirement-only inflation**: Inflation applies only during retirement, not accumulation. During accumulation, contributions are modeled with wage growth but spending is zero, so inflation is irrelevant.

2. **Average rates for projections**: When projecting future retirement years, the system uses historical average inflation rates (calculated once) rather than attempting to project year-by-year inflation. This simplification is necessary since future inflation is unknown, and using historical averages provides reasonable projections.

3. **Category-specific application**: Each expense category uses its own historical average inflation rate if available. This captures real-world dynamics where medical inflation (typically 4-5% historically) exceeds overall inflation (typically 2-3%), providing more accurate projections.

4. **Cumulative compounding**: Inflation compounds over time using `(1 + r)^t` rather than linear adjustment. This accurately models how spending increases year-over-year with inflation.

---

## 4. Simulation Methodology

### 4.1 Hybrid Simulation

**Methodology**: The application uses a hybrid simulation approach that combines historical data for the accumulation phase with Monte Carlo projections for the retirement phase. This is the sole simulation method implemented.

**Rationale for Hybrid Approach**:

1. **Historical accumulation**: Using actual historical returns for accumulation provides realism and captures real market events (recessions, bull markets). This is especially valuable for long accumulation periods (20-40 years) where substantial historical data exists.

2. **Monte Carlo retirement**: Retirement periods extend into the future where no historical data exists. Monte Carlo projections calibrated to historical statistics provide probabilistic forecasts for unknown future periods.

3. **Seamless transition**: The approach seamlessly combines both methods in a single path, randomly selecting historical blocks for accumulation and generating Monte Carlo returns for retirement, then stepping through both phases together.

**Accumulation Phase**:

- Uses available historical market data (rolling window backtesting)
- Extracts contiguous blocks of historical returns matching accumulation horizon
- Randomly selects start point for each simulation path: `start = random(0, available_periods - accumulation_periods + 1)`
- If insufficient historical data: Extends with Monte Carlo projections calibrated from available data
- Steps through applying contributions, asset returns, and annual rebalancing

**Key Decision**: Using contiguous blocks (not random sampling with replacement) preserves temporal correlation and market regime effects. A 30-year accumulation period uses 30 consecutive years of actual market history, capturing how market conditions evolve over time.

**Retirement Phase**:

- Always uses Monte Carlo projections
- Calibrated to historical data statistics (mean log-returns and covariance matrix)
- Uses Cholesky decomposition for correlated asset returns: `log_returns = z @ L.T + mean` where `L` is Cholesky factor and `z` is standard normal
- Path generation similar to accumulation but with generated returns instead of historical

**Key Decision**: Using log-returns for Monte Carlo (not arithmetic returns) ensures returns compound correctly and prevents negative portfolio values. The transformation: `log_return = log(1 + arithmetic_return)` ensures statistical properties are preserved while enabling proper compounding.

**Calibration Process**:

1. Log-return transformation: `log_returns = log(1 + arithmetic_returns)`
2. Parameter estimation:
   - Mean log-returns: `mean = mean(log_returns, axis=0)`
   - Covariance matrix: `cov = cov(log_returns.T)`
3. Cholesky decomposition: `L = cholesky(cov + jitter*I)` where `jitter = 1e-12` for numerical stability
4. Generate correlated returns: `z ~ N(0,1)`, then `log_returns = z @ L.T + mean`

**Key Decisions**:

1. **Covariance matrix**: Captures asset correlations from historical data. A diversified portfolio's risk depends on correlations between assets, not just individual volatilities.

2. **Numerical stability jitter**: Adding `1e-12 * I` to the covariance matrix before Cholesky decomposition prevents numerical errors from near-singular matrices while having negligible impact on results.

3. **Single calibration**: The Monte Carlo parameters are calibrated once from all historical data, not separately for accumulation vs. retirement. This ensures consistency and uses the full data for parameter estimation.

**Path Generation**:

- For accumulation: Randomly selects historical return blocks (or generates Monte Carlo if insufficient data)
- For retirement: Generates Monte Carlo returns using calibrated parameters
- Combines into full path: `full_returns = concatenate(historical_block, monte_carlo_returns)`
- Steps through entire path with contributions/withdrawals and rebalancing

**Key Decision**: Each simulation path is independent. For 1000 paths, 1000 different random selections of historical accumulation periods and 1000 different Monte Carlo retirement sequences are generated. This provides comprehensive probabilistic coverage of possible outcomes.

**Success Criteria**: Portfolio balance remains positive throughout retirement phase (not just terminal balance). A path fails if the portfolio reaches zero or negative at any point during retirement, even if it recovers later.

**Key Decision**: Requiring positive balance throughout retirement is stricter than requiring only positive terminal balance. This prevents paths that "recover" from zero, which would be impossible in reality (you cannot withdraw from an empty portfolio). This criterion aligns with real-world portfolio failure.

**Statistics**: Success rate, core percentiles (P10, P50 median, P90), sample paths for visualization (max 100). Additional percentiles (P25, P75) are calculated from sample paths for enhanced visualization.

**Crypto Asset Handling** (if applicable):

- Return capping: Normal crypto returns capped at 50% (daily) or 20% (monthly) to prevent unrealistic compounding
- Extreme events: 1.5% probability per period of extreme event (65% chance crash: -80% to -70%, 35% chance rally: +150% to +200%)
- Volatility dampening: When projecting beyond available data, volatility reduced up to 50% based on years beyond data

**Key Decisions**:

1. **Return capping**: Prevents extreme outliers from dominating simulations. A single +500% return would unreasonably skew results; capping normal volatility while allowing extreme events captures realistic behavior.

2. **Extreme events**: Models the "fat tail" behavior of crypto where extreme crashes and rallies are more common than normal distributions predict. The 1.5% probability matches historical crypto crash frequencies.

3. **Volatility dampening**: As projections extend further into the future beyond available data, uncertainty increases. Dampening volatility reflects this increased uncertainty and prevents overconfidence in long-term projections.

---

## 5. Statistical Analysis

### 5.1 Success Rate

**Definition**: Portfolio balance positive throughout retirement phase (not just terminal balance). A path is successful if balance remains positive during all retirement periods and terminal balance is positive.

**Calculation**: Success rate equals number of successful paths divided by total paths: `success_rate = successful_paths / total_paths`

**Key Decision**: The success rate metric answers "What percentage of scenarios allow retirement without running out of money?" This is the primary risk metric for retirement planning, more meaningful than average terminal wealth.

### 5.2 Percentiles

**Core Percentiles**: Calculated across all simulation paths at each time period:

- P10 (10th percentile): 10% of paths fall below this value
- P50 (median): 50% of paths fall below this value
- P90 (90th percentile): 90% of paths fall below this value

**Additional Percentiles**: For enhanced visualization, P25 and P75 are derived from sample paths (up to 100 paths) if available, otherwise interpolated from core percentiles.

**Key Decision**: Using percentiles (not means) for visualization provides more robust summaries of distributions. Means can be skewed by outliers; percentiles show the distribution center and spread without sensitivity to extreme values.

### 5.3 Terminal Wealth Distribution

Terminal balances analyzed with mean, median, percentiles, and histogram visualization.

**Key Decision**: Terminal wealth distribution provides complementary information to success rate. A portfolio might have high success rate but very low terminal wealth in successful paths, or vice versa. Both metrics together provide comprehensive risk assessment.

---

## 6. Visualization

**Tools**: Plotly for interactive visualizations

**Key Visualizations**:

1. **Portfolio Charts**:

   - P10-P90 confidence band (80% coverage) with P25-P75 inner band (50% coverage) and median path
   - Age/year-based x-axis with adaptive tick intervals
   - Individual percentile labels (P10, P25, P50/Median, P75, P90) for clarity

2. **Allocation Explorer**: Interactive slider showing year-by-year allocation accounting for rebalancing, drift, contributions/withdrawals

3. **Savings Breakdown**: Dual-axis chart showing cumulative portfolio balance, contributions, returns (primary), and annual contributions/returns (secondary)

**Key Decisions**:

1. **Percentile bands**: Visualizing P10-P90 (80% coverage) and P25-P75 (50% coverage) provides intuitive understanding of outcome distributions. Users can see "most likely" outcomes (50% range) and "worst/best cases" (80% range).

2. **Percentile naming**: Using "Px" format (P10, P25, etc.) rather than percentage ranges avoids confusion. The legend clearly shows which line represents which percentile.

3. **Adaptive ticks**: X-axis tick intervals adjust based on time horizon (2 years for short, 5 for medium, 10 for long) to ensure readability without clutter.

---

## 7. Validation and Error Handling

**Input Validation**: Age constraints, non-negative financial values, weights sum to 1.0, category percentages sum to 100%.

**Error Handling**: Custom exceptions (DataError, SimulationError, ValidationError) with graceful degradation:

- Missing category CPI → overall CPI
- Missing wage data → fixed contributions
- Insufficient historical data → Monte Carlo extension

**Key Decision**: Graceful degradation ensures the application continues functioning even with incomplete data. Rather than failing entirely, the system uses fallback values (e.g., overall CPI instead of category CPI) to maintain functionality while alerting users to data limitations.

---

## 8. Performance Optimizations

**Caching**:

- LRU cache (maxsize=64) for market data downloads to avoid redundant API calls
- In-memory caching for CPI/wage data after first load
- Session state caching for simulation results to avoid re-computation on UI interactions

**Efficiency**:

- Vectorized NumPy operations for portfolio calculations
- Pre-calculated annual wages for all accumulation years (cached before simulation loop)
- Limited sample paths (100 max) for visualization to reduce memory usage

**Key Decisions**:

1. **Pre-calculation**: Wage projections are calculated once before the simulation loop rather than inside the loop. For 1000 paths × 30 years, this saves 30,000 redundant calculations.

2. **Sample path limitation**: Storing all 1000 paths for visualization would require significant memory. Storing 100 randomly sampled paths provides representative visualization while keeping memory usage manageable.

3. **LRU cache size**: 64 entries balances memory usage with cache hit rates. Most users simulate similar portfolios (stocks/bonds), so 64 entries covers common combinations while preventing unbounded memory growth.

---

## 9. Technical Architecture

**Modular Design**: Separation of concerns across services:

- **DataService**: Data acquisition and processing (market, CPI, wage data)
- **PortfolioService**: Portfolio mathematics and state management (rebalancing, withdrawals)
- **SimulationService**: Hybrid simulation engine (historical accumulation, Monte Carlo retirement)
- **Components**: UI components (sidebar, charts, results)

**Type Safety**: Dataclasses for all data structures (SimulationParams, SimulationResult, PortfolioState, WithdrawalParams, ExpenseCategory). Type hints throughout enable IDE support and catch errors early.

**Configuration**: Centralized configuration with defaults, ticker mappings, and simulation parameters. Single source of truth for constants ensures consistency.

**Key Decisions**:

1. **Service separation**: Clear boundaries between data, portfolio logic, and simulation logic enable independent testing and modification. Changes to portfolio rebalancing don't affect data fetching, for example.

2. **Dataclasses**: Type-safe data structures prevent runtime errors from incorrect data shapes. IDE autocomplete and type checking catch many errors before execution.

3. **Centralized config**: All configuration parameters (defaults, crypto handling, cache sizes) in one place makes tuning and maintenance easier.

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
- **Simulation**: Hybrid approach combines historical data and Monte Carlo projections; Monte Carlo assumes normality and may miss extreme tail events; historical portion limited by available data

**Key Discussion**:

1. **Normal distribution assumption**: Monte Carlo assumes log-returns are normally distributed. Real markets exhibit fat tails (extreme events more common than normal distribution predicts). Crypto assets partially address this with extreme event injection, but standard assets don't. This may underestimate tail risk.

2. **No taxes**: Tax-advantaged accounts (401k, IRA) are common in retirement planning, but taxable accounts have different dynamics. Future enhancement could model tax impacts.

3. **Perfect rebalancing**: Real rebalancing has transaction costs and may be done with thresholds rather than exactly annually. This simplification overstates portfolio performance slightly, but is acceptable for planning purposes.

4. **Historical data limitations**: For assets with limited history (e.g., new ETFs), the system must rely more on Monte Carlo, reducing realism. Historical ticker mapping helps but doesn't eliminate this issue entirely.

---

## 11. Future Enhancements

**Potential**: Tax modeling, transaction costs, Social Security, healthcare costs, longevity risk, regime detection, stress testing.

**Data**: Additional asset classes, international data, alternative data, real-time updates.

**Key Opportunities**:

1. **Tax modeling**: Different withdrawal strategies (taxable vs. tax-advantaged accounts) significantly impact outcomes. Modeling tax brackets, capital gains, and required minimum distributions would add substantial value.

2. **Regime detection**: Markets operate in different regimes (bull, bear, high volatility, low volatility). Identifying and modeling regime transitions could improve projections.

3. **Healthcare costs**: Medical expenses often increase faster than inflation in retirement. Explicit healthcare cost modeling with age-dependent multipliers would improve accuracy.

---

## 12. Conclusion

The application employs a hybrid simulation approach that combines historical backtesting for the accumulation phase with Monte Carlo projections for the retirement phase. This unified method leverages the realism of historical market data where available while using statistical projections for future periods. Key strengths include robust data handling, sophisticated portfolio management, category-specific inflation adjustments, wage-based contributions, and comprehensive visualization. The modular architecture supports future enhancements.

**Key Design Philosophy**: The system prioritizes realism and accuracy where possible (historical data, category-specific inflation) while using simplifications (annual rebalancing, average inflation rates) that maintain tractability without sacrificing core insights. The hybrid approach balances these goals by using the most appropriate method for each phase of the retirement planning horizon.

---

## References

- Yahoo Finance API (via yfinance library)
- Bureau of Labor Statistics (BLS) CPI and wage data
- Modern Portfolio Theory
- Monte Carlo simulation techniques

---

_Report Generated: 2024_
_Application Version: 1.0_
