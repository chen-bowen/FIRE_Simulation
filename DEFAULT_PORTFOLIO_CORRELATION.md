# Correlation in Default "Moderate (60/40)" Portfolio

## Portfolio Composition

**Default Preset**: "Moderate (60/40)"

- **US Stocks**: 60% → Ticker: `^GSPC` (S&P 500 Index)
- **Bonds**: 40% → Ticker: `BND` (Vanguard Total Bond Market ETF, maps to `VBMFX` for historical data)

## How Correlation Works

### Step 1: Historical Data Collection

The system downloads historical price data for both assets:

```python
# Example: 10 years of monthly returns (120 months)
returns_df = pd.DataFrame({
    '^GSPC': [...],  # S&P 500 monthly returns
    'BND':   [...]   # Bond ETF monthly returns (maps to VBMFX for older data)
})
# Shape: (120, 2) - 120 months, 2 assets
```

**Note**: `BND` is a bond ETF that tracks bond prices (not yields). The system automatically maps `BND` to `VBMFX` (Vanguard Total Bond Market Index Fund) for historical data extending back to the 1980s. This provides proper bond price data rather than yield indices.

### Step 2: Calculate Correlation from Historical Returns

The system calculates the covariance matrix from log-returns:

```python
# Convert to log-returns
log_returns = np.log1p(returns_df.to_numpy())  # log(1 + r)

# Calculate covariance matrix
cov = np.cov(log_returns.T)  # Shape: (2, 2)
```

### Step 3: Example Covariance Matrix

For a typical stocks/bonds portfolio over 10 years of monthly data:

```python
cov = [[0.0025, -0.0002],   # [Stocks variance, Stocks-Bonds covariance]
       [-0.0002, 0.0004]]   # [Bonds-Stocks covariance, Bonds variance]
```

**Interpretation:**

- `cov[0,0] = 0.0025`: Stocks variance = 0.0025 → monthly volatility ≈ 5%
- `cov[1,1] = 0.0004`: Bonds variance = 0.0004 → monthly volatility ≈ 2%
- `cov[0,1] = -0.0002`: **Negative covariance** (stocks and bonds tend to move in opposite directions)

### Step 4: Calculate Correlation Coefficient

From the covariance matrix, we can calculate the correlation:

```python
correlation = cov[0,1] / (sqrt(cov[0,0]) * sqrt(cov[1,1]))
            = -0.0002 / (sqrt(0.0025) * sqrt(0.0004))
            = -0.0002 / (0.05 * 0.02)
            = -0.0002 / 0.001
            = -0.2  # Negative correlation ≈ -0.2 to -0.3 (typical for stocks/bonds)
```

**What this means:**

- **Correlation ≈ -0.2 to -0.3**: Stocks and bonds have a slight negative correlation
- When stocks go up, bonds tend to go slightly down (or vice versa)
- This provides **diversification benefit** - bonds help cushion stock market downturns

### Step 5: Real-World Example

**Typical monthly returns (simplified):**

| Month      | Stocks (^GSPC) | Bonds (^TNX) | Portfolio (60/40)                    |
| ---------- | -------------- | ------------ | ------------------------------------ |
| Good month | +3%            | -0.5%        | 60% × 3% + 40% × (-0.5%) = **+1.6%** |
| Bad month  | -5%            | +1.5%        | 60% × (-5%) + 40% × 1.5% = **-2.1%** |
| Neutral    | +1%            | +0.3%        | 60% × 1% + 40% × 0.3% = **+0.72%**   |

**Key insight**: In bad stock months (market crashes), bonds typically gain, reducing portfolio losses. This is the **diversification benefit** captured by negative correlation.

### Step 6: How Correlation is Preserved in Monte Carlo

When generating Monte Carlo returns for retirement, the system uses Cholesky decomposition to preserve this correlation structure:

```python
# Cholesky decomposition of covariance matrix
L = np.linalg.cholesky(cov)
# L ≈ [[0.05,  0.0 ],
#      [-0.004, 0.02]]

# Generate correlated returns
z = rng.standard_normal(size=(360, 2))  # Independent random numbers
mc_log_returns = z @ L.T + log_means
```

**What happens:**

- `L[0,0] = 0.05`: Controls stocks volatility
- `L[1,1] = 0.02`: Controls bonds volatility
- `L[1,0] = -0.004`: **Negative value** creates negative correlation

When `z[0]` (stock random number) is large and positive, stocks go up. But the `-0.004 * z[0]` term makes bonds go slightly down, preserving the negative correlation.

### Step 7: Concrete Example of Correlated Generation

```python
# Generate one month's returns
z = [1.5, 0.5]  # Stock: high positive, Bond: slightly positive (independent)

# Apply Cholesky transformation
mc_log_return_stock = z[0] * L[0,0] + z[1] * L[0,1] + log_mean_stock
                    = 1.5 * 0.05 + 0.5 * 0.0 + 0.008
                    = 0.075 + 0.008 = 0.083

mc_log_return_bond  = z[0] * L[1,0] + z[1] * L[1,1] + log_mean_bond
                    = 1.5 * (-0.004) + 0.5 * 0.02 + 0.003
                    = -0.006 + 0.01 + 0.003 = 0.007

# Convert to arithmetic returns
stock_return = exp(0.083) - 1 ≈ 8.65%
bond_return  = exp(0.007) - 1 ≈ 0.70%
```

**Result**: Stocks had a big positive month (+8.65%), bonds had a small positive month (+0.70%). The negative correlation is preserved - bonds didn't go up as much as they would independently.

**Another example (stock crash):**

```python
z = [-2.0, 0.8]  # Stock: big negative, Bond: slightly positive

mc_log_return_stock = -2.0 * 0.05 + 0.0 + 0.008 = -0.092
stock_return = exp(-0.092) - 1 ≈ -8.8%

mc_log_return_bond  = -2.0 * (-0.004) + 0.8 * 0.02 + 0.003
                    = 0.008 + 0.016 + 0.003 = 0.027
bond_return = exp(0.027) - 1 ≈ 2.7%
```

**Result**: Stocks crashed (-8.8%), but bonds gained (+2.7%). This is the diversification benefit in action!

## Why This Matters

**Without correlation preservation:**

- Stock returns and bond returns would be independent
- A -20% stock month might come with a -1% bond month (both losing money)
- Portfolio would lose: 60% × (-20%) + 40% × (-1%) = -12.4%

**With correlation preservation:**

- Stock returns and bond returns have negative correlation
- A -20% stock month typically comes with a +2% bond month (bonds gain)
- Portfolio loses: 60% × (-20%) + 40% × 2% = -11.2%

**Diversification benefit**: The portfolio loses less because bonds gain when stocks fall.

## Code Implementation

The actual calculation happens in `app/services/simulation_service.py`:

```python
# Lines 252-266: Calculate covariance and Cholesky factor
log_returns = np.log1p(asset_returns)
log_means = np.mean(log_returns, axis=0)
cov = np.cov(log_returns.T)  # 2x2 covariance matrix
L = self._safe_cholesky(cov)  # Cholesky decomposition

# Lines 387-391: Generate correlated returns
z = rng.standard_normal(size=(remaining_periods, n_assets))
mc_log_returns = z @ L.T + log_means  # Correlated log-returns
mc_arith_returns = np.expm1(mc_log_returns)  # Convert to arithmetic
```

## Summary

1. **Historical data**: System downloads 10+ years of monthly returns for ^GSPC (stocks) and ^TNX (bonds)

2. **Covariance calculation**: Calculates how stocks and bonds moved together historically

   - Typically negative covariance (stocks up → bonds down, and vice versa)

3. **Correlation coefficient**: Usually around -0.2 to -0.3 (slight negative correlation)

4. **Monte Carlo preservation**: Cholesky decomposition ensures future Monte Carlo returns maintain this correlation structure

5. **Diversification benefit**: When stocks crash in simulations, bonds tend to gain, reducing portfolio losses

This is why the 60/40 portfolio is considered "balanced" - the negative correlation between stocks and bonds provides natural diversification and risk reduction.
