# VectorBT Backtester

A Jupyter research notebook for exploring vectorized trading signals, historical backtests, and parameter searches with VectorBT and Optuna.

**[Open the notebook](VectorBT%20Backtest.ipynb)**

## Notebook guide

| Section | What it demonstrates |
| --- | --- |
| Getting price data | Download AAPL daily prices through `vbt.YFData` |
| Creating signals | Moving-average crossover signals filtered by RSI |
| Simple backtest | Portfolio statistics, plots, and readable position records |
| Window optimization | Compare moving-average window combinations |
| Trailing stop loss and take profit | Custom Numba-compiled adjustment callbacks |
| FPE box | Delay candidate entries until a price margin is reached |
| Grid search | Compare entry-margin parameters |
| Bayesian search | Optimize stop parameters with Optuna's TPE sampler |
| Optuna early stopping | Stop a study after a configured interval without improvement |

## Run the notebook

1. Clone this repository and open it as your working directory.
2. Use a Jupyter environment with NumPy, pandas, Numba, VectorBT, Optuna, and the Yahoo Finance integration required by `vbt.YFData`.
3. Open `VectorBT Backtest.ipynb` and run the import, data, and signal cells before the backtest sections.
4. Review each section's cash, position size, fees, stop settings, and optimization budget before executing it.

The repository does not contain a pinned dependency manifest or lockfile. Its original API calls are preserved, so package compatibility needs validation in the environment you choose.

## Experiment controls

The default data example uses AAPL at a 1-day interval. The first backtest starts with 1000 in cash, charges a 0.001 proportional fee, and sets stop loss and take profit to 0.025 and 0.05. Later examples deliberately change some of these assumptions.

The Optuna sections request up to 1000 trials and use a TPE sampler seeded with 42. Reduce the trial budget for an initial smoke run. The optimization examples evaluate parameters on the data supplied to the objective; they do not implement a separate out-of-sample evaluation.

Data downloads can change over time. Record the data snapshot, package versions, search space, and evaluation period when comparing results. Saved or newly generated returns are experiment outputs, not a validated performance benchmark.

## License

[GNU General Public License v3.0](LICENSE).
