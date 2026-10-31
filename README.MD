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

The complete notebook environment is not locked. `requirements-checks.txt` pins only the dependencies for the focused callback checks below; it does not install VectorBT, pandas, Optuna, or Yahoo Finance. Validate compatibility separately before running the full notebook.

## Experiment controls

The default data example uses AAPL at a 1-day interval. The first backtest starts with 1000 in cash, charges a 0.001 proportional fee, and sets stop loss and take profit to 0.025 and 0.05. Later examples deliberately change some of these assumptions.

The Optuna sections request up to 1000 trials and use a TPE sampler seeded with 42. Reduce the trial budget for an initial smoke run. The optimization examples evaluate parameters on the data supplied to the objective; they do not implement a separate out-of-sample evaluation.

Data downloads can change over time. Record the data snapshot, package versions, search space, and evaluation period when comparing results. Saved or newly generated returns are experiment outputs, not a validated performance benchmark.

## Stop adjustments and focused checks

The adjustment callbacks compute percentage profit in the current position's direction. Rising prices benefit long positions; falling prices benefit short positions. Existing thresholds are preserved: stop loss adjusts at 20%, 25%, and 40% profit, and take profit adjusts at 50%. Flat positions, non-finite context values, nonpositive entry prices, and non-finite calculated returns retain the current stop/trailing settings.

The margin-filter timing cell copies entry signals inside each timed iteration because `apply_margin` mutates its signal array. Every repetition therefore starts from the same entries; the timing includes the cost of that copy.

Run the focused checks with Python 3.11:

```bash
python -m pip install -r requirements-checks.txt
python -m unittest discover -s tests -v
```

The tests extract the actual callback definitions from the notebook and compile them with Numba in nopython mode. They cover long/short profit and loss, threshold boundaries, invalid/flat contexts, position-size independence, and repeatable benchmark inputs. CI uses Python 3.11 and the same pinned test dependencies. No VectorBT installation, price download, portfolio simulation, or Optuna search is performed by these checks; they do not establish profitable trading performance or full-notebook compatibility.

## License

[GNU General Public License v3.0](LICENSE).

