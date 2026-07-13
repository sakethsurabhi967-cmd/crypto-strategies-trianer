# Crypto Strategies Trainer — EMA 9/21 AI Trading Agent

Train a GPU-accelerated AI model that learns when EMA 9/21 crossover signals
are worth taking on crypto **spot and futures** markets, find the **best
timescales** per coin via backtesting, and run a **predictor** that turns the
trained models into live LONG / SHORT / FLAT signals.

> ⚠️ **Disclaimer:** educational tooling, not financial advice. Crypto (and
> especially leveraged futures) trading can lose your entire stake. Backtest
> performance does not guarantee future results. Paper trade first.

## How it works

```
 ccxt (binance / bybit / okx ...)          Colab GPU                 your machine
┌───────────────────────────┐   ┌────────────────────────────┐   ┌──────────────────┐
│ src/data.py               │   │ src/train.py               │   │ src/predict.py   │
│ OHLCV spot + futures      ├──▶│ GRU model per              ├──▶│ loads models/*.pt│
│ 15m 30m 1h 4h history     │   │ (symbol, timeframe)        │   │ live JSON signals│
└───────────────────────────┘   │ saves models/*.pt bundles  │   │ LONG/SHORT/FLAT  │
                                └────────────┬───────────────┘   └──────────────────┘
                                             ▼
                                src/backtest.py — ranks timeframes
                                (raw EMA cross vs model-filtered)
```

- **Strategy core:** EMA 9 / EMA 21. Golden cross → long bias, death cross →
  short bias (futures) or exit (spot).
- **The AI part:** a GRU sequence model (PyTorch) reads 64 candles of
  EMA/momentum/volatility features and predicts whether price moves
  **up / flat / down** over the next bars. Crossover trades are only taken
  when the model agrees with enough confidence — it filters out the whipsaw
  crossovers that kill naive EMA strategies.
- **Best time scales:** the backtester runs every timeframe in `config.yaml`
  (default `15m 30m 1h 4h`) and ranks them by Sharpe ratio per symbol, with
  fees and slippage included (`models/backtest_ranking.json`).
- **Model bundles:** each `models/SYMBOL_TF_market.pt` file contains weights +
  normalisation stats + hyperparameters, so the predictor needs nothing else.

## 1. Train in Google Colab (GPU)

1. Open `notebooks/train_colab.ipynb` in Colab
   ([open directly](https://colab.research.google.com/github/sakethsurabhi967-cmd/crypto-strategies-trianer/blob/claude/crypto-trading-ema-agent-h4hi0j/notebooks/train_colab.ipynb)).
2. `Runtime → Change runtime type → GPU`.
3. Run all cells. It clones this repo, downloads history, trains every
   (symbol, timeframe) pair on the GPU, backtests + ranks timescales, and
   gives you `trained_models.zip` (download or save to Google Drive).

## 2. Run the predictor with the trained models

```bash
git clone https://github.com/sakethsurabhi967-cmd/crypto-strategies-trianer.git
cd crypto-strategies-trianer
pip install -r requirements.txt
unzip ~/Downloads/trained_models.zip   # -> models/*.pt

python -m src.predict                  # one signal pass over all bundles
python -m src.predict --loop 60        # refresh every 60 seconds
python -m src.predict --symbols BTC/USDT --timeframes 1h --json signals.json
```

Example output (one JSON object per model):

```json
{"symbol": "BTC/USDT", "timeframe": "1h", "market": "spot",
 "close": 64231.5, "ema9": 64180.2, "ema21": 63950.7,
 "ema_state": "bullish", "prob_up": 0.63, "prob_down": 0.14,
 "action": "LONG", "suggested_stop": 62996.1, "atr_pct": 0.96}
```

The predictor **does not place orders** — pipe the JSON into your own
execution or alerting layer and keep a human in the loop.

## Local / CLI usage (no Colab)

```bash
pip install -r requirements.txt
python -m src.train                                   # everything in config.yaml
python -m src.train --market futures                  # futures models (enables SHORT)
python -m src.train --symbols BTC/USDT --timeframes 1h 4h
python -m src.backtest                                # rank timescales, raw EMA cross
python -m src.backtest --use-model                    # model-filtered comparison
```

Training auto-detects CUDA; any NVIDIA GPU (or Colab T4) works.

## Configuration

Everything lives in `config.yaml`: symbols, timeframes, EMA periods, label
horizon, model size, epochs, fees/slippage and the confidence threshold the
predictor requires before acting. Futures data uses Binance USDT-M perpetuals
(`binanceusdm`) with automatic fallback to Bybit/OKX if an exchange is
unreachable (helpful on Colab IPs).

## Repo layout

| Path | Purpose |
|---|---|
| `src/data.py` | OHLCV download (spot + futures) with CSV caching & exchange fallback |
| `src/features.py` | EMA 9/21, RSI, ATR, crossover features + 3-class labels |
| `src/model.py` | GRU classifier + self-contained model bundle save/load |
| `src/train.py` | GPU training loop, chronological splits, early stopping |
| `src/backtest.py` | Fee-aware backtests, timeframe ranking, model-filtered mode |
| `src/predict.py` | The predictor: trained models → live JSON signals |
| `notebooks/train_colab.ipynb` | End-to-end Colab GPU workflow |
| `config.yaml` | All knobs in one place |
