import argparse
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRADING_DAYS = 252


@dataclass
class BacktestResult:
    data: pd.DataFrame
    metrics: Dict[str, float]
    model_name: str


def load_data(csv_path: str | None = None, ticker: str | None = None, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Load OHLCV data either from a CSV file or via yfinance.

    CSV should contain at least: Date, Close, Volume.
    Preferred columns: Date, Open, High, Low, Close, Adj Close, Volume.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if "Date" not in df.columns:
            raise ValueError("CSV must contain a 'Date' column.")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    elif ticker:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is not installed. Install it with `pip install yfinance` or pass --csv_path."
            ) from exc

        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError(f"No data downloaded for ticker {ticker}.")
        df.index = pd.to_datetime(df.index)
    else:
        raise ValueError("Provide either csv_path or ticker.")

    required = {"Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df[[c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]]
    return df.dropna()



def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)
    out["momentum_3"] = out["Close"] / out["Close"].shift(3) - 1
    out["momentum_10"] = out["Close"] / out["Close"].shift(10) - 1
    out["momentum_20"] = out["Close"] / out["Close"].shift(20) - 1
    out["volatility_5"] = out["ret_1"].rolling(5).std()
    out["volatility_20"] = out["ret_1"].rolling(20).std()
    out["sma_5"] = out["Close"].rolling(5).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_ratio_5_20"] = out["sma_5"] / out["sma_20"] - 1
    out["price_vs_sma20"] = out["Close"] / out["sma_20"] - 1
    out["volume_change_1"] = out["Volume"].pct_change(1)
    out["volume_z_20"] = (out["Volume"] - out["Volume"].rolling(20).mean()) / out["Volume"].rolling(20).std()
    out["rsi_14"] = compute_rsi(out["Close"], 14)
    if {"High", "Low"}.issubset(out.columns):
        out["intraday_range"] = (out["High"] - out["Low"]) / out["Close"]
    else:
        out["intraday_range"] = np.nan

    # Target: tomorrow's close higher than today's close.
    out["fwd_return_1"] = out["Close"].shift(-1) / out["Close"] - 1
    out["target"] = (out["fwd_return_1"] > 0).astype(int)

    return out.dropna()



def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"target", "fwd_return_1"}
    return [c for c in df.columns if c not in exclude and df[c].dtype != "O"]



def build_model(model_name: str, random_state: int = 42):
    if model_name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ])
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError("model_name must be 'logistic' or 'random_forest'.")



def performance_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series, signals: pd.Series, y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> Dict[str, float]:
    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.loc[strategy_returns.index].dropna()

    equity = (1 + strategy_returns).cumprod()
    bench_equity = (1 + benchmark_returns).cumprod()

    ann_return = equity.iloc[-1] ** (TRADING_DAYS / len(equity)) - 1 if len(equity) > 0 else np.nan
    ann_vol = strategy_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(TRADING_DAYS) if strategy_returns.std() > 0 else np.nan
    max_dd = ((equity / equity.cummax()) - 1).min()

    bench_ann_return = bench_equity.iloc[-1] ** (TRADING_DAYS / len(bench_equity)) - 1 if len(bench_equity) > 0 else np.nan
    bench_vol = benchmark_returns.std() * np.sqrt(TRADING_DAYS)
    bench_sharpe = (benchmark_returns.mean() / benchmark_returns.std()) * np.sqrt(TRADING_DAYS) if benchmark_returns.std() > 0 else np.nan
    bench_max_dd = ((bench_equity / bench_equity.cummax()) - 1).min()

    trades = signals.diff().fillna(0).abs()
    turnover = trades.sum()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "strategy_total_return": equity.iloc[-1] - 1,
        "strategy_annual_return": ann_return,
        "strategy_annual_vol": ann_vol,
        "strategy_sharpe": sharpe,
        "strategy_max_drawdown": max_dd,
        "benchmark_total_return": bench_equity.iloc[-1] - 1,
        "benchmark_annual_return": bench_ann_return,
        "benchmark_annual_vol": bench_vol,
        "benchmark_sharpe": bench_sharpe,
        "benchmark_max_drawdown": bench_max_dd,
        "turnover": turnover,
    }



def backtest(
    df_feat: pd.DataFrame,
    model_name: str = "logistic",
    train_size: float = 0.7,
    prob_long: float = 0.55,
    prob_short: float = 0.45,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    feature_cols = get_feature_columns(df_feat)
    split_idx = int(len(df_feat) * train_size)
    train = df_feat.iloc[:split_idx].copy()
    test = df_feat.iloc[split_idx:].copy()

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    model = build_model(model_name)
    model.fit(X_train, y_train)

    prob_up = model.predict_proba(X_test)[:, 1]
    pred = (prob_up >= 0.5).astype(int)

    # Trading rule: long / flat / short based on probability thresholds.
    signal = np.where(prob_up >= prob_long, 1, np.where(prob_up <= prob_short, -1, 0))

    result = test.copy()
    result["prob_up"] = prob_up
    result["prediction"] = pred
    result["signal"] = signal

    # Position decided at close t and applied over next day's return already stored as fwd_return_1.
    result["gross_strategy_return"] = result["signal"] * result["fwd_return_1"]

    # Transaction cost whenever position changes.
    tc = (transaction_cost_bps / 10000.0) * pd.Series(signal, index=result.index).diff().abs().fillna(np.abs(signal[0]))
    result["transaction_cost"] = tc
    result["strategy_return"] = result["gross_strategy_return"] - result["transaction_cost"]
    result["benchmark_return"] = result["fwd_return_1"]
    result["strategy_equity"] = (1 + result["strategy_return"]).cumprod()
    result["benchmark_equity"] = (1 + result["benchmark_return"]).cumprod()

    metrics = performance_metrics(
        strategy_returns=result["strategy_return"],
        benchmark_returns=result["benchmark_return"],
        signals=result["signal"],
        y_true=y_test,
        y_pred=pred,
        y_prob=prob_up,
    )

    print("\nClassification report:\n")
    print(classification_report(y_test, pred, digits=4))

    return BacktestResult(data=result, metrics=metrics, model_name=model_name)



def plot_results(result: BacktestResult, output_prefix: str = "results") -> None:
    df = result.data

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["strategy_equity"], label=f"{result.model_name} strategy")
    plt.plot(df.index, df["benchmark_equity"], label="Buy & hold")
    plt.title("Strategy vs Buy-and-Hold Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_equity_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["prob_up"], label="Predicted probability of up move")
    plt.axhline(0.55, linestyle="--", linewidth=1, label="Long threshold")
    plt.axhline(0.45, linestyle="--", linewidth=1, label="Short threshold")
    plt.title("Model Output Probabilities")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_probabilities.png", dpi=150)
    plt.close()



def print_metrics(metrics: Dict[str, float]) -> None:
    print("\nBacktest metrics:\n")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:28s}: {value: .4f}")
        else:
            print(f"{key:28s}: {value}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Machine Learning Trading Signal Project")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV containing OHLCV data")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker to download via yfinance")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date for yfinance")
    parser.add_argument("--end", type=str, default=None, help="End date for yfinance")
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "random_forest"])
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--prob_long", type=float, default=0.55)
    parser.add_argument("--prob_short", type=float, default=0.45)
    parser.add_argument("--transaction_cost_bps", type=float, default=5.0)
    parser.add_argument("--output_prefix", type=str, default="results")
    args = parser.parse_args()

    raw = load_data(csv_path=args.csv_path, ticker=args.ticker, start=args.start, end=args.end)
    feat = engineer_features(raw)
    result = backtest(
        feat,
        model_name=args.model,
        train_size=args.train_size,
        prob_long=args.prob_long,
        prob_short=args.prob_short,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    print_metrics(result.metrics)

    result.data.to_csv(f"{args.output_prefix}_backtest.csv")
    plot_results(result, output_prefix=args.output_prefix)
    print(f"\nSaved backtest data to {args.output_prefix}_backtest.csv")
    print(f"Saved plots to {args.output_prefix}_equity_curve.png and {args.output_prefix}_probabilities.png")


if __name__ == "__main__":
    main()
