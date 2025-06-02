from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env so that OPENAI_API_KEY is set
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

KAGGLE_OVERVIEW = """
This context comes from the Kaggle competition page “JPX Tokyo Stock Exchange Prediction”:
https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/overview

Summary:
- The competition is hosted by JPX (Japan Exchange Group) and AlpacaJapan.
- Participants must build a model that ranks about 2000 stocks each trading day by expected return.
- Evaluation metric: daily spread return Sharpe ratio, defined as:
    • Each day, buy top‐ranked 200 stocks (by predicted return) and short bottom‐ranked 200.
    • Weight the “buy”‐leg linearly from weight 2 for the very top down to weight 1 for rank 200.
    • Similarly for the “short”‐leg.
    • Compute daily spread return = (weighted average actual return of buy‐leg) – (weighted average actual return of short‐leg).
    • Over the validation period, Sharpe = mean(daily spread return) / std(daily spread return).
- Stocks eligible: top 2000 by market cap, must be listed >1 year as of 2021‐12‐31.
- Exclude stocks under supervision or to be delisted in private period.
- Data provided: historical prices, stock info, options, etc.
"""

KAGGLE_METRIC_DEF = """
This context comes from the Kaggle code notebook “JPX Competition Metric Definition”:
https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition

Summary:
- Shows Python implementation of calc_spread_return_sharpe:
    • For each day, ensures Rank ∈ [0, N−1].
    • Weights = linspace(start=2, stop=1, num=200).
    • purchase = sum_{i=0 to 199} (Target of rank i ⋅ weight[i]) / mean(weights).
    • short = sum_{i=0 to 199} (Target of bottom ranks ⋅ weight[i]) / mean(weights).
    • daily spread return = purchase − short.
    • Then Sharpe = mean(daily_returns) / std(daily_returns).
- Provides complete notebook with examples, sample_submission format, etc.
"""

WIKIPEDIA_SHARPE = """
This context comes from Wikipedia’s “Sharpe ratio”:
https://en.wikipedia.org/wiki/Sharpe_ratio

Summary:
- Sharpe ratio measures risk‐adjusted return.
- Formula: S = (E[R_p] − R_f) / σ_p, where:
    • E[R_p]: expected portfolio return
    • R_f: risk‐free rate (often zeroed in competitions)
    • σ_p: standard deviation of portfolio returns
- In daily spread context, R_f is often treated as zero. So Sharpe ≈ mean(daily_RET) / std(daily_RET).
- A higher Sharpe implies better risk‐adjusted performance; values >1 are considered good, >2 excellent.
"""

def ask_llm(prompt: str) -> str:
    system_message = {
        "role": "system",
        "content": (
            "You are a financial ML assistant familiar with the JPX competition. "
            "Please use the following contextual summaries in your answers:\n\n"
            f"{KAGGLE_OVERVIEW}\n\n"
            f"{KAGGLE_METRIC_DEF}\n\n"
            f"{WIKIPEDIA_SHARPE}\n\n"
            "When responding, leverage this context to give precise, helpful answers."
        )
    }

    user_message = {
        "role": "user",
        "content": prompt
    }

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[system_message, user_message],
        temperature=0.5,
        max_tokens=1500,
    )
    return response.choices[0].message.content
