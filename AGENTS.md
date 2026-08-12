# AGENTS.md - FINS5545 Project B

## Project

This project builds NexaVest, a systematic multi-asset investment platform for FINS5545 Project B.

The project includes:
- Equity, Crypto and Combined funds
- Equal Weight, Minimum Variance, Maximum Sharpe and Risk Parity methods
- Walk-forward out-of-sample backtesting
- Finance-aware VADER sentiment analysis
- Sector sentiment indices
- Sentiment-enhanced portfolio extensions
- A Streamlit investor dashboard

The main assignment requirements are in PROJECT_BRIEF.md and the provided context files.

## Coding Rules

- Work only inside the z5537596_projectB folder.
- Keep the analysis reproducible from scripts.
- Do not commit raw source data.
- Save derived app data under results/data/.
- Save report tables under results/tables/.
- Save report figures under results/figures/.
- The deployed Streamlit app must read pre-computed results and must not rerun portfolio optimisation or VADER.

## Backtest Rules

- Use walk-forward out-of-sample testing.
- Portfolio weights must use only information available before each rebalance date.
- Never use future observations when estimating portfolio weights.
- Equity and Combined funds use the equity trading calendar and 252-day annualisation.
- Crypto funds use the crypto calendar and 365-day annualisation.
- Apply the specified rebalance frequency consistently.
- Combined funds must remain long-only and fully invested.
- Total cryptocurrency exposure in Combined funds must not exceed 25%.
- Include turnover-based transaction costs in reported net performance.
- Check that training_end is earlier than rebalance_date.

## Sentiment Rules

- Treat the provided news data as headlines, not full articles.
- Do not claim headline sentiment measures investors' true sentiment.
- Build ticker-day sentiment before aggregating to sector sentiment.
- Equal-weight ticker sentiment within each sector.
- Treat no-news ticker-days according to the documented neutral policy and retain coverage information.
- Lag sentiment by at least one trading day before it can affect portfolio decisions.
- Do not describe lower VADER neutrality as evidence of return predictability.
- Keep sentiment effects on the equity component only.

## Validation and Interpretation

- Check solver success and portfolio weight constraints.
- Verify weights sum approximately to one.
- Check for negative weights where long-only constraints apply.
- Check the Combined crypto cap after every rebalance.
- Compare gross and net performance when transaction costs matter.
- Interpret annualised return, volatility, Sharpe ratio and maximum drawdown together rather than selecting a fund using return alone.
- Do not tune parameters after seeing final out-of-sample results merely to improve reported performance.
- Report negative sentiment-fusion results honestly.

## AI Workflow

AI may assist with:
- coding
- debugging
- explaining errors
- suggesting validation checks
- improving chart design
- helping organise report structure

AI output must be reviewed before use.

For important AI-assisted changes:
1. record the prompt,
2. record what the AI suggested,
3. check the suggestion against PROJECT_BRIEF.md and project outputs,
4. document any correction or rejection,
5. keep the final economic interpretation as my own work.

Prompt logs and AI-use notes are stored in ai/.
