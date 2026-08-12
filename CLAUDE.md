# NexaVest Project B Agent Instructions

## Project objective
This project develops NexaVest, a systematic multi-asset investment application using equity, cryptocurrency, and equity-news data.

## Working rules
- Preserve the Project A data-cleaning and return-construction conventions.
- Use walk-forward out-of-sample evaluation for portfolio results.
- Never use future observations when estimating portfolio weights or sentiment signals.
- Equity portfolios use a 252-observation estimation window and rebalance every 21 trading observations.
- Crypto portfolios use a 365-observation estimation window and rebalance every 30 calendar observations.
- Combined portfolios use the equity trading calendar.
- Keep portfolios long-only and fully invested.
- Preserve the 25% maximum crypto sleeve in Combined funds.
- Apply a 10 bps turnover-based transaction-cost assumption.
- Do not change model parameters after inspecting final out-of-sample performance simply to improve results.

## Sentiment design
- Use the finance-aware VADER implementation in `src/sentiment.py`.
- Use lagged historical sentiment only.
- Preserve the coverage-adjusted sector signal design.
- Sentiment tilts only the equity sleeve; crypto weights are preserved.

## Reproducibility
- `python scripts/run_part_b.py` should reproduce the Project B outputs.
- Required outputs should remain under `results/data`, `results/tables`, and `results/figures`.
- Do not manually edit generated CSV results.
- Keep code changes consistent with the existing module interfaces.
