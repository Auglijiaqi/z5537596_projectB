# AI Notes - FINS5545 Project B

## How I Used AI

I used AI throughout Project B mainly as a coding, debugging and review assistant. I used it to help structure the walk-forward backtest, investigate technical errors, design validation checks, improve the sentiment workflow, organise the Streamlit investor journey, and review the project against the assignment brief.

I did not treat AI output as automatically correct. For important parts of the project, I checked suggestions against the project brief, generated outputs and my own implementation.

## How I Checked AI

The most important validation involved the portfolio backtest. I checked the rebalance audit to confirm that every training period ended before the corresponding rebalance date. I also verified the different equity and crypto calendars, portfolio weight constraints, the 25% crypto cap for Combined funds, transaction costs and solver success.

For sentiment analysis, I compared plain VADER with my finance-aware extension. Although the finance lexicon reduced the neutral classification rate, I did not interpret this as proof that sentiment could predict returns. I therefore evaluated the signal separately through out-of-sample sentiment-portfolio fusion.

The fusion results were slightly worse than the base Minimum Variance portfolios. I chose not to retune the sentiment parameter after seeing these results. Instead, I retained the negative result and discussed possible reasons such as headline noise, uneven news coverage, turnover and transaction costs.

## When I Did Not Rely on AI

I did not rely on AI alone for final economic interpretation or submission decisions. I reviewed the report structure and figures myself and compared them with the project brief and teacher guidance.

I also manually tested the Streamlit investor journey and ran the provided `check_handin.py` script. The first final-stage check identified temporary files such as `.DS_Store`, `__pycache__` and `.pyc`. I removed them and reran the checker until all 23 checks passed.

## Reflection

AI made development faster, especially for coding, debugging and identifying possible validation checks. However, I found that its suggestions were most useful when treated as starting points rather than final answers.

The parts that required the most judgement were checking for look-ahead bias, handling the 252-day versus 365-day calendars correctly, interpreting sentiment validation cautiously, and deciding to report the negative sentiment-fusion result rather than modifying the model simply to improve performance.

Overall, I used AI as an iterative assistant, while validation, interpretation and final decisions remained my responsibility.
