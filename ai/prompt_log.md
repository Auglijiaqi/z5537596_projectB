# AI Workflow Log - FINS5545 Project B

## About this log

This is a curated reconstruction of selected AI-assisted workflows used during my Project B development.

It is not intended to be a verbatim transcript of every AI conversation. Instead, it records the most important tasks where AI contributed to coding, debugging, validation, design or project organisation, together with the checks and decisions I made before accepting the output.

I treated AI suggestions as proposals rather than final answers.

---

# Case 1 - Preventing Look-Ahead Bias in the Walk-Forward Backtest

## Task / Prompt

I asked AI to help design and review a walk-forward out-of-sample portfolio backtest for Equity, Crypto and Combined funds.

A key requirement I highlighted was that portfolio weights must only use information available before each rebalance date.

## AI Contribution

AI helped structure the backtest around:

- a historical estimation window,
- a rebalance date,
- weights estimated from historical observations,
- application of those weights to the subsequent holding period,
- an audit table recording training and rebalance dates.

## Risk Identified

The main risk was that a technically correct-looking backtest could still contain hidden look-ahead if the estimation window overlapped with the live holding period.

I therefore did not rely only on the reported performance.

## My Validation

I generated and inspected a rebalance audit.

I checked that:

`training_end < rebalance_date`

for every rebalance.

The final audit contained 436 rebalance records, and all records passed this condition.

I also checked that all optimisation runs completed successfully and no fallback solution was used.

## My Decision

I accepted the final walk-forward implementation only after these audit checks.

This became an important safeguard because good portfolio performance alone would not demonstrate that the backtest was valid.

---

# Case 2 - Equity and Crypto Calendar Treatment

## Task / Prompt

I used AI to help review how the portfolio code should handle the different trading calendars of equities and cryptocurrencies.

## AI Contribution

AI helped organise separate assumptions for the two asset classes rather than applying one common annualisation rule.

## Risk Identified

Equities trade approximately 252 days per year, while cryptocurrencies trade every day.

Using 252 for Crypto would make its annualised risk and return statistics inconsistent with the underlying data frequency.

## My Validation and Correction

I checked the final implementation and confirmed:

### Equity and Combined
- estimation window: 252 observations
- rebalance step: 21 trading observations
- annualisation factor: 252
- first OOS date: 4 January 2021

### Crypto
- estimation window: 365 observations
- rebalance step: 30 calendar observations
- annualisation factor: 365
- first OOS date: 1 January 2021

## My Decision

I retained separate equity and crypto calendar assumptions.

This was not only a coding choice but also affected how the performance statistics were interpreted in the report.

---

# Case 3 - Improving VADER without Over-Claiming the Result

## Task / Prompt

I asked AI to help improve standard VADER for financial news headlines.

I wanted the extension to recognise financial expressions that generic VADER may classify as neutral.

## AI Contribution

AI helped propose a finance-specific lexicon and a sentiment workflow consisting of:

1. headline-level sentiment,
2. ticker-day aggregation,
3. equal-weighted sector aggregation,
4. a coverage measure,
5. a lag before the signal is used in portfolio decisions.

The final finance-aware lexicon contained 25 additional financial terms.

## Validation

I compared plain VADER with finance-aware VADER.

The results were:

- Plain VADER neutral rate: approximately 48.9%
- Finance-aware neutral rate: approximately 46.9%
- 8,741 headline scores changed

## Important Interpretation Check

A lower neutral classification rate could easily be presented as evidence that the new model is “better”.

I rejected that interpretation.

The comparison only shows that the finance-aware lexicon captures a broader range of financial language.

It does not demonstrate:

- higher classification accuracy against ground truth,
- return predictability,
- or improved portfolio performance.

## My Decision

I kept the finance-aware VADER as an innovation, but clearly separated language coverage from investment forecasting ability.

I later recommended manually labelled financial-news validation as a future improvement.

---

# Case 4 - Keeping a Negative Sentiment-Fusion Result

## Task / Prompt

I asked AI to help integrate sector sentiment into the Equity Minimum Variance and Combined Minimum Variance portfolios.

## AI Contribution

The resulting approach used:

- lagged relative sector sentiment,
- equity-only sentiment tilts,
- coverage-adjusted signal strength,
- a sentiment tilt parameter of 0.30.

For the Combined fund, the crypto allocation was preserved so that sentiment only changed the equity component.

## Validation

I verified that:

- sentiment was lagged before portfolio use,
- the Combined crypto sleeve was unchanged by sentiment,
- the 25% crypto cap remained satisfied,
- portfolio constraints remained valid.

I then compared the base and sentiment-enhanced funds out of sample.

### Equity Minimum Variance
Sharpe:
0.543 -> 0.518

### Combined Minimum Variance
Sharpe:
0.551 -> 0.526

Annualised returns also fell by approximately 0.3 percentage points and maximum drawdowns became slightly deeper.

## Decision Point

After seeing this result, one possible response would have been to repeatedly change the sentiment parameter until the final performance improved.

I chose not to do this.

The 0.30 sentiment tilt remained fixed rather than being retrospectively optimised against the final OOS results.

## My Decision

I reported the underperformance rather than hiding it.

This changed my interpretation of the sentiment feature: it became evidence that additional information does not automatically create investment value.

It also motivated my recommendation that future fusion models should jointly consider signal strength, news coverage, turnover and transaction costs.

---

# Case 5 - Turning the Analysis into an Investor Product

## Task / Prompt

I used AI to help organise the Streamlit application around the investor journey required by the project.

## AI Contribution

The dashboard was structured around:

- Compare Funds
- Fund Fact Sheet
- Build Allocation
- Sentiment Lens
- Methodology
- Data Check

## Implementation Risk

An early design concern was whether the deployed app should rerun portfolio optimisation and sentiment analysis whenever a user opened the app.

This would make the application slower and less reproducible, and could cause deployment problems.

## My Validation

I checked that the final app reads pre-computed outputs from the `results/` directory.

The deployed app does not rerun:

- the full portfolio backtest,
- portfolio optimisation,
- or VADER sentiment scoring.

I manually tested the investor journey locally and checked that the main tabs loaded correctly.

## My Decision

I retained the pre-computed architecture because it keeps the dashboard lightweight and ensures that the figures shown to investors are consistent with the report.

---

# Case 6 - Final Submission Audit

## Task / Prompt

I used AI as a final reviewer to compare my project with the brief and help identify mechanical submission issues.

## AI Contribution

AI suggested checking:

- report length,
- required figures and tables,
- report filename,
- project structure,
- temporary files,
- the hand-in checker.

## My Validation

I manually revised the report and reduced the final written report to 10 pages.

The report contains:

- performance metrics,
- return-risk comparison,
- Growth of $1,
- drawdown,
- portfolio weights,
- sector sentiment,
- VADER validation,
- sentiment fusion before-and-after,
- app investor journey,
- critical reflection,
- three recommendations.

I then ran:

`python scripts/check_handin.py`

The first final-stage run passed 21 checks but identified:

- `.DS_Store`
- `results/.DS_Store`
- `__pycache__`
- `.pyc`

as files that should be removed.

I removed these files and reran the checker.

The final result was:

`23 checks passed.`

`All checks passed - ready to zip and deploy.`

## My Decision

I treated the hand-in checker as a mechanical validation tool, not as proof that the academic quality of the project was complete.

I separately reviewed the report, app, AI workflow and economic interpretation.

---

# Overall Reflection

AI substantially accelerated the technical development of NexaVest, especially during coding, debugging, validation and report organisation.

However, the most valuable part of the workflow was not simply obtaining AI-generated code.

The main decisions that required my own judgement were:

- verifying no look-ahead rather than trusting backtest performance,
- correctly separating 252-day and 365-day market calendars,
- enforcing the 25% Combined crypto cap,
- distinguishing broader VADER vocabulary coverage from predictive accuracy,
- retaining a negative sentiment-fusion result instead of tuning it away,
- checking that the deployed app uses pre-computed results,
- and independently interpreting the financial implications of the outputs.

The project therefore used AI as an iterative assistant, while validation and final decision-making remained part of my own workflow.
