ROLE
You are a senior quantitative software engineer. You will build "FinAIBot", an AI platform that generates, backtests, improves and validates Forex trading strategies, according to a strict client acceptance protocol. Work in small milestones. After each milestone, stop, summarize what you built, show how to run the tests, and wait for my confirmation before starting the next one.

PROJECT SUMMARY
- Market: Forex, 3 currency pairs (configurable, to be confirmed), up to 10 years of history.
- The AI designs complete strategies (entries, exits, indicators, SL/TP, sizing, filters, sessions, "when not to trade"), tests them, analyses failures, and produces new versions.
- Strategies are validated by an independent backtester + validator + risk manager that the AI can never modify or bypass.
- MetaTrader 5 demo integration comes LAST, only after historical validation. Real-money trading is out of scope.

TECH STACK
- Engine (backtester, metrics, validator, risk manager, AI loop): Python 3.11+, pandas, NumPy, Numba where needed, pytest.
- API: FastAPI. Database: PostgreSQL (SQLAlchemy). Market data: Parquet files, never edited after import.
- Frontend (later milestone): React + TypeScript, TradingView Lightweight Charts.
- Long jobs: task queue. Generated code, if ever executed, runs only in a sandbox (Docker, no network, CPU/memory/time limits).
- Use a Git-friendly structure: /engine, /api, /frontend, /tests, /docs, /data (git-ignored).

NON-NEGOTIABLE RULES
1. All results (profit, costs, returns, metrics) are computed from recorded trades and market data. Never from manual input or text produced by the AI.
2. No look-ahead bias: every data point is timestamped and the engine only sees data available at the simulated moment.
3. Reproducibility: same data version + same strategy version + same parameters = same results. Store data hashes, config, strategy version and run ID with every run.
4. The validator and risk manager are separate modules. The AI and the strategies cannot change their rules, thresholds or protections.
5. Costs must include spread, commission, slippage, swaps/overnight financing and currency conversion where applicable, without double counting.
6. Distinguish realized P&L from unrealized P&L. Open positions at month end are marked at market price with estimated liquidation costs.
7. Never mark a strategy ACCEPTED if any mandatory metric was not computed or any required test was not run.

STRATEGY FORMAT
Define a versioned JSON schema for strategies: id, version, created_at, parent_version, pair, timeframe, entry rules, exit rules, indicators + parameters, stop-loss, take-profit, risk/reward, position sizing, trend/volatility/market filters, allowed sessions, invalidation conditions, capital protection rules. Validate every strategy against the schema before running it. Strategies are data, not arbitrary code.

BACKTESTER REQUIREMENTS
- Load and validate data (gaps, duplicates, weekends/holidays, bad ticks).
- Simulate orders, positions, margin, leverage and exposure; handle rejected or partially filled orders.
- Variable spreads when the data allows it. Intrabar execution must not rely only on close prices.
- Log every trade and important event. Produce equity and drawdown curves.
- Daily loss is measured on equity (realized + unrealized + costs). Cap: 5% of initial capital, with a preventive stop at 4%.
- Total loss is measured against initial capital. Cap: 10%, with a preventive stop at 8%.
- Limits are checked DURING the test, not only at the end. Exceeding a cap rejects the test.

VALIDATION RULES (all checked individually, no global score can compensate)
- R-01 Net return >= 3% in EVERY complete test month (not just on average)
- R-02 Daily loss <= 5% of initial capital
- R-03 Total loss <= 10% of initial capital
- R-04 Net Profit Factor >= 1.30
- R-05 Out-of-sample Profit Factor >= 1.15
- R-06 Net expectancy per trade strictly positive
- R-07 Max drawdown <= 8% of initial capital
- R-08 Annualized Sharpe >= 1.00
- R-09 At least 300 trades over the historical period
- R-10 At least 12 complete months tested
- R-11 Net result still positive with variable costs increased by 50%
- R-12 Out-of-sample monthly return >= 3% in every complete month
- R-13 Zero unresolved critical execution errors
- R-14 Zero risk-limit breaches
Thresholds live in a configuration file frozen before official runs. They must not be adjusted after seeing results.
Final status, generated automatically: ACCEPTED, REJECTED (at least one eliminatory condition violated), or INCOMPLETE TEST (missing data or evidence).

TEST DESIGN
- Chronological split: development/optimization data vs. independent validation data. Reserve at least 4 full months out-of-sample, locked away from the AI.
- Support walk-forward testing and a year-by-year mode (N-1, N-2 ... N-10). A year can only be passed if every mandatory criterion is met. Missing or unreliable years are never counted as validated.
- Two modes: "exploration" (free, not official) and "official validation" (locked protocol, results cannot be edited, no skipping years).
- Maximum number of improvement attempts per strategy is configurable. After the limit the strategy is "not validated". Log every attempt and variant to detect overfitting.

AI IMPROVEMENT LOOP (later milestone)
After each test: analyse results and failures, identify likely causes of losses/drawdown, form hypotheses, generate a new strategy version (as JSON), re-test under identical conditions, compare versions, keep the original and the full history. Optimize for risk, stability, costs and out-of-sample robustness, not only historical profit.

REPORT (generated automatically per strategy)
Strategy ID and version; pair, timeframe, period, initial capital; monthly final capital and net return; gross profit, cost breakdown by category, net profit; Profit Factor, win rate, win/loss ratio, expectancy; Sharpe, max drawdown, recovery factor; max daily and total loss observed; number of trades and losing-streak statistics; out-of-sample and walk-forward results; increased-cost scenario; security test results and incidents; list of passed and failed criteria; final decision with justification; run ID, timestamp, data references and everything needed to reproduce.

MANDATORY TESTS (automated, each reported as PASS/FAIL with initial conditions, expected action, observed result)
Daily and total preventive-stop triggers; loss cap exceeded despite protection; broker disconnection with an open position; rejected or partial order; missing or inconsistent market data; month rollover with open positions; engine restart mid-run. Also: metrics checked against an independent reference calculation, no future-data leakage, no double-counted costs, reproducibility check.

MILESTONES (do one at a time)
M1. Repo structure, strategy JSON schema, data loader/validator, backtester core with the cost model, equity curve, trade log, risk limits. Test on a small sample or synthetic dataset.
M2. Metrics module (all indicators above), checked against an independent implementation.
M3. Validator (R-01..R-14), frozen config, automatic status, report generator.
M4. AI loop: generation, analysis, versioning, attempt budget, overfitting guards.
M5. FastAPI + PostgreSQL, then the React frontend (dashboard, validation center, version manager, reports, TradingView-style replay).
M6. MetaTrader 5 demo connector (Windows, `MetaTrader5` package): connection, data, demo orders, comparison of simulated vs. observed execution.

WORKING METHOD
- Start with M1 only. Before coding, list your assumptions and any questions that block you.
- Write tests alongside the code. Keep functions small and documented.
- Do not invent market data or results. If you need data you do not have, use clearly labeled synthetic data and say so.
- Do not claim any strategy is profitable. The platform is judged by whether it measures and enforces the rules correctly, not by whether a strategy passes.
- At the end of each milestone, give: what was built, how to run it, test output, known limitations, and what the next milestone needs.
## IMPLEMENTATION CONTRACT AND MILESTONE ACCEPTANCE

### A. Requirement traceability

Maintain a requirements checklist for every milestone. Each requirement must have:
- A unique identifier.
- Its implementation status: `PASS`, `FAIL`, `BLOCKED`, or `NOT STARTED`.
- The relevant source files.
- The associated automated tests.
- The actual verification command and result.
- Any known limitation or dependency.

A requirement may be marked `PASS` only when its implementation and required tests have been verified. A passing test suite does not automatically mean that every requirement is satisfied.

### B. Financial definitions and accounting

Document the precise formulas and conventions used for:
- Equity, realized P&L, unrealized P&L, and net P&L.
- Daily loss and total loss.
- Maximum drawdown.
- Net Profit Factor and net expectancy.
- Sharpe and Sortino ratios.
- Monthly and out-of-sample returns.
- Used margin, free margin, exposure, and liquidation values.

Specify the measurement frequency, timezone, denominator, treatment of open positions, and behavior when a metric cannot be calculated.

Every transaction cost must be accounted for exactly once. All reports must reconcile with the underlying trade ledger and equity curve.

### C. Risk-limit state transitions

Define the exact behavior at each preventive threshold and eliminatory loss cap.

Risk checks must run at the required simulation events, including applicable intrabar execution events. The engine must record the triggering measurement, timestamp, action taken, and resulting test status.

If the engine cannot establish whether a risk limit was respected because required data or calculations are unavailable, it must not declare the corresponding criterion passed.

### D. Execution assumptions

Document the timing and price assumptions for entries, exits, gaps, stop-losses, take-profits, spread, slippage, commissions, financing, rejected orders, and partial fills.

When the available data cannot support an execution simulation, mark the limitation explicitly. Never fabricate fills or prices.

### E. Data and reproducibility

For each run, record:
- Immutable dataset identifiers and hashes.
- Strategy ID, version, and parameters.
- Frozen validation configuration and its hash.
- Engine/software version and relevant dependency versions.
- Initial capital, account-currency assumptions, cost model, and execution settings.
- Run ID, timestamp, test status, and reproducible result references.

The same inputs and software version must produce the same deterministic results within the documented numerical tolerances.

### F. Out-of-sample protection

Enforce the validation-data boundary in the data-access and optimization layers. The AI must not access reserved validation data or its results during strategy generation and optimization.

Any breach of this boundary must be logged and must prevent the strategy from being accepted under the official protocol.

### G. Milestone completion gate

At the end of each milestone:
1. Map every milestone requirement to implementation and test evidence.
2. Run all required tests and checks that are supported by the implemented functionality.
3. Identify every failed, blocked, incomplete, or deferred requirement.
4. Do not report the milestone as complete while mandatory requirements remain unsatisfied.
5. Present the results and wait for explicit approval before starting the next milestone.

Mandatory tests belonging to later milestones must remain explicitly tracked until they can be executed. Missing evidence must never be represented as a successful result.

Final strategy status must be calculated automatically from the frozen acceptance protocol: `ACCEPTED`, `REJECTED`, or `INCOMPLETE TEST`. No strategy may be marked `ACCEPTED` when a mandatory criterion or required test has not been verified.