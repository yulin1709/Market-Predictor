@echo off
:: AI Market Impact Predictor — Daily Pipeline
:: Runs every weekday morning to fetch news, fill prices, backfill outcomes
:: Set up via Windows Task Scheduler to run at 08:00 MYT (UTC+8)

set ROOT=%~dp0..
set PYTHON=%ROOT%\.venv\Scripts\python.exe
set LOG=%ROOT%\logs\daily_%date:~-4,4%%date:~-7,2%%date:~0,2%.log

:: Create logs folder if missing
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

echo ===== Daily Pipeline %date% %time% ===== >> "%LOG%" 2>&1

:: 1. Fetch last 2 days of news (fast, catches today + yesterday)
echo [1/3] Collecting news... >> "%LOG%" 2>&1
"%PYTHON%" "%ROOT%\market_predictor\data\collect_news.py" --days 2 --max-pages 2 >> "%LOG%" 2>&1

:: 2. Fill exit prices for yesterday's predictions
echo [2/3] Backfilling actuals... >> "%LOG%" 2>&1
"%PYTHON%" "%ROOT%\market_predictor\data\backfill_actuals.py" >> "%LOG%" 2>&1

:: 3. Fetch latest prices
echo [3/3] Collecting prices... >> "%LOG%" 2>&1
"%PYTHON%" "%ROOT%\market_predictor\data\collect_prices.py" >> "%LOG%" 2>&1

echo ===== Done %time% ===== >> "%LOG%" 2>&1
