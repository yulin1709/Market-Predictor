@echo off
REM ============================================================
REM run_pipeline.bat — Daily pipeline with exit code checking
REM Schedule THIS file in Windows Task Scheduler, not individual scripts.
REM ============================================================

echo [pipeline] Starting daily pipeline at %date% %time%

echo [pipeline] Step 1: Collecting news...
python market_predictor\data\collect_news.py
if %errorlevel% neq 0 (
    echo [pipeline] ERROR: collect_news.py failed. Stopping.
    goto :end
)

echo [pipeline] Step 2: Checking news feed health...
python market_predictor\data\check_news_feed.py
if %errorlevel% neq 0 (
    echo [pipeline] WARNING: Insufficient articles today. Stopping pipeline.
    echo [pipeline] Signal generation skipped — no predictions will be logged.
    goto :end
)

echo [pipeline] Step 3: Aligning and labelling...
python market_predictor\data\align_and_label.py

echo [pipeline] Step 4: Extracting entities...
python market_predictor\features\extract_entities.py

echo [pipeline] Step 5: Backfilling prices and outcomes...
python market_predictor\data\backfill_actuals.py

echo [pipeline] Done at %time%
goto :eof

:end
echo [pipeline] Pipeline stopped due to insufficient news data.
