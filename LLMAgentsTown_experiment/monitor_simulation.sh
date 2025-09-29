#!/bin/bash

LOG_FILE="logs/simulation_20250927_030029.log"
REPORT_FILE="logs/hourly_report.txt"
PID=52610
LAST_HOUR=""

echo "=== SIMULATION MONITORING STARTED ===" > "$REPORT_FILE"
echo "Start Time: $(date)" >> "$REPORT_FILE"
echo "PID: $PID" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

while true; do
    # Check if simulation is still running
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] ❌ SIMULATION STOPPED" | tee -a "$REPORT_FILE"
        exit 1
    fi

    # Get current simulation time
    CURRENT_TIME=$(tail -100 "$LOG_FILE" | grep -E "Time: Day [0-9]+, Hour [0-9]+" | tail -1)

    if [ -n "$CURRENT_TIME" ] && [ "$CURRENT_TIME" != "$LAST_HOUR" ]; then
        LAST_HOUR="$CURRENT_TIME"

        echo "" >> "$REPORT_FILE"
        echo "========================================" >> "$REPORT_FILE"
        echo "[$(date +%H:%M:%S)] $CURRENT_TIME" | tee -a "$REPORT_FILE"
        echo "========================================" >> "$REPORT_FILE"

        # Check for errors
        ERRORS=$(tail -500 "$LOG_FILE" | grep -E "ERROR|CRITICAL|Traceback|AttributeError|KeyError" | tail -5)
        if [ -n "$ERRORS" ]; then
            echo "⚠️  ERRORS FOUND:" | tee -a "$REPORT_FILE"
            echo "$ERRORS" | tee -a "$REPORT_FILE"
        fi

        # Check for failed actions
        FAILED=$(tail -500 "$LOG_FILE" | grep -E "Failed to|failed|FAILED" | grep -v "No failed" | tail -3)
        if [ -n "$FAILED" ]; then
            echo "❌ FAILED ACTIONS:" | tee -a "$REPORT_FILE"
            echo "$FAILED" | tee -a "$REPORT_FILE"
        fi

        # Check for emergency replans
        EMERGENCY=$(tail -500 "$LOG_FILE" | grep -E "EMERGENCY|emergency.*replan|critically low energy" | tail -3)
        if [ -n "$EMERGENCY" ]; then
            echo "🚨 EMERGENCY EVENTS:" | tee -a "$REPORT_FILE"
            echo "$EMERGENCY" | tee -a "$REPORT_FILE"
        fi

        # Check for critical energy
        CRITICAL_ENERGY=$(tail -500 "$LOG_FILE" | grep -E "CRITICAL.*energy|energy.*0|forcing to residence" | tail -3)
        if [ -n "$CRITICAL_ENERGY" ]; then
            echo "⚡ CRITICAL ENERGY ISSUES:" | tee -a "$REPORT_FILE"
            echo "$CRITICAL_ENERGY" | tee -a "$REPORT_FILE"
        fi

        # Check for API failures
        API_FAIL=$(tail -500 "$LOG_FILE" | grep -E "API.*failed|API.*error|timeout" | tail -2)
        if [ -n "$API_FAIL" ]; then
            echo "🌐 API ISSUES:" | tee -a "$REPORT_FILE"
            echo "$API_FAIL" | tee -a "$REPORT_FILE"
        fi

        # If no issues found
        if [ -z "$ERRORS" ] && [ -z "$FAILED" ] && [ -z "$EMERGENCY" ] && [ -z "$CRITICAL_ENERGY" ] && [ -z "$API_FAIL" ]; then
            echo "✅ No issues detected" | tee -a "$REPORT_FILE"
        fi
    fi

    # Check every 30 seconds
    sleep 30
done