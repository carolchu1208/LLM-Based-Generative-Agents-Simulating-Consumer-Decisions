#!/usr/bin/env python
# coding: utf-8

"""
Stability Test Runner - Monitors and runs multiple iterations of the 7-day simulation.

This script launches main_simulation.py as a SEPARATE SUBPROCESS for each run,
ensuring complete isolation between runs (fresh MemoryManager, fresh timestamp).

Usage:
    python stability_test_runner.py [--runs N] [--start-from M]

Arguments:
    --runs N        Number of iterations to run (default: 20)
    --start-from M  Start from iteration M (useful for resuming, default: 1)
"""

import os
import sys
import json
import time
import argparse
import subprocess
import glob
from datetime import datetime


def get_latest_memory_file(records_dir):
    """Get the most recently created memory file."""
    pattern = os.path.join(records_dir, 'simulation_agents', 'agents_memories_*.jsonl')
    files = glob.glob(pattern)
    if not files:
        return None, None
    latest = max(files, key=os.path.getctime)
    # Extract timestamp from filename
    filename = os.path.basename(latest)
    timestamp = filename.replace('agents_memories_', '').replace('.jsonl', '')
    return latest, timestamp


def get_latest_day_hour(memory_file):
    """Get the latest day and hour from a memory file."""
    if not memory_file or not os.path.exists(memory_file):
        return 0, 0
    try:
        # Read last line
        with open(memory_file, 'rb') as f:
            f.seek(-2, 2)
            while f.read(1) != b'\n':
                f.seek(-2, 1)
            last_line = f.readline().decode()

        data = json.loads(last_line)
        return data.get('day', 0), data.get('hour', 0)
    except:
        return 0, 0


def monitor_simulation(process, run_number, records_dir, start_timestamp):
    """
    Monitor a running simulation subprocess.

    Returns:
        tuple: (success: bool, end_timestamp: str, days_completed: int)
    """
    last_day, last_hour = 0, 0
    last_file_size = 0
    stall_count = 0

    print(f"\n[Run {run_number}] Monitoring simulation...")

    while process.poll() is None:
        # Find the memory file for this run
        memory_file, timestamp = get_latest_memory_file(records_dir)

        if memory_file and timestamp and timestamp > start_timestamp:
            # Check progress
            current_size = os.path.getsize(memory_file)
            day, hour = get_latest_day_hour(memory_file)

            if day != last_day or hour != last_hour:
                print(f"[Run {run_number}] Day {day}, Hour {hour} - File: {current_size/1024:.1f}KB")
                last_day, last_hour = day, hour
                stall_count = 0
            elif current_size == last_file_size:
                stall_count += 1
                if stall_count > 5:  # 5 minutes of no progress (5 × 60 sec)
                    print(f"[Run {run_number}] WARNING: No progress for 5 minutes")

            last_file_size = current_size

        time.sleep(60)  # Check every 60 seconds

    # Process ended - check exit code
    exit_code = process.returncode

    # Get final state
    memory_file, end_timestamp = get_latest_memory_file(records_dir)
    if memory_file and end_timestamp and end_timestamp > start_timestamp:
        days_completed, _ = get_latest_day_hour(memory_file)
    else:
        days_completed = 0
        end_timestamp = None

    success = (exit_code == 0 and days_completed >= 7)

    return success, end_timestamp, days_completed, exit_code


def run_single_experiment(run_number, total_runs, python_path, script_dir, records_dir):
    """
    Run a single simulation as a subprocess.

    Returns:
        dict with run results
    """
    result = {
        'run_id': run_number,
        'timestamp': None,
        'status': 'unknown',
        'start_time': None,
        'end_time': None,
        'duration_seconds': 0,
        'days_completed': 0,
        'exit_code': None,
        'error': None
    }

    start_time = datetime.now()
    result['start_time'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
    start_timestamp = start_time.strftime('%Y%m%d_%H%M%S')

    print("\n" + "=" * 70)
    print(f"  STABILITY TEST - Run {run_number}/{total_runs}")
    print(f"  Started at: {result['start_time']}")
    print("=" * 70)

    # Launch main_simulation.py as subprocess
    main_script = os.path.join(script_dir, 'main_simulation.py')

    try:
        process = subprocess.Popen(
            [python_path, '-u', main_script],  # -u for unbuffered output
            cwd=script_dir
            # stdout/stderr not captured - prints directly to terminal
        )

        print(f"[Run {run_number}] Subprocess started (PID: {process.pid})")

        # Monitor the simulation
        success, end_timestamp, days_completed, exit_code = monitor_simulation(
            process, run_number, records_dir, start_timestamp
        )

        result['timestamp'] = end_timestamp
        result['days_completed'] = days_completed
        result['exit_code'] = exit_code

        if success:
            result['status'] = 'completed'
            print(f"\n[Run {run_number}] ✓ COMPLETED (Days: {days_completed})")
        else:
            result['status'] = 'error'
            result['error'] = f'Exit code: {exit_code}, Days: {days_completed}'
            print(f"\n[Run {run_number}] ✗ FAILED (Exit: {exit_code}, Days: {days_completed})")

    except KeyboardInterrupt:
        print(f"\n[Run {run_number}] Interrupted by user")
        if process:
            process.terminate()
            process.wait()
        result['status'] = 'interrupted'
        result['error'] = 'User interrupted'
        raise

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"\n[Run {run_number}] Error: {e}")

    finally:
        end_time = datetime.now()
        result['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
        result['duration_seconds'] = (end_time - start_time).total_seconds()

    return result


def run_stability_test(total_runs=20, start_from=1):
    """
    Run multiple iterations of the simulation for stability testing.
    Each run is a separate subprocess for complete isolation.
    """
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    records_dir = os.path.join(script_dir, '..', 'LLMAgentsTown_memory_records')
    results_dir = os.path.join(script_dir, '..', 'stability_test_results')
    logs_dir = os.path.join(results_dir, 'run_logs')

    # Find Python executable (use the one running this script)
    python_path = sys.executable

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    manifest_path = os.path.join(results_dir, 'run_manifest.json')

    overall_start = datetime.now()

    print("\n" + "#" * 70)
    print(f"#  STABILITY TEST: {total_runs} iterations (subprocess mode)")
    print(f"#  Starting from run: {start_from}")
    print(f"#  Started at: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Python: {python_path}")
    print("#" * 70 + "\n")

    # Load or create manifest
    if start_from > 1 and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            results = manifest.get('runs', [])
    else:
        manifest = {
            'total_runs_planned': total_runs,
            'start_time': overall_start.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': 'subprocess',
            'runs': []
        }
        results = []

    completed_runs = sum(1 for r in results if r.get('status') == 'completed')
    failed_runs = sum(1 for r in results if r.get('status') not in ['completed', None])

    try:
        for run_num in range(start_from, total_runs + 1):
            result = run_single_experiment(
                run_num, total_runs, python_path, script_dir, records_dir
            )
            results.append(result)

            if result['status'] == 'completed':
                completed_runs += 1
            else:
                failed_runs += 1

            # Update manifest
            manifest['runs'] = results
            manifest['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            # Save individual run log
            run_log_path = os.path.join(logs_dir, f'run_{run_num:03d}.json')
            with open(run_log_path, 'w') as f:
                json.dump(result, f, indent=2)

            # Progress summary
            print("\n" + "-" * 50)
            print(f"Run {run_num}: {result['status'].upper()}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"Duration: {result['duration_seconds']/60:.1f} minutes")
            print(f"Progress: {run_num}/{total_runs} ({run_num/total_runs*100:.1f}%)")
            print(f"Completed: {completed_runs} | Failed: {failed_runs}")
            print("-" * 50)

            # Pause between runs
            if run_num < total_runs:
                print("\nStarting next run in 10 seconds...")
                time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nStability test interrupted by user.")

    finally:
        overall_end = datetime.now()
        total_duration = (overall_end - overall_start).total_seconds()

        manifest['end_time'] = overall_end.strftime('%Y-%m-%d %H:%M:%S')
        manifest['total_duration_seconds'] = total_duration
        manifest['completed_runs'] = completed_runs
        manifest['failed_runs'] = failed_runs
        manifest['runs'] = results

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "#" * 70)
        print("#  STABILITY TEST COMPLETE")
        print("#" * 70)
        print(f"\nTotal runs: {len(results)}")
        print(f"Completed: {completed_runs}")
        print(f"Failed: {failed_runs}")
        print(f"Duration: {total_duration/3600:.2f} hours")
        print(f"\nManifest: {manifest_path}")

        # List output files
        print("\nOutput files created:")
        pattern = os.path.join(records_dir, 'simulation_agents', 'agents_memories_*.jsonl')
        recent_files = sorted(glob.glob(pattern), key=os.path.getctime, reverse=True)[:total_runs]
        for f in recent_files:
            size = os.path.getsize(f) / 1024 / 1024
            print(f"  {os.path.basename(f)} ({size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Run stability tests (subprocess mode)'
    )
    parser.add_argument('--runs', type=int, default=20,
                       help='Number of iterations (default: 20)')
    parser.add_argument('--start-from', type=int, default=1,
                       help='Start from run N (default: 1)')

    args = parser.parse_args()

    if args.runs < 1:
        print("Error: runs must be >= 1")
        sys.exit(1)

    if args.start_from < 1 or args.start_from > args.runs:
        print(f"Error: start-from must be 1-{args.runs}")
        sys.exit(1)

    run_stability_test(total_runs=args.runs, start_from=args.start_from)


if __name__ == "__main__":
    main()
