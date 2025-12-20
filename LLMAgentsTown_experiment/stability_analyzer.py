#!/usr/bin/env python
# coding: utf-8

"""
Stability Analyzer - Analyzes variance across multiple simulation runs.

This script reads the run_manifest.json created by stability_test_runner.py,
parses all the record files from each run, and computes variance analysis.

Usage:
    python stability_analyzer.py [--manifest PATH]

Output:
    stability_test_results/aggregated_analysis.json
"""

import os
import sys
import json
import argparse
import math
from collections import defaultdict
from typing import Dict, List, Any, Optional


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute mean, std_dev, coefficient of variation, min, and max.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {'mean': 0, 'std_dev': 0, 'cv': 0, 'min': 0, 'max': 0, 'count': 0}

    n = len(values)
    mean = sum(values) / n

    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0

    cv = std_dev / mean if mean != 0 else 0

    return {
        'mean': round(mean, 2),
        'std_dev': round(std_dev, 2),
        'cv': round(cv, 4),
        'min': round(min(values), 2),
        'max': round(max(values), 2),
        'count': n
    }


def parse_metrics_log(filepath: str) -> Dict[str, Any]:
    """
    Parse metrics_log JSONL file for economic metrics.

    Returns:
        Dictionary with aggregated economic metrics
    """
    metrics = {
        'total_revenue': 0,
        'transaction_count': 0,
        'discount_transactions': 0,
        'discount_revenue': 0,
        'revenue_by_shop': defaultdict(float),
        'revenue_by_day': defaultdict(float),
        'transactions_by_day': defaultdict(int),
        'revenue_by_meal_type': defaultdict(float)
    }

    if not os.path.exists(filepath):
        return metrics

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get('event_type') == 'sale':
                    data = record.get('data', {})
                    amount = data.get('amount', 0)
                    shop = data.get('shop_name', 'Unknown')
                    day = data.get('day', 0)
                    has_discount = data.get('has_discount', False)
                    meal_types = data.get('meal_types', [])

                    metrics['total_revenue'] += amount
                    metrics['transaction_count'] += 1
                    metrics['revenue_by_shop'][shop] += amount
                    metrics['revenue_by_day'][day] += amount
                    metrics['transactions_by_day'][day] += 1

                    if has_discount:
                        metrics['discount_transactions'] += 1
                        metrics['discount_revenue'] += amount

                    for meal_type in meal_types:
                        metrics['revenue_by_meal_type'][meal_type] += amount

            except json.JSONDecodeError:
                continue

    # Convert defaultdicts to regular dicts
    metrics['revenue_by_shop'] = dict(metrics['revenue_by_shop'])
    metrics['revenue_by_day'] = dict(metrics['revenue_by_day'])
    metrics['transactions_by_day'] = dict(metrics['transactions_by_day'])
    metrics['revenue_by_meal_type'] = dict(metrics['revenue_by_meal_type'])

    return metrics


def parse_agents_memories(filepath: str) -> Dict[str, Any]:
    """
    Parse agents_memories JSONL file for behavior metrics.

    Returns:
        Dictionary with aggregated behavior metrics
    """
    metrics = {
        'total_travel_steps': 0,
        'travel_by_agent': defaultdict(int),
        'replan_count': 0,
        'replans_by_agent': defaultdict(int),
        'conversation_count': 0,
        'conversations_by_agent': defaultdict(int),
        'plan_creation_count': 0,
        'memory_count': 0
    }

    if not os.path.exists(filepath):
        return metrics

    seen_conversations = set()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                metrics['memory_count'] += 1

                record_type = record.get('type', '')
                agent = record.get('agent_name', 'Unknown')
                content = record.get('content', {})

                if record_type == 'TRAVEL':
                    steps = content.get('steps', 0)
                    metrics['total_travel_steps'] += steps
                    metrics['travel_by_agent'][agent] += steps

                elif record_type == 'PLAN_UPDATE':
                    metrics['replan_count'] += 1
                    metrics['replans_by_agent'][agent] += 1

                elif record_type == 'PLAN_CREATION':
                    metrics['plan_creation_count'] += 1

                elif record_type == 'CONVERSATION':
                    # Count unique conversations (avoid double-counting)
                    dialogue = content.get('dialogue', '')
                    participants = tuple(sorted(content.get('participants', [])))
                    day = record.get('day', 0)
                    hour = record.get('hour', 0)
                    turn = content.get('turn_number', 0)

                    conv_key = (participants, day, hour, turn)
                    if conv_key not in seen_conversations:
                        seen_conversations.add(conv_key)
                        if turn == 0:  # Only count first turn as new conversation
                            metrics['conversation_count'] += 1
                            for p in participants:
                                metrics['conversations_by_agent'][p] += 1

            except json.JSONDecodeError:
                continue

    # Convert defaultdicts to regular dicts
    metrics['travel_by_agent'] = dict(metrics['travel_by_agent'])
    metrics['replans_by_agent'] = dict(metrics['replans_by_agent'])
    metrics['conversations_by_agent'] = dict(metrics['conversations_by_agent'])

    return metrics


def parse_daily_summary(filepath: str) -> Dict[str, Any]:
    """
    Parse daily_summary JSON file for agent financial states.

    Returns:
        Dictionary with agent final states and aggregated metrics
    """
    metrics = {
        'days_completed': 0,
        'agent_final_money': {},
        'agent_total_income': defaultdict(float),
        'agent_total_expenses': defaultdict(float),
        'agent_final_grocery': {},
        'agent_final_energy': {}
    }

    if not os.path.exists(filepath):
        return metrics

    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return metrics

    # Count days completed
    day_keys = [k for k in data.keys() if k.startswith('day_')]
    metrics['days_completed'] = len(day_keys)

    # Get final day data (day_7 or highest available)
    final_day_key = f'day_{metrics["days_completed"]}'
    if final_day_key not in data:
        final_day_key = sorted(day_keys)[-1] if day_keys else None

    if final_day_key and final_day_key in data:
        final_day = data[final_day_key]

        # Extract final states
        metrics['agent_final_money'] = {
            agent: info.get('money', 0)
            for agent, info in final_day.get('financial', {}).items()
        }

        metrics['agent_final_grocery'] = final_day.get('grocery', {})
        metrics['agent_final_energy'] = final_day.get('energy', {})

    # Calculate total income and expenses across all days
    for day_key in day_keys:
        day_data = data.get(day_key, {})
        financial = day_data.get('financial', {})

        for agent, info in financial.items():
            if isinstance(info, dict):
                metrics['agent_total_income'][agent] += info.get('daily_income', 0)
                metrics['agent_total_expenses'][agent] += info.get('daily_expenses', 0)

    metrics['agent_total_income'] = dict(metrics['agent_total_income'])
    metrics['agent_total_expenses'] = dict(metrics['agent_total_expenses'])

    return metrics


def analyze_runs(manifest: Dict[str, Any], records_base_path: str) -> Dict[str, Any]:
    """
    Analyze all runs from the manifest and compute variance.

    Args:
        manifest: The run manifest dictionary
        records_base_path: Base path to LLMAgentsTown_memory_records

    Returns:
        Aggregated analysis dictionary
    """
    runs = manifest.get('runs', [])

    # Collect metrics from all runs
    all_economic = []
    all_behavior = []
    all_financial = []

    for run in runs:
        if run.get('status') != 'completed':
            continue

        timestamp = run.get('timestamp')
        if not timestamp:
            continue

        # Build file paths
        metrics_path = os.path.join(
            records_base_path, 'simulation_metrics',
            f'metrics_log_{timestamp}.jsonl'
        )
        memories_path = os.path.join(
            records_base_path, 'simulation_agents',
            f'agents_memories_{timestamp}.jsonl'
        )
        summary_path = os.path.join(
            records_base_path, 'simulation_daily_summaries',
            f'daily_summary_{timestamp}.json'
        )

        # Parse each file
        economic = parse_metrics_log(metrics_path)
        behavior = parse_agents_memories(memories_path)
        financial = parse_daily_summary(summary_path)

        all_economic.append(economic)
        all_behavior.append(behavior)
        all_financial.append(financial)

    # Compute variance analysis
    analysis = {
        'total_runs': len(runs),
        'completed_runs': len(all_economic),
        'failed_runs': len(runs) - len(all_economic),
        'economic_variance': {},
        'behavior_variance': {},
        'per_agent_variance': {},
        'per_shop_variance': {},
        'outlier_runs': []
    }

    if not all_economic:
        return analysis

    # Economic variance
    analysis['economic_variance'] = {
        'total_revenue': compute_statistics([e['total_revenue'] for e in all_economic]),
        'transaction_count': compute_statistics([e['transaction_count'] for e in all_economic]),
        'discount_transactions': compute_statistics([e['discount_transactions'] for e in all_economic]),
        'discount_revenue': compute_statistics([e['discount_revenue'] for e in all_economic]),
    }

    # Revenue by shop variance
    shops = set()
    for e in all_economic:
        shops.update(e['revenue_by_shop'].keys())

    for shop in shops:
        values = [e['revenue_by_shop'].get(shop, 0) for e in all_economic]
        analysis['per_shop_variance'][shop] = compute_statistics(values)

    # Behavior variance
    analysis['behavior_variance'] = {
        'total_travel_steps': compute_statistics([b['total_travel_steps'] for b in all_behavior]),
        'replan_count': compute_statistics([b['replan_count'] for b in all_behavior]),
        'conversation_count': compute_statistics([b['conversation_count'] for b in all_behavior]),
        'memory_count': compute_statistics([b['memory_count'] for b in all_behavior]),
    }

    # Per-agent variance
    agents = set()
    for f in all_financial:
        agents.update(f['agent_final_money'].keys())

    for agent in agents:
        agent_stats = {}

        # Final money
        values = [f['agent_final_money'].get(agent, 0) for f in all_financial]
        agent_stats['final_money'] = compute_statistics(values)

        # Total income
        values = [f['agent_total_income'].get(agent, 0) for f in all_financial]
        agent_stats['total_income'] = compute_statistics(values)

        # Total expenses
        values = [f['agent_total_expenses'].get(agent, 0) for f in all_financial]
        agent_stats['total_expenses'] = compute_statistics(values)

        # Travel steps
        values = [b['travel_by_agent'].get(agent, 0) for b in all_behavior]
        agent_stats['travel_steps'] = compute_statistics(values)

        # Replans
        values = [b['replans_by_agent'].get(agent, 0) for b in all_behavior]
        agent_stats['replans'] = compute_statistics(values)

        analysis['per_agent_variance'][agent] = agent_stats

    # Identify outlier runs (>2 standard deviations from mean)
    revenue_stats = analysis['economic_variance']['total_revenue']
    mean_revenue = revenue_stats['mean']
    std_revenue = revenue_stats['std_dev']

    for i, e in enumerate(all_economic):
        if std_revenue > 0:
            z_score = abs(e['total_revenue'] - mean_revenue) / std_revenue
            if z_score > 2:
                analysis['outlier_runs'].append({
                    'run_index': i + 1,
                    'metric': 'total_revenue',
                    'value': e['total_revenue'],
                    'z_score': round(z_score, 2)
                })

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Analyze variance across stability test runs'
    )
    parser.add_argument(
        '--manifest', type=str,
        default=None,
        help='Path to run_manifest.json'
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'stability_test_results')
    records_dir = os.path.join(script_dir, '..', 'LLMAgentsTown_memory_records')

    manifest_path = args.manifest or os.path.join(results_dir, 'run_manifest.json')

    # Check manifest exists
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run stability_test_runner.py first to generate runs.")
        sys.exit(1)

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print("\n" + "=" * 60)
    print("  STABILITY VARIANCE ANALYSIS")
    print("=" * 60)
    print(f"\nManifest: {manifest_path}")
    print(f"Total runs planned: {manifest.get('total_runs_planned', 'N/A')}")
    print(f"Runs in manifest: {len(manifest.get('runs', []))}")

    # Analyze runs
    analysis = analyze_runs(manifest, records_dir)

    # Save analysis
    output_path = os.path.join(results_dir, 'aggregated_analysis.json')
    os.makedirs(results_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to: {output_path}")

    # Print summary
    print("\n" + "-" * 60)
    print("  SUMMARY")
    print("-" * 60)
    print(f"\nCompleted runs: {analysis['completed_runs']}/{analysis['total_runs']}")
    print(f"Failed runs: {analysis['failed_runs']}")

    print("\n--- Economic Metrics ---")
    for metric, stats in analysis['economic_variance'].items():
        print(f"  {metric}:")
        print(f"    Mean: {stats['mean']}, Std: {stats['std_dev']}, CV: {stats['cv']}")

    print("\n--- Behavior Metrics ---")
    for metric, stats in analysis['behavior_variance'].items():
        print(f"  {metric}:")
        print(f"    Mean: {stats['mean']}, Std: {stats['std_dev']}, CV: {stats['cv']}")

    print("\n--- Shop Revenue ---")
    for shop, stats in analysis['per_shop_variance'].items():
        print(f"  {shop}: Mean={stats['mean']}, CV={stats['cv']}")

    if analysis['outlier_runs']:
        print("\n--- Outlier Runs ---")
        for outlier in analysis['outlier_runs']:
            print(f"  Run {outlier['run_index']}: {outlier['metric']}={outlier['value']} (z={outlier['z_score']})")
    else:
        print("\n--- No Outlier Runs Detected ---")

    # Stability assessment
    print("\n" + "=" * 60)
    print("  STABILITY ASSESSMENT")
    print("=" * 60)

    issues = []
    for metric, stats in analysis['economic_variance'].items():
        if stats['cv'] > 0.15:
            issues.append(f"  - {metric}: CV={stats['cv']} > 0.15 (variable)")

    for metric, stats in analysis['behavior_variance'].items():
        if stats['cv'] > 0.30:
            issues.append(f"  - {metric}: CV={stats['cv']} > 0.30 (unstable)")

    if issues:
        print("\nPotential stability issues:")
        for issue in issues:
            print(issue)
    else:
        print("\nAll metrics within acceptable variance thresholds.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
