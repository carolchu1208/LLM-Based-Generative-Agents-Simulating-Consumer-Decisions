#!/usr/bin/env python
# coding: utf-8

"""
Debug script to run simulation with saved plans from previous runs.

This script allows you to:
1. Skip LLM plan generation (saves time)
2. Use exact same plans from your last run 
3. Test your updated functions quickly
4. See if your fixes work with the same scenarios

Usage:
1. Run normal simulation until you want to interrupt (Ctrl+C)
2. Plans will be automatically saved via MemoryManager
3. Run this script to reuse those plans and test your changes

Note: All plan management is now handled by MemoryManager for consistency
with other data persistence (memories, conversations, etc.)
"""

import os
import sys
from town_main_simulation import TownSimulation, TimeManager

def main():
    """Run simulation with saved plans for debugging."""
    try:
        # Get configuration path
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Stability_Agents_Config_Test.json"
        )
        
        # Verify config file exists
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)
            
        print("🚀 Starting simulation with plan reuse for debugging...")
        print("This will automatically use the LATEST saved plans from your previous run.")
        print()
        
        # Initialize simulation
        simulation = TownSimulation(config_path)
        
        # 🔑 ENABLE PLAN REUSE FOR DEBUGGING
        print("Enabling plan reuse and checking for saved plans...")
        simulation.set_plan_reuse(True)
        
        # Set initial time to 7:00 on day 1
        TimeManager.set_time(7)
        TimeManager.set_current_day(1)
        
        print(f"\nStarting simulation at Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
        print("Will use saved plans - no waiting for LLM responses!")
        
        # Run the simulation
        simulation.run_simulation()
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 