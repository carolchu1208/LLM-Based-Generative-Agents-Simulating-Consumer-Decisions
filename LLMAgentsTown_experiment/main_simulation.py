#!/usr/bin/env python
# coding: utf-8

"""
Main simulation controller for the town simulation.

This file orchestrates the town simulation by:
1. Initializing all necessary components
2. Managing the simulation flow
3. Coordinating between different managers
4. Handling atomic operations
5. Managing simulation state

File Dependencies:
1. agent_configuration.json: Town and agent configuration
2. town_main_simulation.py: Core simulation components
3. Stability_Memory_Manager.py: Memory management
4. Stability_Metrics_Manager.py: Metrics tracking
5. stability_classes.py: Core classes (Agent, Location, etc.)
6. prompt_manager.py: Prompt templates
7. deepseek_model_manager.py: LLM interaction
8. energy_constants.py: Energy system constants
"""

import os
import sys
import json
import threading
import time
import traceback
import logging
import signal

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Optional, Any, Union
import re

# Local imports
from memory_manager import MemoryManager
from metrics_manager import StabilityMetricsManager
from simulation_execution_classes import (
    Agent, Location, TownMap, PlanExecutor,
    ConversationManager, SimulationSettings
)
from simulation_types import TimeManager
from simulation_constants import (
    ENERGY_DECAY_PER_HOUR, ENERGY_THRESHOLD_LOW,
    ENERGY_MAX, ENERGY_MIN, ENERGY_COST_WORK_HOUR, ENERGY_COST_PER_STEP,
    ENERGY_GAIN_RESTAURANT_MEAL, ENERGY_GAIN_SNACK, ENERGY_GAIN_HOME_MEAL, ENERGY_GAIN_NAP
)
from prompt_manager import PromptManager
from llm_deepseek_manager import ModelManager
from shared_trackers import LocationLockManager, SharedLocationTracker

class TownSimulation:
    """Main town simulation controller."""
    
    def __init__(self, config_path: str):
        """Initialize the town simulation."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize components
        self.memory_mgr = MemoryManager()
        self.metrics_mgr = StabilityMetricsManager(memory_manager=self.memory_mgr)
        self.location_lock_mgr = LocationLockManager()
        self.location_tracker = SharedLocationTracker()
        # ConversationManager will be initialized after locations are available
        
        # Initialize prompt and model managers
        self.prompt_mgr = PromptManager(location_tracker=self.location_tracker, config_data=self.config)
        self.model_mgr = ModelManager()
        
        # Initialize town map first
        self.town_map = TownMap(
            world_locations_data=self.config.get('town_map_grid', {}).get('world_locations', {}),
            travel_paths_data=self.config.get('town_map_grid', {}).get('travel_paths', [])
        )
        
        # Initialize locations and agents
        self.locations = self._initialize_locations()
        self.agents = self._initialize_agents()
        
        # Initialize ConversationManager now that locations are available
        self.conversation_mgr = ConversationManager(self.memory_mgr, self.locations)
        
        # Update metrics manager with agents after initialization
        self.metrics_mgr = StabilityMetricsManager(memory_manager=self.memory_mgr, agents=self.agents)
        
        # Initialize plan executor
        self.plan_executor = PlanExecutor(
            location_lock_mgr=self.location_lock_mgr,
            conversation_mgr=self.conversation_mgr,
            memory_mgr=self.memory_mgr,
            metrics_mgr=self.metrics_mgr,
            location_tracker=self.location_tracker,
            agents=self.agents,
            locations=self.locations,
            prompt_mgr=self.prompt_mgr,
            model_mgr=self.model_mgr,
            config_data=self.config
        )
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
        
        # Initialize simulation state
        self.running = True
        self.paused = False
        
        # Reset time to simulation start
        TimeManager.reset_time()
        
        # Set simulation reference for location tracker
        self.location_tracker.set_simulation(self)
        
        print("\nInitializing simulation components...")
        print(f"Total locations: {len(self.locations)}")
        print(f"Total agents: {len(self.agents)}")
        print(f"Simulation start: Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
        print(f"Simulation end: Day 7, Hour 24")
        print("\nInitialization complete!")
        
        # Initialize plan reuse system
        self.reuse_plans = False  # Set to True to skip LLM planning

    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            os.makedirs('logs', exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/town_simulation.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("Logging initialized")
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            traceback.print_exc()

    def _check_simulation_end(self):
        """Check if simulation should end."""
        print(f"[DEBUG] _check_simulation_end() called")
        
        current_day = TimeManager.get_current_day()
        current_hour = TimeManager.get_current_hour()
        end_day = 8  # End at Day 8, Hour 0 (cleaner than Day 7, Hour 24)
        end_hour = 0
        
        print(f"[DEBUG] Current day: {current_day}, target end day: {end_day}")
        print(f"[DEBUG] Current hour: {current_hour}, target end hour: {end_hour}")
        
        # Check if we've reached Day 8, Hour 0
        if current_day >= end_day and current_hour >= end_hour:
            print(f"[DEBUG] Simulation end condition met: Day {current_day} >= {end_day}, Hour {current_hour} >= {end_hour}")
            return True
        
        print(f"[DEBUG] Simulation end condition not met yet")
        return False

    def run_simulation(self):
        """Run the main simulation loop."""
        try:
            print("\nStarting simulation...")
            print(f"Initial time: Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
            
            print("[DEBUG] About to check simulation end condition...")
            end_condition = self._check_simulation_end()
            print(f"[DEBUG] Simulation end condition: {end_condition}")
            print(f"[DEBUG] Running state: {self.running}")
            print(f"[DEBUG] Paused state: {self.paused}")
            
            print("[DEBUG] About to enter main simulation loop...")
            
            while self.running:
                # Check simulation end condition once per iteration
                simulation_end = self._check_simulation_end()
                print(f"[DEBUG] Simulation loop iteration - running: {self.running}, paused: {self.paused}, simulation_end: {simulation_end}")
                
                if simulation_end:
                    print("[DEBUG] Simulation end condition met, breaking loop")
                    break
                
                if not self.paused:
                    print(f"[DEBUG] Calling _process_hour() for Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
                    self._process_hour()
                    
                    # Print progress every hour
                    print(f"\nTime: Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
                    
                    # Process end of hour
                    print(f"[DEBUG] Calling _process_hour_end() for Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
                    self._process_hour_end(TimeManager.get_current_day(), TimeManager.get_current_hour())
                    
                    # Store current day before advancing time to detect day transitions
                    day_before_advance = TimeManager.get_current_day()
                    
                    # Advance time after processing current hour
                    print(f"[DEBUG] Advancing time from Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
                    TimeManager.advance_time(1)
                    print(f"[DEBUG] Time advanced to Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
                    
                    # Check if we've transitioned to a new day and trigger end-of-day processing for the previous day
                    day_after_advance = TimeManager.get_current_day()
                    if day_after_advance > day_before_advance:
                        print(f"[DEBUG] Day transition detected: Day {day_before_advance} → Day {day_after_advance}")
                        print(f"[DEBUG] Calling _process_end_of_day() for completed Day {day_before_advance}")
                        # Temporarily set day back to process end-of-day for the completed day
                        original_day = TimeManager.get_current_day()
                        TimeManager.set_current_day(day_before_advance)
                        self._process_end_of_day()
                        TimeManager.set_current_day(original_day)
                else:
                    print("[DEBUG] Simulation is paused, skipping hour processing")
                
                time.sleep(1)  # Small delay to prevent CPU overuse
            
            print("\nSimulation completed!")
            self._cleanup()
            
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            self._cleanup()
        except Exception as e:
            print(f"\nError in simulation: {str(e)}")
            traceback.print_exc()
            self._cleanup()

    def _process_hour(self):
        """Process the current hour for all agents."""
        print(f"[DEBUG] _process_hour() started for Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
        
        current_day = TimeManager.get_current_day()
        current_hour = TimeManager.get_current_hour()
        
        try:
            # Create daily plans if it's 7:00
            if current_hour == 7:
                print(f"[DEBUG] Creating daily plans for Day {current_day}")
                
                # Clear previous day's plans first
                print(f"[DEBUG] Clearing previous day's plans for all agents")
                for agent_name, agent in self.agents.items():
                    agent.daily_plan = None
                    print(f"[DEBUG] Cleared plan for {agent_name}")
                
                # Reset daily financial tracking for new day
                print(f"[DEBUG] Resetting daily financial tracking for all agents")
                for agent_name, agent in self.agents.items():
                    agent.daily_income = 0.0
                    agent.daily_expenses = 0.0
                    print(f"[DEBUG] Reset daily financials for {agent_name}")
                
                # Generate new plans for the current day
                self._create_daily_plans(current_hour, current_day)
            
            # PRE-HOUR CHECK: Check for emergency conditions BEFORE executing plans
            # This prevents agents from executing interrupted travel when they need immediate food
            print(f"[DEBUG] Pre-hour emergency check for {len(self.agents)} agents")
            for agent_name, agent in self.agents.items():
                try:
                    current_energy = agent.energy_system.get_energy(agent.name)
                    
                    # CRITICAL: Check for zero energy first - force to residence immediately
                    if current_energy <= 0:
                        print(f"[CRITICAL] {agent_name} has {current_energy} energy - forcing to residence for recovery!")
                        
                        # Clear any interrupted travel and plans
                        if hasattr(agent, 'interrupted_travel'):
                            agent.interrupted_travel = None
                        agent.daily_plan = None
                        
                        # Force agent to residence immediately
                        residence_coords = agent.town_map.get_coordinates_for_location(agent.residence)
                        if residence_coords:
                            self.location_tracker.update_agent_position(
                                agent.name, 
                                agent.residence, 
                                residence_coords, 
                                TimeManager.get_current_hour()
                            )
                            print(f"[CRITICAL] {agent_name} forced to {agent.residence} for energy recovery")
                        else:
                            print(f"[ERROR] Could not find coordinates for {agent.residence}")
                        
                        # Skip further processing for this agent this hour
                        continue
                    
                    # Check if agent has critically low energy AND has interrupted travel
                    if (current_energy <= ENERGY_THRESHOLD_LOW and 
                        hasattr(agent, 'interrupted_travel') and 
                        agent.interrupted_travel):
                        
                        print(f"[WARNING] {agent_name} has critically low energy ({current_energy}) AND interrupted travel - clearing interrupted travel for emergency!")
                        
                        # Clear interrupted travel immediately to prevent old travel from executing
                        agent.clear_interrupted_travel()
                        
                        # Trigger emergency replan right now
                        emergency_context = {
                            'error': f'Critical energy level ({current_energy}/{ENERGY_MAX}) with interrupted travel. Immediate food needed.',
                            'replan_needed': True,
                            'reason_code': 'CRITICAL_ENERGY_WITH_INTERRUPTED_TRAVEL'
                        }
                        self._handle_emergency_replan(agent, emergency_context)
                        print(f"[INFO] Pre-hour emergency replan completed for {agent_name} - interrupted travel cleared and new plan generated.")
                        
                except Exception as e:
                    print(f"[ERROR] Error in pre-hour emergency check for {agent_name}: {str(e)}")
                    traceback.print_exc()
            
            print(f"[DEBUG] Processing {len(self.agents)} agents")
            
            # Process each agent
            for agent_name, agent in self.agents.items():
                try:
                    print(f"[DEBUG] Processing agent: {agent_name}")
                    
                    # Check if this is sleep time (23-6) - handle automatic sleep regardless of plan
                    if current_hour == 23 or (0 <= current_hour <= 6):
                        print(f"[DEBUG] Sleep hour detected ({current_hour}) - handling automatic sleep for {agent_name}")
                        try:
                            # Execute sleep logic through PlanExecutor (passes None as plan to trigger sleep)
                            success, result = self.plan_executor.execute(agent, None, current_hour)
                            if success:
                                print(f"[DEBUG] Automatic sleep handled for {agent_name}")
                            else:
                                print(f"[DEBUG] Sleep handling failed for {agent_name}: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            print(f"[ERROR] Error handling sleep for {agent_name}: {str(e)}")
                            traceback.print_exc()
                    # Execute agent's plan for current time if available (non-sleep hours)
                    elif agent.daily_plan:
                        print(f"[DEBUG] Executing plan for {agent_name}")
                        try:
                            # Execute the plan through PlanExecutor
                            success, result = self.plan_executor.execute(agent, agent.daily_plan, current_hour)
                            
                            # Check for a replan signal on failure
                            if not success and result.get('replan_needed'):
                                print(f"[INFO] Plan failed for {agent_name}. Reason: {result.get('error')}. Triggering emergency replan.")
                                self._handle_emergency_replan(agent, result)
                                
                                # Retry execution with the newly updated plan
                                print(f"[INFO] Retrying execution for {agent_name} with updated plan.")
                                success, result = self.plan_executor.execute(agent, agent.daily_plan, current_hour)

                            if success:
                                print(f"[DEBUG] Successfully executed plan for {agent_name}")
                            else:
                                print(f"[DEBUG] Failed to execute plan for {agent_name}. Final error: {result.get('error')}")

                        except Exception as e:
                            print(f"[ERROR] Error executing plan for {agent_name}: {str(e)}")
                            traceback.print_exc()
                    else:
                        print(f"[DEBUG] No plan available for {agent_name} during non-sleep hours")
                        print(f"[DEBUG] Agent {agent_name} daily_plan attribute: {getattr(agent, 'daily_plan', 'NOT_SET')}")
                    
                    print(f"[DEBUG] Agent {agent_name} processed successfully")
                    
                    # Energy decay is applied in _process_hour_end() to avoid duplication
                    
                except Exception as e:
                    print(f"[ERROR] Unhandled error processing agent {agent_name}: {str(e)}")
                    traceback.print_exc()
            
            print(f"[DEBUG] _process_hour() completed for Day {current_day}, Hour {current_hour}")
            
        except Exception as e:
            print(f"[ERROR] Unhandled error in _process_hour: {str(e)}")
            traceback.print_exc()

    def _handle_emergency_replan(self, agent: 'Agent', failure_context: Dict[str, Any]):
        """Handles the logic for an agent to reactively replan after a failure."""
        try:
            current_time = TimeManager.get_current_hour()
            current_day = TimeManager.get_current_day()

            # Step 1: Find a viable alternative location
            target_location = None
            for loc_name, location in self.locations.items():
                if location.location_type in ['restaurant', 'local_shop'] and location.is_open(current_time):
                    meal_cost = location.prices.get('meal', 20.0)
                    if agent.can_afford_purchase(meal_cost):
                        target_location = location # Found a valid, affordable, open restaurant
                        break
            
            if not target_location:
                print(f"[WARNING] {agent.name} could not find any open and affordable restaurant to fix failed plan.")
                return

            # Step 2: Determine emergency plan timing - avoid sleep hours
            replan_start_hour = current_time
            replan_end_hour = current_time + 1
            
            # Check if current or next hour is sleep time (23-6)
            def is_sleep_hour(hour):
                return hour == 23 or (0 <= hour <= 6)
            
            if is_sleep_hour(replan_start_hour) or is_sleep_hour(replan_end_hour):
                print(f"[INFO] Emergency replan during sleep hours detected. Current: {current_time}, Next: {replan_end_hour}")
                
                # If we're at hour 22, plan for hours 21-22 (go back 1 hour)
                if current_time == 22:
                    replan_start_hour = 21
                    replan_end_hour = 22
                    print(f"[INFO] Adjusted emergency replan to hours {replan_start_hour}-{replan_end_hour} to avoid sleep")
                # If we're in sleep hours (23-6), this shouldn't happen as sleep should override
                # But if it does, we'll skip the emergency replan since sleep provides energy
                else:
                    print(f"[INFO] Skipping emergency replan during sleep hours - agent will recover energy through sleep")
                    return

            # Step 3: Get the emergency prompt from the PromptManager
            # Get current location from agent position
            agent_position = self.location_tracker.get_agent_position(agent.name)
            current_location = agent_position[0] if agent_position else agent.residence
            
            prompt_context = {
                'name': agent.name,
                'current_time': replan_start_hour,  # Use adjusted timing
                'current_day': current_day,
                'current_location': current_location,
                'money': agent.money,
                'reason': failure_context.get('error', 'reason unknown'),
                'target_location': target_location.name
            }
            emergency_prompt = self.prompt_mgr.get_emergency_replan_prompt(prompt_context)

            # Step 4: Call the LLM to generate the 2-hour micro-plan
            print(f"[INFO] Generating emergency plan for {agent.name} to go to {target_location.name} (hours {replan_start_hour}-{replan_end_hour}).")
            raw_plan_response = self.model_mgr.generate(emergency_prompt, "daily_plan")
            
            # Step 5: Extract and clean the plan
            cleaned_plan = self.memory_mgr.extract_and_clean_plan(raw_plan_response, "emergency_replan")
            if cleaned_plan['status'] != 'success' or not cleaned_plan.get('activities'):
                print(f"[ERROR] Failed to extract a valid emergency plan for {agent.name} from LLM response.")
                return

            # Step 6: Update the agent's main daily plan with the new micro-plan
            agent.update_plan_slice(cleaned_plan['activities'], start_hour=replan_start_hour)
            
            # Clear any interrupted travel state since agent has a new emergency plan
            if hasattr(agent, 'clear_interrupted_travel'):
                agent.clear_interrupted_travel()
                print(f"[INFO] Cleared interrupted travel state for {agent.name} due to emergency replan.")
            
            print(f"[INFO] Successfully updated {agent.name}'s plan with emergency actions for hours {replan_start_hour}-{replan_end_hour}.")

        except Exception as e:
            print(f"[ERROR] Critical error during emergency replan for {agent.name}: {str(e)}")
            traceback.print_exc()

    def _process_hour_end(self, current_day: int, current_hour: int):
        """Process end of hour events."""
        print(f"[DEBUG] _process_hour_end() started for Day {current_day}, Hour {current_hour}")
        
        try:
            # Apply natural energy decay to all agents
            print(f"Applying natural energy decay ({ENERGY_DECAY_PER_HOUR} energy) to all agents")
            for agent_name, agent in self.agents.items():
                try:
                    # Get energy before decay for detailed logging
                    energy_before = agent.energy_system.get_energy(agent.name)
                    
                    # Apply natural decay
                    result = agent.energy_system.update_energy(agent.name, -ENERGY_DECAY_PER_HOUR)
                    if result.success:
                        energy_after = agent.energy_system.get_energy(agent.name)
                        print(f"  {agent_name}: {energy_before} → {energy_after} (natural decay -{ENERGY_DECAY_PER_HOUR})")
                        
                        # Log energy history for debugging
                        history_result = agent.energy_system.get_energy_history(agent.name)
                        if history_result.success and len(history_result.value) >= 2:
                            recent_changes = history_result.value[-3:]  # Last 3 changes
                            print(f"    Recent energy changes for {agent_name}:")
                            for change in recent_changes:
                                change_amount = change.get('change', 0)
                                new_level = change.get('new_level', 0)
                                print(f"      {change_amount:+} → {new_level}")
                    else:
                        print(f"  {agent_name}: Failed to apply energy decay - {result.error}")
                except Exception as e:
                    print(f"  {agent_name}: Error applying energy decay - {str(e)}")
            
            # Check energy thresholds after all state updates are complete
            print(f"Checking energy thresholds for emergency intervention...")
            for agent_name, agent in self.agents.items():
                try:
                    current_energy = agent.energy_system.get_energy(agent.name)
                    
                    # CRITICAL: Check for zero energy first - force to residence immediately
                    if current_energy <= 0:
                        print(f"[CRITICAL] {agent_name} has {current_energy} energy - forcing to residence for recovery!")
                        
                        # Clear any interrupted travel and plans
                        if hasattr(agent, 'interrupted_travel'):
                            agent.interrupted_travel = None
                        agent.daily_plan = None
                        
                        # Force agent to residence immediately
                        residence_coords = agent.town_map.get_coordinates_for_location(agent.residence)
                        if residence_coords:
                            self.location_tracker.update_agent_position(
                                agent.name, 
                                agent.residence, 
                                residence_coords, 
                                TimeManager.get_current_hour()
                            )
                            print(f"[CRITICAL] {agent_name} forced to {agent.residence} for energy recovery")
                        else:
                            print(f"[ERROR] Could not find coordinates for {agent.residence}")
                        
                        # Skip further processing for this agent
                        continue
                    
                    # Check if agent is at critically low energy levels
                    if current_energy <= ENERGY_THRESHOLD_LOW:
                        print(f"[WARNING] {agent_name} has critically low energy ({current_energy}) - triggering emergency food replan!")
                        emergency_context = {
                            'error': f'Critical energy level ({current_energy}/{ENERGY_MAX}). Immediate food needed.',
                            'replan_needed': True,
                            'reason_code': 'CRITICAL_ENERGY_LOW'
                        }
                        self._handle_emergency_replan(agent, emergency_context)
                        print(f"[INFO] Emergency replan completed for {agent_name} due to low energy.")
                    
                    elif current_energy <= ENERGY_THRESHOLD_LOW:
                        # Check if agent has a meal planned in the next 2 hours
                        has_upcoming_meal = False
                        if agent.daily_plan and 'activities' in agent.daily_plan:
                            for activity in agent.daily_plan['activities']:
                                activity_hour = activity.get('time', -1)
                                activity_action = activity.get('action', '')
                                # Check if there's an eating activity in the next 2 hours
                                if (current_hour <= activity_hour <= current_hour + 2 and 
                                    activity_action == 'eat'):
                                    has_upcoming_meal = True
                                    break
                        
                        if not has_upcoming_meal:
                            print(f"[WARNING] {agent_name} has low energy ({current_energy}) and no upcoming meal - triggering preventive food replan!")
                            emergency_context = {
                                'error': f'Low energy level ({current_energy}/{ENERGY_MAX}). Food needed soon.',
                                'replan_needed': True,
                                'reason_code': 'PREVENTIVE_ENERGY_LOW'
                            }
                            self._handle_emergency_replan(agent, emergency_context)
                            print(f"[INFO] Preventive replan completed for {agent_name} due to low energy without upcoming meal.")
                    
                except Exception as e:
                    print(f"[ERROR] Error checking energy thresholds for {agent_name}: {str(e)}")
                    traceback.print_exc()
            
            # Perform coordinated save check
            self._coordinated_save_check(current_day, current_hour)
                
        except Exception as e:
            print(f"Error in _process_hour_end: {str(e)}")
            traceback.print_exc()

    def _coordinated_save_check(self, current_day: int, current_hour: int):
        """Simplified unified saving strategy: Save all data types together."""
        try:
            # Save everything every hour (unified approach)
            print(f"[SAVE] Hourly unified save for all data types...")
            
            # Save memories (with buffer check)
            if len(self.memory_mgr._pending_memories) >= 200:
                print(f"[SAVE] Memory buffer full ({len(self.memory_mgr._pending_memories)} items)")
            self.memory_mgr.save_memories(force=True)
            
            # Save metrics
            self.metrics_mgr.save_metrics()
            
            # Save conversation logs
            self.conversation_mgr.save_conversation_logs()
            
            print(f"[SAVE] ✅ All data types saved successfully")
            
        except Exception as e:
            print(f"Error in coordinated save check: {str(e)}")
            traceback.print_exc()

    def _process_end_of_day(self):
        """Process end of day events."""
        try:
            current_day = TimeManager.get_current_day()
            
            print(f"\n=== End of Day {current_day} Processing ===")
            
            # 1. Generate and save daily summaries
            # Hourly saves have already persisted memory, metrics, and conversation logs.
            print("[SAVE] Generating and saving daily summaries...")
            self.memory_mgr.save_daily_summaries(self.metrics_mgr, current_day)
            
            # 2. Clear buffers for next day
            print("[CLEANUP] Clearing buffers for next day...")
            self.memory_mgr._pending_memories.clear()
            self.metrics_mgr.clear_daily_metrics()
            
            print(f"\nEnd of Day {current_day}")
            print("✅ Daily summaries saved successfully")
            print("✅ Buffers cleared for next day")
            
        except Exception as e:
            print(f"Error in _process_end_of_day: {str(e)}")
            traceback.print_exc()

    def handle_interrupt(self, signum, frame):
        """Handle simulation interruption."""
        if not self.running:
            print("\nSimulation already shutting down...")
            sys.exit(0)  # Force exit if already shutting down
        
        try:
            print("\nSimulation interrupted. Saving state...")
            self.running = False
            
            # Save daily plans for debugging reuse
            print("Saving daily plans for debugging...")
            try:
                self.memory_mgr.save_current_plans(self.agents)
            except Exception as e:
                print(f"Error saving daily plans: {str(e)}")
            
            # Cancel any pending futures and shutdown executor
            if hasattr(self, '_executor'):
                print("Shutting down thread pool executor...")
                self._executor.shutdown(wait=False, cancel_futures=True)
            
            # Release all locks from LocationLockManager
            if hasattr(self, 'location_lock_mgr'):
                print("Releasing location locks...")
                self.location_lock_mgr.release_all_locks()
            
            # Force save all memories through MemoryManager
            if hasattr(self, 'memory_mgr'):
                print("Saving memories...")
                try:
                    self.memory_mgr.force_save_memories()
                except Exception as e:
                    print(f"Error saving memories: {str(e)}")
            
            # Save final metrics and daily summaries
            if hasattr(self, 'metrics_mgr'):
                print("Saving metrics and daily summaries...")
                try:
                    daily_summary = self.metrics_mgr.generate_daily_summary(TimeManager.get_current_day())
                    self.metrics_mgr.store_daily_summary(TimeManager.get_current_day(), daily_summary)
                    self.metrics_mgr.save_final_metrics()
                except Exception as e:
                    print(f"Error saving metrics: {str(e)}")
            
            # Save conversation logs
            if hasattr(self, 'conversation_mgr'):
                print("Saving conversation logs...")
                try:
                    self.conversation_mgr.save_conversation_logs()
                except Exception as e:
                    print(f"Error saving conversation logs: {str(e)}")
            
            print("\n" + "="*50)
            print("🚀 DEBUGGING TIP:")
            print("To reuse these plans in your next run, add this line after creating TownSimulation:")
            print("simulation.set_plan_reuse(True)")
            print("="*50)
            
            print("Cleanup completed. Exiting...")
            sys.exit(0)  # Force exit after cleanup
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            traceback.print_exc()
            # Force exit even if cleanup fails
            sys.exit(1)

    def _save_simulation_state(self):
        """Save the current simulation state."""
        try:
            # Save daily summaries using metrics manager
            self.metrics_mgr.store_daily_summary(TimeManager.get_current_day(), self.metrics_mgr.generate_daily_summary(TimeManager.get_current_day()))
            
            # Save other state
            self.memory_mgr.save_memories(force=True)
            self.metrics_mgr.save_metrics(force=True)
            
        except Exception as e:
            print(f"Error saving simulation state: {str(e)}")
            traceback.print_exc()

    def _cleanup(self):
        """Clean up resources before exiting."""
        try:
            # Save daily plans for debugging reuse
            self.memory_mgr.save_current_plans(self.agents)
            
            # Save final daily summaries using metrics manager
            self.metrics_mgr.store_daily_summary(TimeManager.get_current_day(), self.metrics_mgr.generate_daily_summary(TimeManager.get_current_day()))
            
            # Save final state
            self.memory_mgr.save_memories(force=True)
            self.metrics_mgr.save_final_metrics()
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            traceback.print_exc()

    def _save_plans_for_debugging(self, current_day: int, current_hour: int):
        """Save all agents' daily plans for debugging reuse."""
        try:
            # Collect current plans
            current_plans = {}
            for agent_name, agent in self.agents.items():
                if agent.daily_plan:
                    current_plans[agent_name] = agent.daily_plan

            if not current_plans:
                print("[SAVE] No plans to save")
                return

            # Delegate to memory_manager for actual saving
            self.memory_mgr.save_daily_plans(current_day, current_hour, current_plans)

        except Exception as e:
            print(f"[ERROR] Error saving plans for debugging: {str(e)}")
            traceback.print_exc()



    def _get_latest_saved_plans(self) -> Optional[tuple]:
        """Find and load the most recent saved plans from consolidated file."""
        try:
            if not os.path.exists(self.saved_plans_file):
                print(f"[LOAD] No saved plans file found at {self.saved_plans_file}")
                return None
            
            # Load the consolidated plans file
            with open(self.saved_plans_file, 'r') as f:
                all_saved_plans = json.load(f)
            
            if not all_saved_plans:
                print(f"[LOAD] No saved plans found in {self.saved_plans_file}")
                return None
            
            # Find the most recent save (highest day, then highest hour)
            latest_key = None
            latest_day = -1
            latest_hour = -1
            
            for save_key, save_data in all_saved_plans.items():
                day = save_data.get('day', 0)
                hour = save_data.get('hour', 0)
                
                if day > latest_day or (day == latest_day and hour > latest_hour):
                    latest_day = day
                    latest_hour = hour
                    latest_key = save_key
            
            if not latest_key:
                print(f"[LOAD] Could not find valid latest plans")
                return None
            
            latest_save = all_saved_plans[latest_key]
            plans_data = latest_save.get('plans', {})
            
            print(f"[LOAD] Found latest saved plans from Day {latest_day}, Hour {latest_hour}")
            print(f"[LOAD] Loaded plans for {len(plans_data)} agents")
            print(f"[LOAD] Available saves in file: {len(all_saved_plans)}")
            
            return latest_day, plans_data
            
        except Exception as e:
            print(f"Error finding latest saved plans: {str(e)}")
            traceback.print_exc()
            return None

    def set_plan_reuse(self, enable: bool = True):
        """Enable or disable plan reuse for debugging."""
        self.reuse_plans = enable
        if enable:
            print("[DEBUG] Plan reuse ENABLED - will load saved plans instead of generating new ones")
            self.memory_mgr.show_available_saved_plans()
        else:
            print("[DEBUG] Plan reuse DISABLED - will generate new plans normally")

    def _show_available_saved_plans(self):
        """Show what saved plans are available for debugging."""
        try:
            if not os.path.exists(self.saved_plans_file):
                print(f"[DEBUG] No saved plans file found at {self.saved_plans_file}")
                return
            
            # Get file size
            try:
                stat = os.stat(self.saved_plans_file)
                size_kb = stat.st_size / 1024
            except OSError:
                size_kb = 0
            
            # Load and show content
            with open(self.saved_plans_file, 'r') as f:
                all_saved_plans = json.load(f)
            
            if all_saved_plans:
                print(f"[DEBUG] Saved plans file: {self.saved_plans_file} ({size_kb:.1f} KB)")
                print(f"[DEBUG] Available saved sessions:")
                
                # Sort by day and hour
                sorted_saves = sorted(all_saved_plans.items(), 
                                    key=lambda x: (x[1]['day'], x[1]['hour']))
                
                for save_key, save_data in sorted_saves:
                    day = save_data.get('day', 0)
                    hour = save_data.get('hour', 0)
                    agent_count = len(save_data.get('plans', {}))
                    timestamp = save_data.get('timestamp', 'unknown')
                    
                    print(f"[DEBUG]   Day {day}, Hour {hour}: {agent_count} agents ({timestamp})")
            else:
                print(f"[DEBUG] No saved sessions found in {self.saved_plans_file}")
                
        except Exception as e:
            print(f"[DEBUG] Error checking saved plans: {str(e)}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            print(f"\nLoading configuration from: {config_path}")
            print(f"Current working directory: {os.getcwd()}")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            print("Reading config file...")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if not config:
                raise ValueError(f"Empty configuration file: {config_path}")
            
            # Validate required sections
            required_sections = ['town_people', 'town_areas', 'town_map_grid']
            missing_sections = [section for section in required_sections if section not in config]
            if missing_sections:
                raise ValueError(f"Missing required sections in config: {missing_sections}")
            
            print("\nConfiguration loaded successfully")
            print(f"Config keys: {list(config.keys())}")
            
            # Initialize SimulationSettings with the loaded config
            SimulationSettings.initialize(config)
            
            # Print config details
            for section in required_sections:
                if section in config:
                    print(f"Found section '{section}' in config")
                    if section == 'town_people':
                        print(f"Number of agents in config: {len(config[section])}")
                        print(f"Agent names: {list(config[section].keys())}")
            
            return config
            
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            traceback.print_exc()
            raise

    def _initialize_locations(self) -> Dict[str, Location]:
        """Initialize all locations in the town."""
        try:
            locations = {}
            town_areas = self.config.get('town_areas', {})
            
            # Iterate through each area type (retail_and_grocery, residences, dining, business)
            for area_type, area_locations in town_areas.items():
                for location_name, location_data in area_locations.items():
                    try:
                        # Get coordinates from town map
                        coordinates = self.town_map.get_coordinates_for_location(location_name)
                        if not coordinates:
                            print(f"Warning: No coordinates found for location {location_name}")
                            continue
                        
                        # Handle pricing based on area type
                        prices = {}
                        if area_type == 'dining':
                            # For dining locations, extract menu items and their prices
                            menu = location_data.get('menu', {})
                            for meal_type, meal_info in menu.items():
                                prices[meal_type] = meal_info.get('base_price', 15.0)
                        else:
                            # For non-dining locations, use base_price directly
                            base_price = location_data.get('base_price', 15.0)
                            prices['default'] = base_price
                        
                        # Create location instance
                        location = Location(
                            name=location_name,
                            location_type=location_data.get('type', 'unknown'),
                            coordinates=coordinates,
                            hours=location_data.get('hours'),
                            prices=prices,
                            discounts=location_data.get('discount')
                        )
                        
                        locations[location_name] = location
                        print(f"Initialized location: {location_name}")
                        
                    except Exception as e:
                        print(f"Error initializing location {location_name}: {str(e)}")
                        traceback.print_exc()
            
            return locations
            
        except Exception as e:
            print(f"Error in _initialize_locations: {str(e)}")
            traceback.print_exc()
            return {}

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all agents from configuration."""
        try:
            agents = {}
            
            # Get agent data from town_people section
            town_people = self.config.get('town_people', {})
            if not town_people:
                raise ValueError("No town_people found in configuration")
            
            # Initialize location tracker with valid locations
            self.location_tracker = SharedLocationTracker()
            print(f"[DEBUG] Setting valid locations: {list(self.locations.keys())}")
            self.location_tracker.set_valid_locations(set(self.locations.keys()))
            
            # Create agents
            for name, data in town_people.items():
                try:
                    print(f"\nInitializing agent: {name}")
                    
                    # Get basics data
                    basics = data.get('basics', {})
                    if not basics:
                        print(f"Error: No basics data found for agent {name}")
                        continue
                    
                    # Map income to income_info
                    if 'income' in basics:
                        basics['income_info'] = basics.pop('income')
                    
                    # Validate required fields
                    required_fields = ['age', 'income_info', 'residence', 'occupation']
                    missing_fields = [field for field in required_fields if field not in basics]
                    
                    if missing_fields:
                        print(f"Error: Missing required fields for agent {name}: {missing_fields}")
                        print("Current config:")
                        print(json.dumps(basics, indent=2))
                        print("Please ensure all required fields are present in the config file.")
                        continue

                    # Create agent instance
                    agent = Agent(
                        name=name,
                        config=basics,
                        memory_mgr=self.memory_mgr
                    )
                    
                    # Record the agent's initial state in memory.
                    self.memory_mgr.record_initial_state(agent)
                    
                    # Set the town map for the agent
                    agent.town_map = self.town_map
                    
                    # Assign location objects to agent
                    agent.locations = self.locations
                    
                    # Set current location to the actual location object
                    if agent.residence in self.locations:
                        agent.current_location = self.locations[agent.residence]
                    else:
                        raise ValueError(f"Residence {agent.residence} not found in locations")
                    
                    # Initialize location tracking
                    try:
                        residence_coords = self.town_map.get_coordinates_for_location(agent.residence)
                        if residence_coords is None:
                            raise ValueError(f"Could not find coordinates for residence: {agent.residence}")
                            
                        self.location_tracker.update_agent_position(
                            agent.name,
                            agent.residence,
                            residence_coords,
                            time.time()
                        )
                        print(f"Successfully updated location tracking for {name}")
                    except Exception as e:
                        print(f"Error updating location tracking for {name}: {str(e)}")
                        traceback.print_exc()
                    
                    # Add agent to dictionary
                    agents[name] = agent
                    print(f"Successfully initialized agent: {name}")
                    
                except Exception as e:
                    print(f"Error initializing agent {name}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            if not agents:
                print("WARNING: No agents were successfully initialized!")
                print("This could be due to:")
                print("1. Missing or invalid basics data in config")
                print("2. Errors during agent creation")
                print("3. Errors during location tracking initialization")
            else:
                print(f"\nSuccessfully initialized {len(agents)} agents")
                print(f"Agent names: {list(agents.keys())}")
            
            # Save all initial agent states to disk
            print("\nSaving initial memory state for all agents...")
            self.memory_mgr.save_memories(force=True)
            print("Memory initialization complete.")
            
            return agents
            
        except Exception as e:
            print(f"Error in agent initialization: {str(e)}")
            traceback.print_exc()
            raise

    def _create_daily_plans(self, current_hour: int, current_day: int):
        """Create daily plans for all agents or load saved plans for debugging."""
        print(f"[DEBUG] _create_daily_plans() started for Day {current_day}, Hour {current_hour}")
        
        try:
            # Check if plan reuse is enabled
            if self.reuse_plans:
                print(f"[DEBUG] Plan reuse enabled - searching for latest saved plans...")
                
                # Try to load the latest saved plans (any day)
                latest_plans_result = self.memory_mgr.get_latest_saved_plans()
                
                if latest_plans_result:
                    saved_day, saved_plans = latest_plans_result
                    print(f"[DEBUG] Using saved plans from Day {saved_day} for current Day {current_day}")
                    
                    # Assign saved plans to agents
                    plans_loaded = 0
                    for agent_name, agent in self.agents.items():
                        if agent_name in saved_plans:
                            agent.daily_plan = saved_plans[agent_name]
                            plans_loaded += 1
                            print(f"[DEBUG] Loaded saved plan for {agent_name}: {len(agent.daily_plan.get('activities', []))} activities")
                        else:
                            print(f"[WARNING] No saved plan found for {agent_name}, will generate new plan")
                            # Generate plan for this agent only
                            self._generate_single_agent_plan(agent, current_hour, current_day)
                    
                    print(f"[DEBUG] Successfully loaded {plans_loaded}/{len(self.agents)} saved plans from Day {saved_day}")
                    return
                else:
                    print(f"[DEBUG] No saved plans found, will generate new plans")
            
            # Normal plan generation (or fallback when no saved plans)
            print(f"[DEBUG] Generating new plans for {len(self.agents)} agents")
            
            for agent_name, agent in self.agents.items():
                try:
                    self._generate_single_agent_plan(agent, current_hour, current_day)
                except Exception as e:
                    print(f"[ERROR] Error creating plan for agent {agent_name}: {str(e)}")
                    traceback.print_exc()
            
            print(f"[DEBUG] _create_daily_plans() completed for Day {current_day}")

            # Save the generated plans for debugging
            self._save_plans_for_debugging(current_day, current_hour)

        except Exception as e:
            print(f"[ERROR] Error in _create_daily_plans: {str(e)}")
            traceback.print_exc()

    def _generate_single_agent_plan(self, agent: 'Agent', current_hour: int, current_day: int):
        """Generate a daily plan for a single agent."""
        print(f"[DEBUG] Generating plan for agent: {agent.name}")
        
        # Get agent context with all required fields for prompt template
        context = {
            'name': agent.name,
            'age': getattr(agent, 'age', 35),
            'occupation': getattr(agent, 'occupation', 'Worker'),
            'residence': getattr(agent, 'residence', 'Residence'),
            'workplace': getattr(agent, 'workplace', 'Office'),
            'current_time': current_hour,
            'current_day': current_day,
            'current_location': agent.get_current_location_name(),
            'energy_level': agent.energy_system.get_energy(agent.name),
            'grocery_level': agent.grocery_system.get_level(agent.name),
            'money': agent.money,
            'agent': agent,
            'memory_manager': self.memory_mgr,
            'valid_locations': list(self.locations.keys()),
            'ENERGY_MIN': ENERGY_MIN,
            'ENERGY_MAX': ENERGY_MAX,
            'ENERGY_DECAY_PER_HOUR': ENERGY_DECAY_PER_HOUR,
            'ENERGY_COST_WORK_HOUR': ENERGY_COST_WORK_HOUR,
            'ENERGY_COST_PER_STEP': ENERGY_COST_PER_STEP,
            'ENERGY_GAIN_RESTAURANT_MEAL': ENERGY_GAIN_RESTAURANT_MEAL,
            'ENERGY_GAIN_SNACK': ENERGY_GAIN_SNACK,
            'ENERGY_GAIN_HOME_MEAL': ENERGY_GAIN_HOME_MEAL,
            'ENERGY_GAIN_NAP': ENERGY_GAIN_NAP,
            'ENERGY_THRESHOLD_LOW': ENERGY_THRESHOLD_LOW
        }
        
        # DEBUG: Log context for debugging
        print(f"[PLAN_DEBUG] Context for {agent.name}:")
        print(f"[PLAN_DEBUG]   age: {context['age']}")
        print(f"[PLAN_DEBUG]   occupation: {context['occupation']}")
        print(f"[PLAN_DEBUG]   residence: {context['residence']}")
        print(f"[PLAN_DEBUG]   workplace: {context['workplace']}")
        print(f"[PLAN_DEBUG]   energy_level: {context['energy_level']}")
        print(f"[PLAN_DEBUG]   money: {context['money']}")
        print(f"[PLAN_DEBUG]   valid_locations count: {len(context['valid_locations'])}")
        
        # Get planning prompt
        prompt = self.prompt_mgr.get_planning_prompt(agent.name, current_hour, context)
        
        # Get plan from model using generate
        response = self.model_mgr.generate(prompt, prompt_type="daily_plan")
        
        # DEBUG: Log the raw response for debugging failed plans
        print(f"[PLAN_DEBUG] Raw LLM response for {agent.name}:")
        print(f"[PLAN_DEBUG] Response type: {type(response)}")
        print(f"[PLAN_DEBUG] Response length: {len(response) if isinstance(response, str) else 'N/A'}")
        if isinstance(response, str):
            print(f"[PLAN_DEBUG] First 500 chars: {response[:500]}")
            print(f"[PLAN_DEBUG] Last 500 chars: {response[-500:]}")
        else:
            print(f"[PLAN_DEBUG] Full response: {response}")
        
        # Extract and clean the plan using MemoryManager
        cleaned_plan = self.memory_mgr.extract_and_clean_plan(response, "daily_plan")
        
        print(f"[DEBUG] Daily plan generated for {agent.name}: {cleaned_plan.get('activities', [])[:2] if cleaned_plan.get('activities') else 'No activities'}...")
        
        # Store the plan in memory
        self.memory_mgr.record_plan_creation(
            agent_name=agent.name,
            raw_response=response,
            cleaned_plan=cleaned_plan,
            current_time=current_hour,
            current_day=current_day
        )
        
        # Assign the plan to the agent
        agent.daily_plan = cleaned_plan
        print(f"[DEBUG] Plan assigned to {agent.name}: {len(cleaned_plan.get('activities', []))} activities")

def main():
    """Main entry point for the town simulation."""
    simulation = None
    try:
        # Get configuration path
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "agent_configuration.json"
        )
        
        # Verify config file exists
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            print(f"Current working directory: {os.getcwd()}")
            sys.exit(1)
            
        # Initialize simulation
        simulation = TownSimulation(config_path)
        
        # Set initial time to 7:00 on day 1
        TimeManager.set_time(7)
        TimeManager.set_current_day(1)
        
        print(f"\nStarting simulation at Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
        print("Will run until Day 8, Hour 0 (end of Day 7)")
        
        # Run the simulation
        simulation.run_simulation()
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Saving final state...")
        if simulation and hasattr(simulation, 'memory_mgr'):
            try:
                current_day = TimeManager.get_current_day()
                simulation.memory_mgr.save_memories(force=True)
                simulation.memory_mgr.save_daily_summaries(simulation.metrics_mgr, current_day)
            except Exception as e:
                print(f"Error saving final state: {str(e)}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        traceback.print_exc()
        # Emergency saves on error
        if simulation and hasattr(simulation, 'memory_mgr'):
            try:
                current_day = TimeManager.get_current_day()
                simulation.memory_mgr.save_memories(force=True)
                simulation.memory_mgr.save_daily_summaries(simulation.metrics_mgr, current_day)
            except Exception as save_error:
                print(f"Error during emergency save: {str(save_error)}")
                traceback.print_exc()

if __name__ == "__main__":
    main() 