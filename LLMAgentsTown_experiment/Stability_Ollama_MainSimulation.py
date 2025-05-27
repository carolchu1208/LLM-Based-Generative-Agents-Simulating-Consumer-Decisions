#!/usr/bin/env python
# coding: utf-8

"""
Main simulation controller that connects all components:

File Dependencies:
1. Stability_Agents_Config.json: Town and agent configuration
2. deepseek_model_manager.py: LLM interaction handling using DeepSeek API
3. prompt_manager.py: Prompt templates for all interactions
4. stability_classes.py: Core Location and Agent classes
5. Stability_Memory_Manager.py: Memory storage and retrieval
6. Stability_Metrics.py: Fried Chicken Shop metrics
7. simulation_constants.py: Shared constants and settings

Flow:
1. Load configuration from Stability_Agents_Config.json
2. Initialize all managers (memory, model, prompt)
3. Create Location and Agent instances
4. Run simulation loop:
   - Process time-based events (morning, evening)
   - Handle agent actions and interactions
   - Record memories and metrics
   - Generate daily summaries
"""

import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import uuid
import traceback
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import random
import re # For improved action parsing
import threading
import concurrent.futures

from Stability_Memory_Manager import MemoryManager
from deepseek_model_manager import ModelManager as DeepSeekModelManager  # Import DeepSeek specifically
from prompt_manager import PromptManager
from stability_classes import Agent, Location, TownMap
from Stability_Metrics import StabilityMetrics
from simulation_constants import ACTIVITY_TYPES, MEMORY_TYPES

SIMULATION_SETTINGS = {
    'planning_hour': 7,
    'max_days': 7,
    'parallel_execution': True
}

def initialize_simulation():
    """Initialize the simulation environment"""
    try:
        print(f"SIM_LOG: Starting initialization at {datetime.now().strftime('%H:%M:%S')}")
        
        # Initialize managers
        memory_manager = MemoryManager()
        model_manager = DeepSeekModelManager()  # Use DeepSeek model manager
        prompt_manager = PromptManager()
        metrics = StabilityMetrics()
        print("\nInitialized managers and metrics")
        
        # Load town configuration
        config_path = os.path.join(os.path.dirname(__file__), 'Stability_Agents_Config.json')
        with open(config_path, 'r') as f:
            town_data = json.load(f)

        # Initialize TownMap
        town_map_grid_data = town_data.get('town_map_grid')
        town_map_instance = None
        if town_map_grid_data and 'world_locations' in town_map_grid_data and 'travel_paths' in town_map_grid_data:
            town_map_instance = TownMap(
                world_locations_data=town_map_grid_data['world_locations'],
                travel_paths_data=town_map_grid_data['travel_paths']
            )
            print("\nInitialized TownMap with grid data.")
        else:
            print("\nWarning: TownMap grid data not found or incomplete in config. Proceeding without grid-based map.")

        # Initialize locations
        locations = {}
        for category, category_locations in town_data['town_areas'].items():
            for location_name, info in category_locations.items():
                if isinstance(info, str):
                    info = {
                        "description": info,
                        "type": category,
                        "hours": {"open": 8, "close": 22}  # Default hours
                    }
                # Override for community places
                if category == "community":
                    info["hours"]["open"] = 6  # Keep early opening for exercise
                
                # Get grid coordinate for the location
                location_coords: Optional[Tuple[int, int]] = None
                if town_map_instance:
                    location_coords_list = town_map_grid_data['world_locations'].get(location_name)
                    if location_coords_list and isinstance(location_coords_list, list) and len(location_coords_list) == 2:
                        location_coords = tuple(location_coords_list) # type: ignore
                    else:
                        print(f"Warning: Coordinates for location '{location_name}' not found or invalid in town_map_grid.")

                # Create location object, now passing grid_coordinate
                location = Location(
                    location_name, 
                    info['type'], 
                    info.get('capacity', 10),
                    grid_coordinate=location_coords
                )
                
                # Set hours from configuration
                if 'hours' in info:
                    location.hours = info['hours']
                    print(f"Set hours for {location_name}: {location.hours['open']}:00-{location.hours['close']}:00")
                
                # Set base price from configuration
                if 'base_price' in info:
                    location.base_price = info['base_price']
                    print(f"Set base price for {location_name}: ${location.base_price:.2f}")
                
                # Apply discount settings if available
                if 'discount' in info:
                    location.discount = info['discount']
                    print(f"Applied discount settings for {location_name}: {info['discount']}")
                
                locations[location_name] = location
        
        # Initialize agents
        agents = []
        for person_name, person_data in town_data['town_people'].items():
            basics = person_data['basics']
            agent = Agent(
                name=person_name,
                residence=basics['residence'],
                family_role=basics.get('family_role', 'individual'),
                memory_manager=memory_manager,
                model_manager=model_manager,
                prompt_manager=prompt_manager,
                town_map=town_map_instance
            )
            # Set locations before initializing personal context
            agent.locations = locations
            # Set metrics instance
            agent.metrics = metrics
            print(f"\nInitialized agent {person_name}")
            print(f"Residence: {basics['residence']}")
            print(f"Workplace: {basics.get('workplace', 'None')}")
            
            # Initialize money based on income
            income_info = basics.get('income', {})
            if income_info:
                income_amount = income_info.get('amount', 0)
                income_type = income_info.get('type', 'hourly')
                if income_type == 'annual':
                    agent.money = income_amount / 365  # Daily amount
                elif income_type == 'monthly':
                    agent.money = income_amount / 30   # Daily amount
                else:  # hourly
                    agent.money = income_amount * 8    # Daily amount for 8 hours
                print(f"Initial money: ${agent.money:.2f}")
            
            # Initialize personal context which will set current_location
            agent.initialize_personal_context(person_data)
            
            # Ensure current_location is set to a Location object
            if agent.current_location is None:
                print(f"Warning: {agent.name}'s current_location was not set. Setting to residence.")
                if agent.residence in locations:
                    agent.current_location = locations[agent.residence]
                else:
                    print(f"Error: {agent.name}'s residence {agent.residence} not found in locations!")
                    continue
            
            agents.append(agent)
        
        # ADDED: Provide each agent with a list of all agents for perception
        for agent_obj in agents:
            agent_obj.all_agents_list_for_perception = agents

        return {
            'settings': SIMULATION_SETTINGS,
            'locations': locations,
            'agents': agents,
            'metrics': metrics,
            'memory_mgr': memory_manager,
            'model_mgr': model_manager,
            'prompt_mgr': prompt_manager,
            'town_map': town_map_instance
        }
    except Exception as e:
        print(f"Error initializing simulation: {str(e)}")
        traceback.print_exc()
        return None

class Simulation:
    def __init__(self, settings, locations, agents, metrics, memory_mgr, model_mgr, prompt_mgr, town_map: Optional[TownMap]):
        self.settings = settings
        self.locations: Dict[str, Location] = locations
        self.agents: List[Agent] = agents
        self.metrics = metrics
        self.memory_mgr = memory_mgr
        self.model_mgr = model_mgr
        self.prompt_mgr = prompt_mgr
        self.town_map = town_map
        self.current_day = 0
        self.total_days = settings.get('simulation', {}).get('duration_days', 7)
        self.current_hour = 0
        self.conversation_logs = []  # Store all terminal output
        self.location_locks: Dict[Location, threading.Lock] = {loc: threading.Lock() for loc in self.locations.values()}
        self.memory_lock = threading.Lock()
        
        # Generate single timestamp for this simulation run
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create file paths for this simulation run
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        records_dir = os.path.join(base_dir, "LLMAgentsTown_memory_records")
        
        self.metrics_file = os.path.join(records_dir, "simulation_metrics", f"metrics_{self.run_timestamp}.json")
        self.memories_file = os.path.join(records_dir, "simulation_agents", f"consolidated_memories_{self.run_timestamp}.json")
        self.conversation_file = os.path.join(records_dir, "simulation_conversation", f"conversation_{self.run_timestamp}.jsonl")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.memories_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.conversation_file), exist_ok=True)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals by saving state before exit"""
        print("\nReceived interrupt signal. Saving state...")
        self.save_state(is_final_save=False)
        print("State saved. Exiting...")
        sys.exit(0)

    def save_conversation_logs(self):
        """Save all conversation logs to a JSONL file"""
        try:
            with open(self.conversation_file, 'w') as f:
                for log in self.conversation_logs:
                    f.write(json.dumps(log) + '\n')
            print(f"\nConversation logs saved to: {self.conversation_file}")
            return self.conversation_file
        except Exception as e:
            print(f"Error saving conversation logs: {str(e)}")
            traceback.print_exc()
            return None

    def log_conversation(self, agent_name: str, content: str, log_type_key: str = "ACTION_RAW_OUTPUT"):
        """Add a conversation or system log entry using a MEMORY_TYPES key.
           Default log_type_key is 'ACTION_RAW_OUTPUT' for agent actions.
           Use 'SYSTEM_EVENT' or other relevant keys from MEMORY_TYPES for system messages.
        """
        if log_type_key not in MEMORY_TYPES:
            print(f"Warning: Unrecognized log_type_key '{log_type_key}' used in log_conversation. Defaulting to ACTION_RAW_OUTPUT.")
            actual_log_type_value = MEMORY_TYPES['ACTION_RAW_OUTPUT'] # Fallback to a generic type
        else:
            actual_log_type_value = MEMORY_TYPES[log_type_key]

        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # Real-world timestamp
            'simulation_day': self.current_day,
            'simulation_hour': self.current_hour,
            'agent_or_system': agent_name, # Clarified field name
            'log_type': actual_log_type_value, # The string value of the memory type
            'content': content
        }
        with self.memory_lock: # Assuming memory_lock is for conversation_logs specifically or a general critical section lock
            self.conversation_logs.append(log_entry)

    def save_state(self, is_final_save: bool = False):
        """Save current simulation state.
        If is_final_save is True, it saves the consolidated metrics for the whole simulation.
        Otherwise, it can save intermediate states (e.g., end of day).
        """
        try:
            # Save metrics (conditionally based on final save)
            if is_final_save:
                self.metrics.save_final_metrics(self.metrics_file)
                # Also save all daily summaries when it's a final save
                if hasattr(self, 'daily_insights_file_template'):
                    self.metrics.save_all_daily_summaries_to_file(self.daily_insights_file_template)
                else:
                    # Fallback for older versions where the template might not be defined in __init__
                    # This might lead to a new timestamped file for daily summaries.
                    print("Warning: daily_insights_file_template not defined in Simulation.__init__. Daily summaries on final save might use a new timestamp.")
                    self.metrics.save_all_daily_summaries_to_file() # Uses its own timestamping
            
            # Save memory manager state with specific file path
            self.memory_mgr.save_memories(self.memories_file)
            
            # Save conversation logs
            self.save_conversation_logs()
            
            # If not a final save (e.g., interruption or end of day processing)
            if not is_final_save:
                 # Save all cached daily summaries (for completed days so far)
                 if hasattr(self, 'daily_insights_file_template'):
                    self.metrics.save_all_daily_summaries_to_file(self.daily_insights_file_template)
                 else:
                    # Fallback for older versions as above
                    print("Warning: daily_insights_file_template not defined in Simulation.__init__. Daily summaries on non-final save might use a new timestamp.")
                    self.metrics.save_all_daily_summaries_to_file()

        except Exception as e:
            print(f"Error saving state: {str(e)}")
            traceback.print_exc()

    def run_simulation_sequentially(self):
        try:
            print("\nStarting sequential simulation...")
            self.log_conversation("SYSTEM", "Starting sequential simulation...", 'PLANNING_EVENT')
            
            for day in range(1, self.total_days + 1):
                self.current_day = day
                day_start_msg = f"\n=== Starting Day {self.current_day} ==="
                print(day_start_msg)
                self.log_conversation("SYSTEM", day_start_msg, 'PLANNING_EVENT')
                
                self.metrics.new_day()
                self.create_daily_plans_sequentially()
                
                for hour in range(SIMULATION_SETTINGS.get('day_start_hour', 7), SIMULATION_SETTINGS.get('day_end_hour', 24)):
                    self.current_hour = hour
                    time_msg = f"\n=== Current Time: Day {self.current_day}, {hour:02d}:00 ==="
                    print(time_msg)
                    self.log_conversation("SYSTEM", time_msg, 'PLANNING_EVENT')
                    
                    for agent in self.agents:
                        self.run_agent_sequentially(agent)
                
                self.process_end_of_day()
            
            complete_msg = "\n=== Sequential Simulation Complete ==="
            print(complete_msg)
            self.log_conversation("SYSTEM", complete_msg, 'PLANNING_EVENT')
            self.save_state(is_final_save=True)
            success_msg = "\nSequential simulation completed successfully!"
            print(success_msg)
            self.log_conversation("SYSTEM", success_msg, 'PLANNING_EVENT')
        
        except Exception as e:
            error_msg = f"Error in sequential simulation: {str(e)}"
            print(error_msg)
            self.log_conversation("SYSTEM", error_msg, 'SYSTEM_EVENT')
            traceback.print_exc()
            self.save_state(is_final_save=False)

    def run_agent_sequentially(self, agent: Agent):
        try:
            agent.current_time = self.current_hour  # Fix: Use just the current hour
            
            current_location_obj = agent.current_location
            if not isinstance(current_location_obj, Location):
                if agent.residence in self.locations:
                    agent.current_location = self.locations[agent.residence]
                    current_location_obj = agent.current_location
                else:
                    return

            is_work_time = False
            if hasattr(agent, 'work_schedule') and agent.work_schedule:
                work_start = agent.work_schedule.get('start', 9)
                work_end = agent.work_schedule.get('end', 17)
                is_work_time = work_start <= self.current_hour < work_end
            
            context = self.get_agent_context(agent)
            
            action_result_str = agent.generate_contextual_action(context)
            
            # Log the raw action output from LLM
            self.log_conversation(agent.name, action_result_str, 'ACTION_RAW_OUTPUT')
            agent.current_activity = action_result_str
            
            # Simplified energy update based on action content - this is very basic
            if "eat" in action_result_str.lower() or "food" in action_result_str.lower():
                agent.energy_level = min(100, agent.energy_level + 30)
            elif "work" in action_result_str.lower():
                agent.energy_level = max(0, agent.energy_level - 10)
            elif "sleep" in action_result_str.lower() or "rest" in action_result_str.lower():
                agent.energy_level = min(100, agent.energy_level + 50)
            else:
                agent.energy_level = max(0, agent.energy_level - 5)
            agent.energy_level = max(0, agent.energy_level - 2) # Natural decay

            # Conversation generation and logging is now primarily handled within agent.generate_conversation
            # which uses 'CONVERSATION_LOG_EVENT' memory type.
            # If a raw action string implies a conversation, the LLM call within agent.generate_contextual_action
            # or a subsequent call to agent.generate_conversation would log it appropriately.
            # This specific block for conversation logging here might be redundant if agent methods handle it.
            
            # Example: If action implies conversation, it could be a distinct step triggered by action parsing later
            if any(word in action_result_str.lower() for word in ["talk to", "chat with", "discuss with"]):
                # This is a simplified trigger; actual conversation generation should be more robustly handled
                # by the Agent class potentially based on this action string.
                # For now, we assume agent.generate_contextual_action might return a string that IS the conversation,
                # or an action that LEADS to a conversation (handled by agent.generate_conversation method itself)
                pass # Conversation logging is handled by agent.generate_conversation which logs CONVERSATION_LOG_EVENT

            agent.update_state() # Call update_state at the end of the agent's turn, logs AGENT_STATE_UPDATE_EVENT
        
        except Exception as e:
            error_msg = f"Error processing agent {agent.name} sequentially: {str(e)}"
            print(error_msg)
            self.log_conversation("SYSTEM", error_msg, 'SYSTEM_EVENT')
            traceback.print_exc()

    def run_simulation_parallel(self):
        try:
            for day in range(1, self.total_days + 1):
                self.current_day = day
                
                # Start each day at 7 AM
                for hour in range(7, 24):  # Run from 7 AM to 11 PM
                    self.current_hour = hour
                    
                    # Create daily plans at the start of each day (7 AM)
                    if hour == 7:
                        self.create_daily_plans_sequentially()
                    
                    self.run_hour_parallel()
                    
                    # Process end of hour - agents handle their needs
                    for agent in self.agents:
                        # Check and handle food needs
                        if agent.needs_food(self.current_hour):
                            agent.handle_food_needs(self.current_hour)
                        
                        # Check and handle grocery needs
                        if agent.needs_groceries(self.current_hour):
                            grocery_store = agent.find_closest_food_location(self.current_hour)
                            if grocery_store:
                                agent.start_travel_to(grocery_store)
                        
                        # Natural energy decay and state update
                        agent.energy_level = max(0, agent.energy_level - 2)  # Natural energy decay
                        agent.update_state()
                        
                    # Update metrics
                    self.metrics.record_hour_metrics(self.current_day, hour, self.agents)
                    
                # Process end of day
                self.process_end_of_day()
                
            # Save final metrics
            self.metrics.save_metrics()
            
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            traceback.print_exc()

    def create_daily_plans_sequentially(self):
        for agent in self.agents:
            try:
                context = self.get_agent_context(agent)
                plan = agent.create_daily_plan(self.current_hour, context)
                if not plan:
                    print(f"Warning: Failed to create daily plan for {agent.name}")
            except Exception as e:
                print(f"Error creating daily plan for {agent.name}: {str(e)}")
                traceback.print_exc()

    def run_hour_parallel(self):
        with ThreadPoolExecutor(max_workers=len(self.agents) if self.agents else 1) as executor:
            futures = [
                executor.submit(self.run_agent_parallel, agent)
                for agent in self.agents
            ]
            concurrent.futures.wait(futures)

    def extract_target_location_name(self, action_string: str, agent_name: str) -> Optional[str]:
        """Placeholder: Extracts a potential location name from an action string.
        E.g., "I am moving to The Mall" -> "The Mall"
        This needs a more robust implementation (e.g., regex, NLP).
        """
        parts = action_string.lower().split("moving to ")
        if len(parts) > 1:
            potential_location = parts[1].split(" from ")[0].split(" with ")[0].strip()
            return ' '.join(word.capitalize() for word in potential_location.split())
        return None

    def get_agent_context(self, agent: Agent) -> Dict[str, Any]:
        """Builds the context dictionary for an agent's action generation."""
        current_location = agent.current_location
        
        if not current_location or not hasattr(current_location, 'name'):
            current_location_name = agent.residence
            current_location_type = "residence"
            print(f"Warning: Agent {agent.name} had invalid current_location. Defaulting to residence for context.")
            if agent.residence in self.locations:
                agent.current_location = self.locations[agent.residence]
                current_location = agent.current_location
            else:
                print(f"CRITICAL: Agent {agent.name} residence {agent.residence} not in self.locations.")

        else:
            current_location_name = current_location.name
            current_location_type = current_location.type

        nearby_agents_list = []
        if agent.current_location:
            for other_agent in self.agents:
                if other_agent != agent and other_agent.current_location == agent.current_location:
                    nearby_agents_list.append(other_agent.name)
        
        is_work_time = False
        if hasattr(agent, 'work_schedule') and agent.work_schedule:
            work_start = agent.work_schedule.get('start', 9)
            work_end = agent.work_schedule.get('end', 17)
            is_work_time = work_start <= self.current_hour < work_end

        context = {
            'name': agent.name,
            'location': current_location_name,
            'current_location': current_location_name,  # Add this explicitly for planning
            'time': self.current_hour,
            'nearby_agents': nearby_agents_list,
            'location_type': current_location_type,
            'energy_level': agent.energy_level,
            'grocery_level': agent.grocery_level,
            'money': agent.money,
            'current_activity': agent.current_activity if hasattr(agent, 'current_activity') else "idle",
            'daily_plan': agent.daily_plan if hasattr(agent, 'daily_plan') else "No plan specified.",
            'is_work_time': is_work_time,
            'recent_activities': agent.get_recent_activities(limit=3)
        }
        return context

    def run_agent_parallel(self, agent: Agent):
        try:
            agent.current_time = self.current_hour
            context = self.get_agent_context(agent)
            
            action_str = agent.generate_contextual_action(context)
            print(f"Agent {agent.name} at {self.current_hour}:00: {action_str}")
            
            # Log the raw action output from the LLM via the simulation-wide log
            self.log_conversation(agent.name, action_str, 'ACTION_RAW_OUTPUT') 
            agent.current_activity = action_str

            # Action Parsing and Handling
            action_lower = action_str.lower()

            # Handle Conversations
            if any(keyword in action_lower for keyword in ["talk to", "chat with", "discuss with", "speak to", "ask"]):
                conv_context = self.get_agent_context(agent) 
                if conv_context['nearby_agents']:
                    self.log_conversation(agent.name, f"Attempting to converse with {conv_context['nearby_agents']}. Action: '{action_str}'", 'CONVERSATION_LOG_EVENT')
                    conversation_text = agent.generate_conversation(conv_context)
            
        except Exception as e:
            error_msg = f"Error processing agent {agent.name} in parallel: {str(e)}"
            print(error_msg)
            self.log_conversation("SYSTEM", error_msg, 'SYSTEM_EVENT')
            traceback.print_exc()

    def process_end_of_day(self):
        self.log_conversation("SYSTEM", f"End of Day {self.current_day} processing.", 'PLANNING_EVENT')
        daily_summary_data = self.metrics.get_daily_summary_data(self.current_day)
        self.metrics.print_daily_summary(daily_summary_data, self.current_day) 
        self.log_conversation("SYSTEM", f"Daily summary printed for Day {self.current_day}.", 'PLANNING_EVENT')
        self.save_state(is_final_save=False)

if __name__ == "__main__":
    try:
        print("\n=== Starting Simulation ===")
        print(f"Real-world start time: {datetime.now().strftime('%H:%M:%S')}")
        
        sim_data = initialize_simulation()
        if not sim_data:
            raise ValueError("Failed to initialize simulation. Exiting.")
        
        simulation = Simulation(
            settings=sim_data['settings'],
            locations=sim_data['locations'],
            agents=sim_data['agents'],
            metrics=sim_data['metrics'],
            memory_mgr=sim_data['memory_mgr'],
            model_mgr=sim_data['model_mgr'],
            prompt_mgr=sim_data['prompt_mgr'],
            town_map=sim_data['town_map']
        )
        
        simulation.run_simulation_parallel()

    except ValueError as ve:
        print(f"\nInitialization Error: {str(ve)}")
        traceback.print_exc()
    except Exception as e:
        print(f"\nFatal error in main execution: {str(e)}")
        traceback.print_exc()
        if 'simulation' in locals() and simulation is not None:
            try:
                simulation.save_state(is_final_save=False)
                print("Attempted to save state before exiting due to fatal error.")
            except Exception as se:
                print(f"Could not save state during fatal error handling: {se}")
    finally:
        print("\n=== Simulation Ended ===")
        print(f"Real-world end time: {datetime.now().strftime('%H:%M:%S')}")