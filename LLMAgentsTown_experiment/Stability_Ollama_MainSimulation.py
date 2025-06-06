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
8. energy_constants.py: Energy system constants

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
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
import traceback
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import random
import re # For improved action parsing
import threading
import concurrent.futures
from collections import defaultdict

from Stability_Memory_Manager import MemoryManager
from deepseek_model_manager import ModelManager as DeepSeekModelManager  # Import DeepSeek specifically
from prompt_manager import PromptManager
from stability_classes import Agent, Location, TownMap, ConversationState, shared_location_tracker
from Stability_Metrics import MetricsManager
from simulation_constants import (
    SIMULATION_SETTINGS, ACTIVITY_TYPES, MEMORY_TYPES,
    TimeManager, SimulationError, AgentError, LocationError,
    MemoryError, MetricsError, ErrorHandler, ThreadSafeBase,
    ENERGY_MAX, ENERGY_MIN, ENERGY_COST_PER_STEP, ENERGY_DECAY_PER_HOUR,
    ENERGY_COST_WORK_HOUR, ENERGY_COST_PER_HOUR_TRAVEL, ENERGY_COST_PER_HOUR_IDLE,
    ENERGY_GAIN_RESTAURANT_MEAL, ENERGY_GAIN_SNACK, ENERGY_GAIN_HOME_MEAL,
    ENERGY_GAIN_SLEEP, ENERGY_GAIN_NAP, ENERGY_GAIN_CONVERSATION,
    ENERGY_THRESHOLD_LOW,
    shared_memory_buffer,  # Add shared memory buffer import
    MemoryEvent
)

def initialize_simulation() -> Optional[Dict[str, Any]]:
    """Initialize the simulation environment.
    
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing initialized simulation components or None if initialization fails
    """
    try:
        print(f"SIM_LOG: Starting initialization at {datetime.now().strftime('%H:%M:%S')}")

        # Initialize managers with proper directory structure
        base_memory_dir = 'LLMAgentsTown_memory_records'
        os.makedirs(base_memory_dir, exist_ok=True)
        
        # Set up memory manager with simulation_memory directory
        memory_dir = os.path.join(base_memory_dir, 'simulation_memory')
        os.makedirs(memory_dir, exist_ok=True)
        memory_manager = MemoryManager(memory_dir=memory_dir)
        
        # Set up metrics manager with simulation_metrics directory
        metrics_dir = os.path.join(base_memory_dir, 'simulation_metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics = MetricsManager(simulation_dir=metrics_dir)
        
        model_manager = DeepSeekModelManager()
        prompt_manager = PromptManager()
        
        print("\nInitializing managers for new simulation")
        
        # Load test configuration
        config_path = os.path.join(os.path.dirname(__file__), 'Stability_Agents_Config_Test.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError("Test configuration file not found. Please ensure Stability_Agents_Config_Test.json exists.")
        
        print(f"Loading configuration from: {config_path}")
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
                
                # Set hours directly from configuration
                if 'hours' in info:
                    location.hours = info['hours']
                    if 'always_open' in info['hours']:
                        print(f"Set {location_name} as always open")
                    else:
                        print(f"Hours for {location_name}: {info['hours']['open']}:00-{info['hours']['close']}:00")
                
                # Set base price from configuration
                if 'base_price' in info:
                    location.base_price = info['base_price']
                    print(f"Set base price for {location_name}: ${location.base_price:.2f}")
                
                # Apply discount settings if available
                if 'discount' in info:
                    location.discounts = info['discount']  # Note: Changed from discount to discounts to match Location class
                    print(f"Applied discount settings for {location_name}: {info['discount']}")
                
                locations[location_name] = location
        
        # Initialize agents
        agents = []
        for person_name, person_data in town_data['town_people'].items():
            basics = person_data['basics']
            agent = Agent(
                name=basics['name'],
                age=basics['age'],
                occupation=basics['occupation'],
                residence=basics['residence'],
                workplace=basics['workplace'],
                work_schedule=basics['income']['schedule'],
                memory_mgr=memory_manager
            )
            # Set additional attributes after initialization
            agent.model_manager = model_manager
            agent.prompt_manager = prompt_manager
            agent.town_map = town_map_instance
            # Ensure meals_today is initialized
            agent.meals_today = {
                'breakfast': {'handled': False, 'method': None},
                'lunch': {'handled': False, 'method': None},
                'dinner': {'handled': False, 'method': None},
                'snack': {'handled': False, 'method': None}
            }
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
                
                # Calculate daily income for starting money calculation
                daily_income = 0
                if income_type == 'annual':
                    daily_income = income_amount / 365
                elif income_type == 'monthly':
                    daily_income = income_amount / 30
                else:  # hourly
                    work_hours = 8  # Standard work day
                    daily_income = income_amount * work_hours
                
                # Start with 3 days worth of income instead of a full month
                agent.money = daily_income * 3
                print(f"Initial money: ${agent.money:.2f} (3 days of ${daily_income:.2f} daily income)")
            
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
        
        # ADDED: Initialize shared location tracker with all agent positions
        shared_location_tracker.clear()  # Clear any previous data
        
        for agent in agents:
            current_grid_coord = None
            if agent.current_location and hasattr(agent.current_location, 'grid_coordinate'):
                current_grid_coord = agent.current_location.grid_coordinate
            
            shared_location_tracker.update_agent_position(
                agent.name,
                agent.get_current_location_name(),
                current_grid_coord,
                0  # Initial time
            )
            print(f"Registered {agent.name} at {agent.get_current_location_name()} with shared tracker")
        
        print(f"Shared location tracker initialized with {len(agents)} agents")
        
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
    def __init__(self, config: Dict[str, Any]):
        """Initialize the simulation with configuration."""
        try:
            # Initialize basic attributes
            self.config = config
            self.current_time = 0
            self.current_day = 1
            self.total_days = config.get('total_days', 7)
            self.total_hours = self.total_days * 24
            self.max_hours = self.total_days * 24
            self.plans_created = False
            
            # Initialize file paths
            self.base_dir = 'LLMAgentsTown_memory_records'
            os.makedirs(self.base_dir, exist_ok=True)
            
            # Initialize memory manager with proper file paths
            memory_dir = os.path.join(self.base_dir, 'simulation_memory')
            os.makedirs(memory_dir, exist_ok=True)
            self.memory_mgr = MemoryManager(memory_dir)
            
            # Initialize conversation logs directory
            conversation_dir = os.path.join(self.base_dir, 'simulation_conversation')
            os.makedirs(conversation_dir, exist_ok=True)
            self.conversation_log_file = os.path.join(conversation_dir, f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}_conversations.jsonl')
            
            # Initialize agent memories file
            self.agent_memories_file = os.path.join(memory_dir, 'agent_memories.json')
            if not os.path.exists(self.agent_memories_file):
                with open(self.agent_memories_file, 'w') as f:
                    json.dump({}, f)
            
            # Initialize metrics directory and manager
            metrics_dir = os.path.join(self.base_dir, 'simulation_metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            self.metrics = MetricsManager(simulation_dir=metrics_dir)
            
            # Initialize metrics file
            self.metrics_file = os.path.join(metrics_dir, 'metrics.json')
            if not os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'w') as f:
                    json.dump({}, f)
            
            # Initialize daily summary file
            self.daily_summary_file = os.path.join(metrics_dir, 'daily_summaries.jsonl')
            
            # Initialize locations first
            self.locations = {}
            self._initialize_locations()
            
            # Initialize agents with memory manager
            self.agents = []
            self._initialize_agents()
            
            # Initialize locks for thread safety
            self._conversation_locks = {agent.name: threading.Lock() for agent in self.agents}
            self._social_locks = {agent.name: threading.Lock() for agent in self.agents}
            
            # Initialize debug stats
            self.debug_stats = {
                'hourly': defaultdict(lambda: {'successful': 0, 'failed': 0, 'errors': defaultdict(list)}),
                'daily': defaultdict(lambda: {'successful': 0, 'failed': 0, 'errors': defaultdict(list)})
            }
            
            # Record initial state for all agents
            self._record_initial_states()
            
        except Exception as e:
            print(f"Error initializing simulation: {str(e)}")
            traceback.print_exc()
            raise

    def _record_initial_states(self):
        """Record initial states for all agents."""
        try:
            for agent in self.agents:
                # Record initial state
                state_data = {
                    'time': 0,
                    'day': 1,
                    'location': agent.residence,
                    'energy_level': agent.energy_level,
                    'grocery_level': agent.grocery_level,
                    'money': agent.money,
                    'current_activity': 'INITIALIZATION'
                }
                agent.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
                
                # Record initial location
                location_data = {
                    'location': agent.residence,
                    'time': 0,
                    'day': 1,
                    'activity_type': 'INITIALIZATION'
                }
                agent.record_memory('LOCATION_EVENT', location_data)
                
                # Record initial plan
                plan_data = {
                    'time': 0,
                    'day': 1,
                    'location': agent.residence,
                    'plan_type': 'INITIAL',
                    'description': f"Initial daily plan for {agent.name}"
                }
                agent.record_memory('PLANNING_EVENT', plan_data)
                
        except Exception as e:
            print(f"Error recording initial states: {str(e)}")
            traceback.print_exc()

    def _initialize_agents(self):
        """Initialize agents from configuration."""
        try:
            # Get agent configurations
            agent_configs = self.config.get('town_people', {})
            
            # Initialize each agent
            for agent_name, agent_data in agent_configs.items():
                basics = agent_data.get('basics', {})
                
                # Create agent
                agent = Agent(
                    name=basics['name'],
                    age=basics['age'],
                    occupation=basics['occupation'],
                    residence=basics['residence'],
                    workplace=basics['workplace'],
                    work_schedule=basics['income']['schedule'],
                    memory_mgr=self.memory_mgr
                )
                
                # Set locations dictionary
                agent.locations = self.locations
                
                # Initialize personal context which will set current_location
                agent.initialize_personal_context(agent_data)
                
                # Add to agents list
                self.agents.append(agent)
                
                # Register with location tracker
                if hasattr(self, 'location_tracker'):
                    self.location_tracker.update_agent_position(
                        agent_name=agent.name,
                        location_name=agent.residence,
                        grid_coordinate=None,
                        timestamp=0
                    )
                
                print(f"Registered {agent.name} at {agent.residence} with shared tracker")
            
            print(f"Shared location tracker initialized with {len(self.agents)} agents")
            
        except Exception as e:
            print(f"Error initializing agents: {str(e)}")
            traceback.print_exc()

    def _initialize_locations(self):
        """Initialize locations from configuration."""
        try:
            # Get location configurations
            location_configs = self.config.get('town_areas', {})
            
            # Initialize locations for each area type
            for area_type, locations in location_configs.items():
                for location_name, location_data in locations.items():
                    # Get grid coordinates
                    grid_coord = self.config.get('town_map_grid', {}).get('world_locations', {}).get(location_name)
                    
                    # Create location
                    location = Location(
                        name=location_name,
                        type=location_data.get('type', area_type),
                        capacity=location_data.get('capacity', 10),
                        grid_coordinate=tuple(grid_coord) if grid_coord else None
                    )
                    
                    # Set hours if specified
                    if 'hours' in location_data:
                        location.set_hours(location_data['hours'])
                    
                    # Set base price if specified
                    if 'base_price' in location_data:
                        location.base_price = location_data['base_price']
                    
                    # Set discounts if specified
                    if 'discount' in location_data:
                        location.discounts = location_data['discount']
                    
                    # Add to locations dictionary
                    self.locations[location_name] = location
                    
                    print(f"Initialized location: {location_name} ({location_data.get('type', area_type)})")
            
            print(f"Initialized {len(self.locations)} locations")
            
        except Exception as e:
            print(f"Error initializing locations: {str(e)}")
            traceback.print_exc()

    def run_simulation(self):
        """Run the simulation for the specified number of days."""
        try:
            print(f"\n=== Starting Simulation for {self.total_days} Days ===")
            
            # Record simulation start
            start_data = {
                'event_type': 'simulation_start',
                'time': 0,
                'day': 1,
                'total_days': self.total_days,
                'total_agents': len(self.agents)
            }
            for agent in self.agents:
                agent.record_memory('SYSTEM_EVENT', start_data)
            
            # Main simulation loop
            for hour in range(self.total_hours):
                self.current_time = hour
                self.current_day = hour // 24 + 1
                
                print(f"\n=== Day {self.current_day} Hour {hour % 24:02d}:00 ===")
                
                # Process each agent
                for agent in self.agents:
                    try:
                        # Record hourly state update
                        state_data = {
                            'time': hour,
                            'day': self.current_day,
                            'location': agent.get_current_location_name(),
                            'energy_level': agent.energy_level,
                            'grocery_level': agent.grocery_level,
                            'money': agent.money,
                            'current_activity': agent.current_activity
                        }
                        agent.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
                        
                        # Process agent's actions
                        self._process_agent(agent, hour)
                        
                    except Exception as e:
                        print(f"Error processing agent {agent.name}: {str(e)}")
                        traceback.print_exc()
                
                # Process end of hour
                self._process_hour_end(self.current_day, hour % 24)
            
            # Record simulation end
            end_data = {
                'event_type': 'simulation_end',
                'time': self.total_hours,
                'day': self.total_days,
                'total_agents': len(self.agents)
            }
            for agent in self.agents:
                agent.record_memory('SYSTEM_EVENT', end_data)
            
            print("\n=== Simulation Complete ===")
            
        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            traceback.print_exc()
            raise

    def handle_interrupt(self, signum: int, frame: Any) -> None:
        print("\nReceived interrupt signal. Saving state and shutting down...")
        try:
            self.save_state(is_final_save=False)
            print("State saved successfully.")
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
            print("Shutdown complete. Exiting...")
            os._exit(0)
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            os._exit(1)

    def log_conversation(self, agent_name: str, content: str, log_type_key: str = "CONVERSATION_EVENT") -> None:
        """Log a conversation with proper validation."""
        try:
            # Create conversation memory data
            memory_data = {
                'agent_name': agent_name,
                'content': content,
                'time': self.current_time,
                'location': self.get_agent_location(agent_name),
                'type': log_type_key
            }
            
            # Record as CONVERSATION_EVENT
            self.memory_mgr.record_memory(agent_name, log_type_key, memory_data)
            
        except Exception as e:
            print(f"Error logging conversation: {str(e)}")
            traceback.print_exc()

    def save_state(self, is_final_save: bool = False) -> None:
        """Save the current simulation state."""
        try:
            # Save agent memories
            with open(self.agent_memories_file, 'r+') as f:
                try:
                    memories = json.load(f)
                except json.JSONDecodeError:
                    memories = {}
                
                for agent in self.agents:
                    # Get recent memories for each memory type
                    agent_memories = []
                    for memory_type in MEMORY_TYPES:
                        recent_memories = self.memory_mgr.get_recent_memories(agent.name, memory_type, limit=1000)
                        if recent_memories:
                            agent_memories.extend(recent_memories)
                    
                    if agent_memories:
                        if agent.name not in memories:
                            memories[agent.name] = []
                        # Add simulation time to each memory
                        for memory in agent_memories:
                            memory['simulation_day'] = self.current_day
                            memory['simulation_hour'] = self.current_time % 24
                            memory['simulation_time'] = self.current_time
                        memories[agent.name].extend(agent_memories)
                
                f.seek(0)
                json.dump(memories, f, indent=2)
                f.truncate()
            
            # Save metrics
            with open(self.metrics_file, 'r+') as f:
                try:
                    metrics_data = json.load(f)
                except json.JSONDecodeError:
                    metrics_data = {}
                
                metrics_data[f'day_{self.current_day}_hour_{self.current_time % 24}'] = self.metrics.get_current_metrics()
                
                f.seek(0)
                json.dump(metrics_data, f, indent=2)
                f.truncate()
            
            # Save daily summary in JSONL format
            if is_final_save:
                with open(self.daily_summary_file, 'a') as f:
                    summary = {
                        'simulation_day': self.current_day,
                        'simulation_hour': self.current_time % 24,
                        'simulation_time': self.current_time,
                        'metrics': self.metrics.get_current_metrics(),
                        'agent_states': {
                            agent.name: {
                                'energy_level': agent.energy_level,
                                'money': agent.money,
                                'grocery_level': agent.grocery_level,
                                'current_location': agent.get_current_location_name()
                            } for agent in self.agents
                        }
                    }
                    f.write(json.dumps(summary) + '\n')
            
        except Exception as e:
            print(f"Error saving simulation state: {str(e)}")
            traceback.print_exc()

    def run_simulation_sequentially(self):
        """Run the simulation sequentially."""
        try:
            # Start at 7 AM on day 1
            self.current_time = 7 if self.current_day == 1 else 0
            print(f"\n=== Starting Day {self.current_day} at {self.current_time:02d}:00 ===\n")
            
            # Create daily plans for all agents if not already created
            if not self.plans_created:
                self.create_daily_plans_sequentially()
            
            # Run each hour of the day
            while self.current_time < self.max_hours:
                print(f"\n=== Day {self.current_day} Hour {self.current_time:02d}:00 ===\n")
                
                # Run each agent's actions for this hour
                for agent in self.agents:
                    self.run_agent_sequentially(agent, self.current_time)
                
                # Process end of hour
                self._process_hour_end(self.current_day, self.current_time)
                
                # Increment time
                self.current_time += 1
            
            # Process end of day
            self.process_end_of_day()
            
            # Move to next day if not at max days
            if self.current_day < self.total_days:
                self.current_day += 1
                self.current_time = 0  # Reset to midnight for next day
                self.metrics.new_day(force_day=self.current_day)
                print(f"\n=== Starting Day {self.current_day} at {self.current_time:02d}:00 ===\n")
                return True
            else:
                print("\n=== Simulation Complete ===\n")
                return False
                
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            traceback.print_exc()
            return False

    def run_agent_sequentially(self, agent: Agent, current_time: int) -> None:
        """Minimal sequential execution for debugging purposes."""
        try:
            # Basic state update
            agent.current_time = current_time
            agent.update_state()
            print(f"[Day {self.current_day} | Hour {current_time}] [Agent: {agent.name}] [DEBUG] State updated")
            
            # Get context and generate action
            context = self.get_agent_context(agent)
            action = agent.generate_contextual_action(context, current_time)
            print(f"[Day {self.current_day} | Hour {current_time}] [Agent: {agent.name}] [DEBUG] Generated action: {action}")
            
            # Process the action if present
            if action:
                try:
                    # Extract target location if present
                    target_location = self.extract_target_location_name(action, agent.name)
                    if target_location:
                        print(f"[Day {self.current_day} | Hour {current_time}] [Agent: {agent.name}] [DEBUG] Starting travel to {target_location}. Energy: {agent.energy_level}")
                        agent.start_travel_to(target_location)
                    
                    # Handle food needs if energy is low or it's meal time
                    if agent.energy_level < ENERGY_THRESHOLD_LOW or TimeManager.is_meal_time(current_time)[0]:
                        print(f"[Day {self.current_day} | Hour {current_time}] [Agent: {agent.name}] [DEBUG] Low energy ({agent.energy_level}) or meal time, handling food needs")
                        self._handle_food_needs(agent, current_time)
                    
                    # Handle work if it's work time
                    if agent.is_work_time():
                        print(f"[Day {self.current_day} | Hour {current_time}] [Agent: {agent.name}] [DEBUG] Work time, handling work. Energy: {agent.energy_level}")
                        self._handle_work(agent, current_time)
                    
                    # Record successful action
                    self.debug_stats['hourly'][current_time]['successful'] += 1
                    self.debug_stats['daily'][self.current_day]['successful'] += 1
                        
                except Exception as e:
                    # Record failed action with context
                    error_context = {
                        'energy': agent.energy_level,
                        'location': agent.get_current_location_name(),
                        'action': action,
                        'plan': agent.get_current_active_plan(current_time) if hasattr(agent, 'get_current_active_plan') else None
                    }
                    error_info = {
                        'error': str(e),
                        'context': error_context,
                        'traceback': traceback.format_exc()
                    }
                    self.debug_stats['hourly'][current_time]['failed'] += 1
                    self.debug_stats['daily'][self.current_day]['failed'] += 1
                    self.debug_stats['hourly'][current_time]['errors'][agent.name].append(error_info)
                    self.debug_stats['daily'][self.current_day]['errors'][agent.name].append(error_info)
                    
                    print(f"[Day {self.current_day} | Hour {current_time}] [Agent: {agent.name}] [ERROR] Error processing action: {str(e)}")
                    print(f"Context: {json.dumps(error_context, indent=2)}")
                    traceback.print_exc()
            
        except Exception as e:
            # Record failed action with context
            error_context = {
                'energy': agent.energy_level,
                'location': agent.get_current_location_name(),
                'action': None,
                'plan': agent.get_current_active_plan(current_time) if hasattr(agent, 'get_current_active_plan') else None
            }
            error_info = {
                'error': str(e),
                'context': error_context,
                'traceback': traceback.format_exc()
            }
            self.debug_stats['hourly'][current_time]['failed'] += 1
            self.debug_stats['daily'][self.current_day]['failed'] += 1
            self.debug_stats['hourly'][current_time]['errors'][agent.name].append(error_info)
            self.debug_stats['daily'][self.current_day]['errors'][agent.name].append(error_info)
            
            print(f"[Day {self.current_day} | Hour {current_time}] [Agent: {agent.name}] [ERROR] Error in sequential mode: {str(e)}")
            print(f"Context: {json.dumps(error_context, indent=2)}")
            traceback.print_exc()

    def _process_hour_end(self, day: int, hour: int) -> None:
        """Process end of hour activities and record metrics."""
        try:
            # Record metrics at the end of each day (hour 23)
            if hour == 23:
                self.metrics_mgr.record_daily_metrics(day, self.agents, self.memory_mgr)
            
        except Exception as e:
            print(f"Error processing hour end: {str(e)}")
            traceback.print_exc()

    def create_daily_plans_sequentially(self):
        """Create daily plans for all agents sequentially."""
        if self.plans_created:
            print("Daily plans already created, skipping...")
            return
            
        print("\n=== Creating Initial Daily Plans ===")
        
        for agent in self.agents:
            try:
                print(f"\n[DEBUG] Creating initial plan for {agent.name}")
                
                # Set the agent's current time to 7
                agent.current_time = 7
                
                # Get agent context
                context = self.get_agent_context(agent)
                print(f"[DEBUG] Generated context for {agent.name}:")
                for key, value in context.items():
                    print(f"[DEBUG] {key}: {value}")
                
                # Get daily plan prompt
                prompt = self.prompt_mgr.get_prompt('daily_plan', **context)
                
                # Generate plan using LLM
                plan = self.model_mgr.generate(prompt)
                print(f"[DEBUG] Created plan for {agent.name}")
                print(f"[DEBUG] Plan: {plan}")
                
                # Store raw plan text
                agent.daily_plan = plan
                
            except Exception as e:
                print(f"Error creating plan for {agent.name}: {str(e)}")
                traceback.print_exc()
        
        self.plans_created = True
        print("\n=== Daily Plans Created ===\n")

    def handle_plan_review_conversation(self, agent_a: 'Agent', agent_b: 'Agent', current_time: int):
        """Handle a conversation between household members to review and coordinate their daily plans.
        
        The conversation will be initiated by the agent with the alphabetically first name,
        using the unified conversation handling from stability_classes.py.
        """
        try:
            # Mark both agents as in conversation
            agent_a.is_in_conversation = True
            agent_b.is_in_conversation = True
            agent_a.conversation_partners = [agent_b.name]
            agent_b.conversation_partners = [agent_a.name]
            
            # Get satisfaction ratings for both agents
            agent_a_satisfaction = agent_a.get_satisfaction_rating()
            agent_b_satisfaction = agent_b.get_satisfaction_rating()
            
            # Build conversation context
            conversation_context = {
                'agent_a_name': agent_a.name,
                'agent_b_name': agent_b.name,
                'agent_a_plan': agent_a.daily_plan,
                'agent_b_plan': agent_b.daily_plan,
                'agent_a_energy': agent_a.energy_level,
                'agent_b_energy': agent_b.energy_level,
                'agent_a_satisfaction': agent_a_satisfaction,
                'agent_b_satisfaction': agent_b_satisfaction,
                'current_time': current_time,
                'location': agent_a.get_current_location_name(),
                'agent_a_recent_activities': agent_a.get_recent_activities(),
                'agent_b_recent_activities': agent_b.get_recent_activities()
            }
            
            # Use the unified conversation handling
            agent_a.handle_agent_conversation(agent_a, [agent_b])
            
            # Record the conversation in both agents' memories
            agent_a.record_memory('CONVERSATION_LOG_EVENT', {
                'content': f"Had a plan review conversation with {agent_b.name}",
                'time': current_time,
                'location': agent_a.get_current_location_name(),
                'participants': [agent_a.name, agent_b.name],
                'context': conversation_context
            })
            
            agent_b.record_memory('CONVERSATION_LOG_EVENT', {
                'content': f"Had a plan review conversation with {agent_a.name}",
                'time': current_time,
                'location': agent_b.get_current_location_name(),
                'participants': [agent_a.name, agent_b.name],
                'context': conversation_context
            })
            
        except Exception as e:
            print(f"Error in handle_plan_review_conversation: {str(e)}")
            traceback.print_exc()
        finally:
            # Reset conversation state
            agent_a.is_in_conversation = False
            agent_b.is_in_conversation = False
            agent_a.conversation_partners = []
            agent_b.conversation_partners = []

    def create_daily_plans_parallel(self):
        """Create daily plans for all agents in parallel."""
        if self.plans_created:
            print("Daily plans already created, skipping...")
            return
            
        print("\n=== Creating Initial Daily Plans in Parallel ===")
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = []
            for agent in self.agents:
                future = executor.submit(self._create_agent_plan, agent)
                futures.append(future)
            
            # Wait for all plans to be created
            concurrent.futures.wait(futures)
        
        self.plans_created = True
        print("=== Daily Plans Created ===\n")
    
    def _create_agent_plan(self, agent: 'Agent'):
        """Helper method to create a plan for a single agent."""
        try:
            print(f"\n[DEBUG] Creating initial plan for {agent.name}")
            
            # Get agent context
            context = self.get_agent_context(agent)
            print(f"[DEBUG] Generated context for {agent.name}:")
            # Print only essential context information
            essential_keys = [
                'name', 'age', 'occupation', 'workplace', 'work_schedule_start',
                'work_schedule_end', 'residence', 'current_location', 'current_time',
                'energy_level', 'money', 'daily_income', 'grocery_level'
            ]
            for key in essential_keys:
                if key in context:
                    print(f"[DEBUG] {key}: {context[key]}")
            
            # Create plan with proper locking
            with self._conversation_locks[agent.name]:
                plan = agent.create_daily_plan(self.current_time, context)
                if plan:
                    agent.daily_plan = plan
                    # Parse and log activities from plan
                    activities = agent.parse_llm_plan_to_activity(plan)
                    if activities:
                        for activity in activities:
                            agent.add_activity(activity)
                    
                    # Log the plan creation
                    agent.record_memory('PLANNING_EVENT', {
                        'time': self.current_time,
                        'content': plan,
                        'plan_type': 'initial',
                        'location': agent.get_current_location_name()
                    })
                    
        except Exception as e:
            print(f"Error creating plan for {agent.name}: {str(e)}")
            traceback.print_exc()

    def run_simulation_parallel(self):
        """Run the simulation in parallel mode."""
        try:
            # Start at 7 AM on day 1
            self.current_time = 7 if self.current_day == 1 else 0
            
            # Create daily plans for all agents if not already created
            if not self.plans_created:
                self.create_daily_plans_parallel()
            
            # Run each hour of the day
            while self.current_time < self.max_hours:
                # Run each agent's actions for this hour in parallel
                with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                    executor.map(self.run_agent_parallel, self.agents)
                
                # Process end of hour
                self._process_hour_end(self.current_day, self.current_time)
                
                # Increment time
                self.current_time += 1
            
            # Process end of day
            self.process_end_of_day()
            
            # Move to next day if not at max days
            if self.current_day < self.total_days:
                self.current_day += 1
                self.current_time = 0  # Reset to midnight for next day
                self.metrics.new_day(force_day=self.current_day)
                return True
            else:
                print("\nSimulation Complete. Files saved to:")
                print(f"Agent Memories: {self.agent_memories_file}")
                print(f"Metrics: {self.metrics_file}")
                print(f"Daily Summaries: {self.daily_summary_file}")
                print(f"Conversations: {self.conversation_log_file}\n")
                return False
                
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            traceback.print_exc()
            return False

    def extract_target_location_name(self, action_string: str, agent_name: str) -> Optional[str]:
        """Extracts a potential location name from an action string.
        Handles various travel-related phrases and validates against available locations.
        """
        try:
            # Common travel-related phrases
            travel_phrases = [
                "moving to", "going to", "traveling to", "heading to",
                "walking to", "driving to", "visiting"
            ]
            
            # Try each phrase
            for phrase in travel_phrases:
                if phrase in action_string.lower():
                    parts = action_string.lower().split(phrase)
                    if len(parts) > 1:
                        # Extract potential location name
                        potential_location = parts[1].split(" from ")[0].split(" with ")[0].strip()
                        location_name = ' '.join(word.capitalize() for word in potential_location.split())
                        
                        # Validate against available locations
                        if location_name in [loc.name for loc in self.locations.values()]:
                            return location_name
                        else:
                            print(f"[DEBUG] Invalid location '{location_name}' for {agent_name}")
                            return None
            
            return None
            
        except Exception as e:
            print(f"Error extracting location name for {agent_name}: {str(e)}")
            return None

    def get_agent_context(self, agent: 'Agent') -> Dict[str, Any]:
        """Get the current context for an agent."""
        try:
            # Calculate daily income
            daily_income = 0
            if hasattr(agent, 'personal_info') and 'basics' in agent.personal_info:
                income_info = agent.personal_info['basics'].get('income', {})
                if income_info:
                    income_amount = income_info.get('amount', 0)
                    income_type = income_info.get('type', 'hourly')
                    
                    if income_type == 'annual':
                        daily_income = income_amount / 365
                    elif income_type == 'monthly':
                        daily_income = income_amount / 30
                    else:  # hourly
                        work_hours = 8  # Standard work day
                        daily_income = income_amount * work_hours

            # Get basic agent info
            context = {
                'name': agent.name,
                'age': agent.personal_info.get('age', 25) if agent.personal_info else 25,  # Default age if not set
                'occupation': agent.occupation,
                'workplace': agent.workplace,
                'work_schedule_start': agent.work_schedule.get('start_time', 9) if agent.work_schedule else 9,
                'work_schedule_end': agent.work_schedule.get('end_time', 17) if agent.work_schedule else 17,
                'residence': agent.residence,
                'current_location': agent.get_current_location_name(),
                'current_time': agent.current_time,
                'energy_level': agent.energy_level,
                'money': agent.money,
                'daily_income': daily_income,  # Added daily income
                'income_schedule': 'end_of_day',  # Added income schedule
                'grocery_level': agent.grocery_level,
                'recent_activities': agent.get_recent_activities(limit=3),
                'available_locations': [loc.name for loc in self.locations.values()]  # List all locations in town
            }
            
            # Add debug logging
            print(f"[DEBUG] Generated context for {agent.name}:")
            print(f"[DEBUG] Occupation: {context['occupation']}")
            print(f"[DEBUG] Workplace: {context['workplace']}")
            print(f"[DEBUG] Work Schedule: {context['work_schedule_start']}-{context['work_schedule_end']}")
            print(f"[DEBUG] Age: {context['age']}")
            print(f"[DEBUG] Daily Income: ${daily_income:.2f}")
            
            return context
            
        except Exception as e:
            print(f"Error generating context for {agent.name}: {str(e)}")
            traceback.print_exc()
            return {}

    def get_nearby_agents(self, agent: 'Agent') -> List[str]:
        """Get list of agent names that are at the same location as the given agent."""
        try:
            current_location = agent.get_current_location_name()
            if not current_location:
                return []
                
            # Get all agents at the current location
            agents_at_location = []
            for other_agent in self.agents:
                if other_agent.name != agent.name and other_agent.get_current_location_name() == current_location:
                    agents_at_location.append(other_agent.name)
                    
            return agents_at_location
            
        except Exception as e:
            print(f"Error getting nearby agents for {agent.name}: {str(e)}")
            return []

    def run_agent_parallel(self, agent: 'Agent'):
        """Run a single agent's actions in parallel."""
        try:
            with self._conversation_locks[agent.name]:
                # Update agent state
                agent.current_time = self.current_time
                agent.update_state()
                
                # Get context and generate action
                context = self.get_agent_context(agent)
                action = agent.generate_contextual_action(simulation=self, current_time=self.current_time)
                
                # Process the action
                if action:
                    # Extract target location if present
                    target_location = self.extract_target_location_name(action, agent.name)
                    if target_location:
                        # Start travel to target location
                        travel_result = agent.start_travel_to(target_location)
                        if travel_result.startswith("Error"):
                            print(f"Travel error for {agent.name}: {travel_result}")
                            return
                    
                    # Execute travel steps if agent is traveling
                    if agent.is_traveling:
                        message, is_continuing, encounter_info = agent._perform_travel_step()
                        
                        if encounter_info:
                            if encounter_info['type'] == 'agent':
                                # Handle agent encounter
                                other_agent = encounter_info['other_agent']
                                self.handle_agent_conversation(agent, [other_agent])
                            elif encounter_info['type'] == 'location':
                                # Handle location encounter (shop)
                                location_name = encounter_info['location_name']
                                if agent._should_make_purchase_at_location(location_name):
                                    # Generate purchase decision
                                    purchase_decision = agent.generate_structured_purchase_decision(context)
                                    if purchase_decision.get('should_purchase', False):
                                        # Make purchase
                                        purchase_result = agent.make_purchase(
                                            location_name,
                                            purchase_decision.get('item_type', ''),
                                            purchase_decision.get('item_description', '')
                                        )
                                        if purchase_result:
                                            print(f"{agent.name} made a purchase at {location_name}")
                        
                        if not is_continuing:
                            # Travel complete or interrupted
                            agent.is_traveling = False
                            agent.travel_state = None
                    
                    # Handle household interactions if at residence
                    if agent.get_current_location_name() == agent.residence:
                        with self._social_locks[agent.name]:
                            # Get household members
                            household_members = [a for a in self.agents if a.residence == agent.residence and a.name != agent.name]
                            if household_members:
                                # Check if should interact with household members
                                for member in household_members:
                                    should_interact, interaction_context = agent.should_interact_with_household_member(member, self.current_time)
                                    if should_interact:
                                        interaction = agent.handle_household_interaction(self.current_time)
                                        if interaction and interaction != "Quiet time at home due to error":
                                            print(f"\nHousehold interaction for {agent.name}:")
                                            print(interaction)
                    
                    # Handle food needs if not traveling
                    if not agent.is_traveling:
                        self._handle_food_needs(agent, self.current_time)
                    
                    # Handle work if it's work time and not traveling
                    if not agent.is_traveling and agent.is_work_time():
                        self._handle_work(agent, self.current_time)
                    
                    # Update location tracker
                    self.shared_location_tracker.update_agent_position(
                        agent.name,
                        agent.get_current_location_name(),
                        self.town_map.get_coordinates_for_location(agent.get_current_location_name()),
                        self.current_time
                    )
                    
                    # Log the action
                    self.log_conversation(agent.name, action)
            
        except Exception as e:
            print(f"Error running agent {agent.name} in parallel: {str(e)}")
            traceback.print_exc()

    def save_debug_stats(self) -> None:
        """Save debug statistics to file."""
        try:
            debug_file = os.path.join(self.base_dir, f'debug_stats_{self.simulation_id}.jsonl')
            
            # Save daily stats
            daily_stats = {
                'day': self.current_day,
                'stats': self.debug_stats['daily'][self.current_day],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(debug_file, 'a') as f:
                f.write(json.dumps(daily_stats) + '\n')
            
            # Clear the stats for the current day
            self.debug_stats['daily'][self.current_day] = {
                'successful': 0,
                'failed': 0,
                'errors': defaultdict(list)
            }
            
        except Exception as e:
            print(f"Error saving debug stats: {str(e)}")
            traceback.print_exc()

    def process_end_of_day(self):
        """Process end of day activities."""
        try:
            # Create system event memory data
            memory_data = {
                'event_type': 'end_of_day',
                'time': self.current_time,
                'day': self.current_day,
                'description': f"End of day {self.current_day}"
            }
            
            # Record as SYSTEM_EVENT for all agents
            for agent in self.agents:
                agent.record_memory('SYSTEM_EVENT', memory_data)
                
                # Record final state update
                state_data = {
                    'time': self.current_time,
                    'location': agent.get_current_location_name(),
                    'current_activity': 'END_OF_DAY',
                    'energy_level': agent.energy_level,
                    'grocery_level': agent.grocery_level,
                    'money': agent.money
                }
                agent.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
            
            # Process end of day activities
            self._process_hour_end(self.current_day, 23)
            
        except Exception as e:
            print(f"Error processing end of day: {str(e)}")
            traceback.print_exc()

    def handle_agent_conversation(self, initiator: Agent, participants: List[Agent]):
        """Handle a conversation between agents."""
        try:
            # Create interaction memory data
            memory_data = {
                'initiator': initiator.name,
                'participants': [p.name for p in participants],
                'time': self.current_time,
                'location': initiator.get_current_location_name(),
                'interaction_type': 'conversation',
                'activity_type': 'CONVERSATION'
            }
            
            # Record as INTERACTION_EVENT
            initiator.record_memory('INTERACTION_EVENT', memory_data)
            
            # Handle the conversation
            initiator.handle_agent_conversation(initiator, participants)
            
            # Record state update for all participants
            for agent in [initiator] + participants:
                state_data = {
                    'time': self.current_time,
                    'location': agent.get_current_location_name(),
                    'current_activity': 'CONVERSATION',
                    'conversation_partners': [p.name for p in participants if p != agent]
                }
                agent.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
            
        except Exception as e:
            print(f"Error handling agent conversation: {str(e)}")
            traceback.print_exc()

    def _handle_food_needs(self, agent: Agent, current_time: int) -> None:
        """Handle food needs for an agent."""
        try:
            # Check if it's meal time
            is_meal_time, meal_type = agent.is_meal_time(current_time)
            
            if is_meal_time:
                # Check if agent is at home with groceries
                if agent.get_current_location_name() == agent.residence and agent.grocery_level > 0:
                    # Create activity memory data for cooking at home
                    memory_data = {
                        'activity_type': 'DINING',
                        'meal_type': meal_type,
                        'location': agent.residence,
                        'time': current_time,
                        'description': f"Cooking and eating {meal_type} at home",
                        'energy_gain': ENERGY_GAIN_HOME_MEAL,
                        'grocery_used': 1
                    }
                    
                    # Record as ACTIVITY_EVENT
                    agent.record_memory('ACTIVITY_EVENT', memory_data)
                    
                    # Update agent state
                    agent.energy_level = min(agent.energy_level + ENERGY_GAIN_HOME_MEAL, ENERGY_MAX)
                    agent.grocery_level -= 1
                    
                    # Record state update
                    state_data = {
                        'energy_level': agent.energy_level,
                        'grocery_level': agent.grocery_level,
                        'time': current_time,
                        'location': agent.residence,
                        'current_activity': 'DINING'
                    }
                    agent.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
                    
                else:
                    # Find closest food location
                    food_location = agent.find_closest_food_location(current_time)
                    if food_location:
                        # Create travel memory data
                        memory_data = {
                            'target_location': food_location,
                            'start_location': agent.get_current_location_name(),
                            'start_time': current_time,
                            'reason': f"Getting {meal_type}",
                            'activity_type': 'TRAVEL'
                        }
                        
                        # Record as TRAVEL_EVENT
                        agent.record_memory('TRAVEL_EVENT', memory_data)
                        
                        # Start travel to food location
                        agent.start_travel_to(food_location)
            
        except Exception as e:
            print(f"Error handling food needs: {str(e)}")
            traceback.print_exc()

    def _handle_travel(self, agent: Agent, target_location: str) -> None:
        """Handle travel for an agent."""
        try:
            # Create travel memory data
            memory_data = {
                'target_location': target_location,
                'start_location': agent.get_current_location_name(),
                'start_time': self.current_time,
                'status': 'started',
                'activity_type': 'TRAVEL'
            }
            
            # Record as TRAVEL_EVENT
            agent.record_memory('TRAVEL_EVENT', memory_data)
            
            # Start travel
            agent.start_travel_to(target_location)
            
            # Record state update
            state_data = {
                'time': self.current_time,
                'location': agent.get_current_location_name(),
                'current_activity': 'TRAVEL',
                'travel_target': target_location
            }
            agent.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
            
        except Exception as e:
            print(f"Error handling travel: {str(e)}")
            traceback.print_exc()

    def _handle_work(self, agent: Agent, current_time: int) -> None:
        """Handle work for an agent."""
        try:
            # Create activity memory data
            memory_data = {
                'activity_type': 'WORK',
                'location': agent.workplace,
                'time': current_time,
                'description': f"Working at {agent.workplace}",
                'energy_cost': ENERGY_COST_WORK_HOUR
            }
            
            # Record as ACTIVITY_EVENT
            agent.record_memory('ACTIVITY_EVENT', memory_data)
            
            # Update agent state
            agent.energy_level = max(agent.energy_level - ENERGY_COST_WORK_HOUR, ENERGY_MIN)
            
            # Record state update
            state_data = {
                'energy_level': agent.energy_level,
                'time': current_time,
                'location': agent.workplace,
                'current_activity': 'WORK'
            }
            agent.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
            
        except Exception as e:
            print(f"Error handling work: {str(e)}")
            traceback.print_exc()

    def _handle_rest(self, agent: Agent, current_time: int) -> None:
        """Handle rest for an agent."""
        try:
            # Create activity memory data
            memory_data = {
                'activity_type': 'RESTING',
                'location': agent.get_current_location_name(),
                'time': current_time,
                'description': "Resting to recover energy",
                'energy_gain': ENERGY_GAIN_NAP
            }
            
            # Record as ACTIVITY_EVENT
            agent.record_memory('ACTIVITY_EVENT', memory_data)
            
            # Update agent state
            agent.energy_level = min(agent.energy_level + ENERGY_GAIN_NAP, ENERGY_MAX)
            
            # Record state update
            state_data = {
                'energy_level': agent.energy_level,
                'time': current_time,
                'location': agent.get_current_location_name(),
                'current_activity': 'RESTING'
            }
            agent.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
            
        except Exception as e:
            print(f"Error handling rest: {str(e)}")
            traceback.print_exc()

    def save_conversation_logs(self) -> None:
        """Save conversation logs to file."""
        try:
            if hasattr(self, 'conversation_log_file'):
                # Get all conversation memories
                conversation_memories = []
                for agent in self.agents:
                    agent_conversations = self.memory_mgr.get_recent_memories(
                        agent.name,
                        'CONVERSATION_EVENT',
                        limit=1000
                    )
                    if agent_conversations:
                        conversation_memories.extend(agent_conversations)
                
                # Sort by timestamp
                conversation_memories.sort(key=lambda x: x.get('time', 0))
                
                # Write to file
                with open(self.conversation_log_file, 'w') as f:
                    for memory in conversation_memories:
                        f.write(json.dumps(memory) + '\n')
                
                print(f"Conversation logs saved to: {self.conversation_log_file}")
            
        except Exception as e:
            print(f"Error saving conversation logs: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        print("\n=== Starting Simulation ===")
        print(f"Real-world start time: {datetime.now().strftime('%H:%M:%S')}")
        
        sim_data = initialize_simulation()
        if not sim_data:
            raise ValueError("Failed to initialize simulation. Exiting.")
        
        simulation = Simulation(
            config=sim_data['settings']
        )
        
        # Run simulation based on parallel setting
        if sim_data['settings'].get('parallel', False):
            print("\n=== Running Simulation in Parallel Mode ===")
            simulation.run_simulation()
        else:
            print("\n=== Running Simulation in Sequential Mode ===")
            simulation.run_simulation_sequentially()
        
        print(f"\nReal-world end time: {datetime.now().strftime('%H:%M:%S')}")
        print("=== Simulation Complete ===\n")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        traceback.print_exc()
    finally:
        # Save final state
        if 'simulation' in locals():
            simulation.save_state(is_final_save=True)
            simulation.save_conversation_logs()