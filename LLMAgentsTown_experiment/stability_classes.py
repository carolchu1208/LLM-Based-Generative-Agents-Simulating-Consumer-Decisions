from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, Union
import json
from datetime import datetime, time
import traceback
import collections # For deque in BFS
import re
from queue import Queue
import random
import logging
import threading
from dataclasses import dataclass
from collections import defaultdict, deque
import os
import uuid
import time as time_module  # Rename to avoid conflicts
import hashlib

from simulation_constants import (
    SIMULATION_SETTINGS, ACTIVITY_TYPES, MEMORY_TYPES,
    TimeManager, SimulationError, AgentError, LocationError,
    MemoryError, MetricsError, ErrorHandler, ThreadSafeBase,
    ENERGY_MAX, ENERGY_MIN, ENERGY_COST_PER_STEP, ENERGY_DECAY_PER_HOUR,
    ENERGY_COST_WORK_HOUR, ENERGY_COST_PER_HOUR_TRAVEL, ENERGY_COST_PER_HOUR_IDLE,
    ENERGY_GAIN_RESTAURANT_MEAL, ENERGY_GAIN_SNACK, ENERGY_GAIN_HOME_MEAL,
    ENERGY_GAIN_SLEEP, ENERGY_GAIN_NAP, ENERGY_GAIN_CONVERSATION,
    ENERGY_THRESHOLD_LOW, MemoryEvent, SharedMemoryBuffer, shared_memory_buffer
)
from Stability_Memory_Manager import MemoryManager

if TYPE_CHECKING:
    from typing import Type
    Agent_T = Type['Agent']
    from Stability_Ollama_MainSimulation import Simulation

class TownMap:
    def __init__(self, world_locations_data: Dict[str, List[int]], travel_paths_data: List[List[List[int]]]):
        """Initialize TownMap with locations and paths.
        
        Args:
            world_locations_data: Dictionary mapping location names to [x,y] coordinates
            travel_paths_data: List of paths, where each path is a list of [x,y] coordinates
        """
        self.world_locations: Dict[str, Tuple[int, int]] = { 
            name: tuple(coords) for name, coords in world_locations_data.items()
        }
        self.travel_paths_data = travel_paths_data
        
        # Print debug info about paths
        print("\nDEBUG: Initialized TownMap with paths:")
        for path in travel_paths_data:
            print(f"Path: {path}")
        
        # Print debug info about locations
        print("\nDEBUG: World locations:")
        for name, coord in self.world_locations.items():
            print(f"{name} at {coord}")

    def find_path(self, start_coord: Tuple[int, int], end_coord: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path between two coordinates using grid-based BFS.
        Returns None if no path exists."""
        if start_coord == end_coord:
            return [start_coord]
            
        # Define possible moves (up, down, left, right)
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Check if a coordinate is on any defined path
        def is_on_path(coord: Tuple[int, int]) -> bool:
            for path in self.travel_paths_data:
                if list(coord) in path:
                    return True
            return False
            
        # Initialize BFS
        queue = collections.deque([[start_coord]])
        visited = {start_coord}
        
        print(f"\nDEBUG: Starting pathfinding from {start_coord} to {end_coord}")
        print(f"DEBUG: Start coordinate on path: {is_on_path(start_coord)}")
        print(f"DEBUG: End coordinate on path: {is_on_path(end_coord)}")
        
        while queue:
            path = queue.popleft()
            current = path[-1]
            
            # Try each possible move
            for dx, dy in moves:
                next_coord = (current[0] + dx, current[1] + dy)
                
                # Check if this move is valid (on a defined path and not visited)
                if next_coord not in visited and is_on_path(next_coord):
                    if next_coord == end_coord:
                        full_path = path + [next_coord]
                        print(f"DEBUG: Found path: {full_path}")
                        return full_path
                    visited.add(next_coord)
                    queue.append(path + [next_coord])
        
        print(f"DEBUG: No path found between {start_coord} and {end_coord}")
        print("DEBUG: Available paths:", self.travel_paths_data)
        return None

    def get_coordinates_for_location(self, location_name: str) -> Optional[Tuple[int, int]]:
        """Get the coordinates for a named location."""
        return self.world_locations.get(location_name)

    def get_location_name_at_coord(self, coord: Tuple[int, int]) -> Optional[str]:
        """Get the location name at given coordinates."""
        for name, loc_coord in self.world_locations.items():
            if loc_coord == coord:
                return name
        return None

class Location:
    """Location class for managing areas in the town"""
    def __init__(self, name: str, type: str, capacity: int = 10, grid_coordinate: Optional[Tuple[int, int]] = None):
        self.name = name
        self.type = type
        self.capacity = capacity
        self.agents: List['Agent'] = []  # List of agents currently at this location
        self.queue: Queue['Agent'] = Queue()  # Queue of agents waiting to enter
        self.description = ""
        self.base_price = 0.0
        self.hours = {"open": 9, "close": 17}  # Default business hours using open/close format
        self.discounts = {}  # Time-based discounts
        self.grid_coordinate = grid_coordinate  # Store the grid coordinate
        
    def set_hours(self, hours_dict: Dict[str, Any]) -> None:
        """Set the operating hours for the location"""
        if hours_dict.get('always_open', False):
            self.hours = {'always_open': True}
        else:
            # Ensure we have both open and close times
            self.hours = {
                'open': hours_dict.get('open', hours_dict.get('start', 9)),
                'close': hours_dict.get('close', hours_dict.get('end', 17))
            }
            print(f"DEBUG: Setting hours for {self.name}: {self.hours}")

    def is_open(self, current_hour: int) -> bool:
        """Check if location is open at the given hour"""
        try:
            # Handle 24/7 locations
            if self.hours.get('always_open', False):
                return True
            
            # Get opening and closing hours
            open_hour = self.hours.get('open', 9)
            close_hour = self.hours.get('close', 17)
            
            # Check if current hour is within operating hours
            if close_hour > open_hour:
                # Normal case (e.g. 8:00-22:00)
                return open_hour <= current_hour < close_hour
            else:
                # Overnight case (e.g. 22:00-6:00)
                return current_hour >= open_hour or current_hour < close_hour
                
        except Exception as e:
            print(f"Error checking if location is open: {str(e)}")
            return False  # Default to closed on error
            
    def is_full(self) -> bool:
        """Check if location is at capacity"""
        return len(self.agents) >= self.capacity
        
    def add_agent(self, agent: 'Agent') -> bool:
        """Add agent to location"""
        if not self.is_full():
            self.agents.append(agent)
            return True
        return False
        
    def remove_agent(self, agent: 'Agent') -> None:
        """Remove agent from location"""
        self.agents = [a for a in self.agents if a.name != agent.name]
        
    def add_to_queue(self, agent: 'Agent') -> str:
        """Add an agent to the location's queue.
        
        Args:
            agent: The agent to add to the queue
            
        Returns:
            str: Status message about the queue addition
        """
        if agent.name not in self.queue:
            self.queue.append(agent.name)
            return f"{agent.name} added to queue at {self.name}"
        return f"{agent.name} already in queue at {self.name}"

    def get_current_price(self, current_time: int) -> float:
        """Get current price with any active discounts"""
        if not self.discounts or not isinstance(self.discounts, dict):
            return self.base_price
        
        # Use simulation day directly instead of calculating from hours
        current_day = (current_time // 24) + 1
        discount_days = self.discounts.get('days', []) if isinstance(self.discounts, dict) else []
        
        if current_day in discount_days:
            discount_value = self.discounts.get('value', 0) if isinstance(self.discounts, dict) else 0
            discounted_price = self.base_price * (1 - discount_value / 100)
            print(f"DEBUG: Applying {discount_value}% discount on day {current_day}. Base price: ${self.base_price}, Discounted: ${discounted_price}")
            return discounted_price
        
        print(f"DEBUG: No discount on day {current_day}. Regular price: ${self.base_price}")
        return self.base_price

    def process_queue(self) -> Optional[str]:
        """Process waiting queue when space becomes available"""
        while not self.queue.empty():
            next_agent = self.queue.get()
            if self.add_agent(next_agent):
                return f"{next_agent.name} has entered {self.name} from the queue"
        return None

class ConversationState:
    def __init__(self):
        self.participants = []
        self.location = None
        self.start_time = 0
        self.dialogue = []
        self.active = True
        self.history = []

    def add_turn(self, speaker: str, content: str):
        """Add a turn to the conversation."""
        turn = {
            'speaker': speaker,
            'content': content,
            'time': self.start_time
        }
        self.dialogue.append(turn)
        self.history.append(f"{speaker}: {content}")

    def should_continue(self) -> bool:
        """Check if the conversation should continue."""
        return self.active and len(self.dialogue) < 10

    def get_last_turns(self, num_turns: int = 3) -> List[str]:
        """Get the last N turns of the conversation."""
        return self.history[-num_turns:] if self.history else []

class Agent:
    def __init__(self, name: str, age: int, occupation: str, residence: str, workplace: str, 
                 work_schedule: Dict[str, int], memory_mgr: 'MemoryManager'):
        """Initialize an agent with basic attributes and memory manager."""
        self.name = name
        self.age = age
        self.occupation = occupation
        self.residence = residence
        self.workplace = workplace
        self.work_schedule = work_schedule
        self.memory_mgr = memory_mgr
        
        # Initialize state
        self.energy_level = ENERGY_MAX
        self.grocery_level = 0
        self.money = 1000  # Starting money
        self.current_location = None  # Will be set to Location object in initialize_personal_context
        self.current_activity = 'INITIALIZATION'
        self.current_time = 0
        self.is_traveling = False
        self.travel_state = None
        self.simulation = None  # Will be set by simulation
        self.locations = {}  # Will be populated with Location objects
        
        # Initialize memory
        self.initialize_memory()

    def generate_contextual_action(self, simulation: 'Simulation', current_time: int) -> str:
        """Generate an action based on the current context and state."""
        try:
            # Store simulation reference
            self.simulation = simulation
            self.current_time = current_time
            
            # Get current hour
            current_hour = current_time % 24
            
            # Check if agent is traveling
            if self.is_traveling:
                return f"Continuing travel to {self.travel_state['target_location']}"
            
            # Check if at work location during work hours
            if self.is_work_time() and self.get_current_location_name() == self.workplace:
                return "Working at my job"
            
            # Check if at residence during sleep hours
            if TimeManager.is_sleep_time(current_time) and self.get_current_location_name() == self.residence:
                return "Sleeping at home"
            
            # Check if needs food
            if self.needs_food(current_time):
                # Find closest food location
                food_location = self.find_closest_food_location(current_hour)
                if food_location:
                    return f"Traveling to {food_location} to get food"
            
            # Check if needs groceries
            if self.needs_groceries(current_hour):
                # Find closest grocery store
                grocery_location = self.find_closest_grocery_store(current_hour)
                if grocery_location:
                    return f"Traveling to {grocery_location} to buy groceries"
            
            # Check if at residence and should interact with household
            if self.get_current_location_name() == self.residence:
                household_members = [a for a in simulation.agents if a.residence == self.residence and a.name != self.name]
                if household_members:
                    for member in household_members:
                        should_interact, _ = self.should_interact_with_household_member(member, current_time)
                        if should_interact:
                            return f"Interacting with household member {member.name}"
            
            # Check if should make a meal
            is_meal_time, meal_type = TimeManager.is_meal_time(current_time)
            if is_meal_time and self.get_current_location_name() == self.residence and self.grocery_level > 0:
                return f"Making {meal_type} at home"
            
            # Default to resting if no other actions needed
            if self.energy_level < ENERGY_MAX:
                return "Resting to recover energy"
            
            # If all else fails, stay at current location
            return f"Staying at {self.get_current_location_name()}"
            
        except Exception as e:
            print(f"Error generating contextual action for {self.name}: {str(e)}")
            traceback.print_exc()
            return f"Error generating action: {str(e)}"

    def initialize_memory(self):
        """Initialize agent's memory with basic information."""
        try:
            # Record personal information
            personal_data = {
                'name': self.name,
                'age': self.age,
                'occupation': self.occupation,
                'residence': self.residence,
                'workplace': self.workplace,
                'work_schedule': self.work_schedule,
                'time': 0,
                'day': 1
            }
            self.record_memory('AGENT_STATE_UPDATE_EVENT', personal_data)
            
            # Record initial location
            location_data = {
                'location': self.residence,
                'time': 0,
                'day': 1,
                'activity_type': 'INITIALIZATION'
            }
            self.record_memory('LOCATION_EVENT', location_data)
            
            # Record initial state
            state_data = {
                'energy_level': self.energy_level,
                'grocery_level': self.grocery_level,
                'money': self.money,
                'time': 0,
                'day': 1,
                'location': self.residence,
                'current_activity': 'INITIALIZATION'
            }
            self.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
            
        except Exception as e:
            print(f"Error initializing memory for {self.name}: {str(e)}")
            traceback.print_exc()

    def record_memory(self, memory_type_key: str, memory_data: Dict[str, Any]) -> None:
        """Record a memory with proper validation and metadata."""
        try:
            # Validate memory type
            if memory_type_key not in MEMORY_TYPES:
                raise ValueError(f"Invalid memory type: {memory_type_key}")
            
            # Add required metadata
            memory_data.update({
                'agent_name': self.name,
                'location': self.get_current_location_name(),
                'time': memory_data.get('time', 0),
                'day': memory_data.get('day', 1)
            })
            
            # Create memory event
            memory_event = MemoryEvent(
                memory_type=memory_type_key,  # Changed from event_type to memory_type
                data=memory_data
            )
            
            # Add to memory manager
            self.memory_mgr.add_memory(memory_event)
            
        except Exception as e:
            print(f"Error recording memory for {self.name}: {str(e)}")
            traceback.print_exc()

    def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update agent state and record the change."""
        try:
            # Update state attributes
            if 'energy_level' in new_state:
                self.energy_level = new_state['energy_level']
            if 'grocery_level' in new_state:
                self.grocery_level = new_state['grocery_level']
            if 'money' in new_state:
                self.money = new_state['money']
            if 'current_location' in new_state:
                self.current_location = new_state['current_location']
            if 'current_activity' in new_state:
                self.current_activity = new_state['current_activity']
            
            # Record state update
            state_data = {
                'energy_level': self.energy_level,
                'grocery_level': self.grocery_level,
                'money': self.money,
                'location': self.get_current_location_name(),
                'current_activity': self.current_activity,
                'time': new_state.get('time', 0),
                'day': new_state.get('day', 1)
            }
            self.record_memory('AGENT_STATE_UPDATE_EVENT', state_data)
            
        except Exception as e:
            print(f"Error updating state for {self.name}: {str(e)}")
            traceback.print_exc()

    def start_travel_to(self, target_location_name: str, desired_arrival_time: Optional[int] = None) -> str:
        """Start travel to a target location."""
        try:
            # Get current location coordinates
            current_location = self.get_current_location_name()
            current_coords = self.simulation.town_map.get_coordinates_for_location(current_location)
            target_coords = self.simulation.town_map.get_coordinates_for_location(target_location_name)
            
            if not current_coords or not target_coords:
                return f"Error: Could not find coordinates for {current_location} or {target_location_name}"
            
            # Calculate path
            path = self.simulation.town_map.find_path(current_coords, target_coords)
            if not path:
                return f"Error: No valid path found from {current_location} to {target_location_name}"
            
            # Initialize travel state
            self.travel_state = {
                'target_location': target_location_name,
                'path': path,
                'current_step': 0,
                'start_time': self.current_time,
                'desired_arrival_time': desired_arrival_time,
                'status': 'in_progress'
            }
            
            # Set traveling flag
            self.is_traveling = True
            
            # Create travel memory data
            memory_data = {
                'target_location': target_location_name,
                'start_location': current_location,
                'start_time': self.current_time,
                'path_length': len(path),
                'status': 'started',
                'activity_type': 'TRAVEL'
            }
            
            # Record as TRAVEL_EVENT
            self.record_memory('TRAVEL_EVENT', memory_data)
            
            # Update state
            self.current_activity = 'TRAVEL'
            self.update_state({
                'current_activity': 'TRAVEL',
                'time': self.current_time
            })
            
            return f"Starting travel to {target_location_name}"
            
        except Exception as e:
            print(f"Error starting travel for {self.name}: {str(e)}")
            traceback.print_exc()
            return f"Error starting travel: {str(e)}"

    def make_meal(self, meal_type: str) -> bool:
        """Make a meal and record the activity."""
        try:
            if self.grocery_level <= 0:
                return False
            
            # Create activity memory data
            memory_data = {
                'activity_type': 'DINING',
                'meal_type': meal_type,
                'location': self.get_current_location_name(),
                'time': 0,  # Will be updated by simulation
                'day': 1,   # Will be updated by simulation
                'description': f"Cooking and eating {meal_type}",
                'energy_gain': ENERGY_GAIN_HOME_MEAL,
                'grocery_used': 1
            }
            
            # Record as ACTIVITY_EVENT
            self.record_memory('ACTIVITY_EVENT', memory_data)
            
            # Update state
            self.energy_level = min(self.energy_level + ENERGY_GAIN_HOME_MEAL, ENERGY_MAX)
            self.grocery_level -= 1
            self.current_activity = 'DINING'
            
            self.update_state({
                'energy_level': self.energy_level,
                'grocery_level': self.grocery_level,
                'current_activity': 'DINING',
                'time': 0,  # Will be updated by simulation
                'day': 1    # Will be updated by simulation
            })
            
            return True
            
        except Exception as e:
            print(f"Error making meal for {self.name}: {str(e)}")
            traceback.print_exc()
            return False

    def _get_memory_version(self, memory_type: str, data: Dict) -> int:
        """Get version number for memory to prevent conflicts."""
        try:
            # Get existing memories of this type
            existing_memories = self.memory_mgr.get_recent_memories(
                self.name, 
                memory_type, 
                limit=1
            )
            
            # Start with version 1 if no existing memories
            if not existing_memories:
                return 1
                
            # Increment version from last memory
            last_version = existing_memories[0].get('version', 0)
            return last_version + 1
            
        except Exception as e:
            print(f"Error getting memory version: {str(e)}")
            return 1

    def _hash_memory_content(self, data: Dict) -> str:
        """Generate a hash of memory content for duplicate detection."""
        try:
            import hashlib
            import json
            
            # Create a copy of the data to modify
            data_copy = data.copy()
            
            # Convert Location objects to their names
            if 'location' in data_copy and isinstance(data_copy['location'], Location):
                data_copy['location'] = data_copy['location'].name
            
            # Create a stable string representation of the data
            content_str = json.dumps({
                'type': data_copy.get('type', ''),
                'time': data_copy.get('time', 0),
                'location': data_copy.get('location', ''),
                'participants': sorted(data_copy.get('participants', [])),
                'content': data_copy.get('content', '')
            }, sort_keys=True)
            
            # Generate hash
            return hashlib.md5(content_str.encode()).hexdigest()
            
        except Exception as e:
            print(f"Error hashing memory content: {str(e)}")
            return ''

    def _is_duplicate_memory(self, memory_type: str, data: Dict) -> bool:
        """Check if this memory is a duplicate of an existing one."""
        try:
            # Get recent memories of this type
            recent_memories = self.memory_mgr.get_recent_memories(
                self.name,
                memory_type,
                limit=5  # Check last 5 memories
            )
            
            # Check for duplicates based on content hash
            content_hash = data.get('content_hash', '')
            if not content_hash:
                return False
                
            for memory in recent_memories:
                if memory.get('content_hash') == content_hash:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error checking for duplicate memory: {str(e)}")
            return False

    # Add an alias for backward compatibility
    log_memory = record_memory

    def update_state(self):
        """Update agent state and log changes."""
        state_data = {
            'time': self.current_time,
            'location': self.get_current_location_name(),
            'energy': self.energy_level,
            'traveling': self.is_traveling
        }
        self.log_memory('AGENT_STATE_UPDATE_EVENT', state_data)

    def should_interact_with_household_member(self, member: 'Agent', current_time: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if the agent should interact with a household member.
        
        Args:
            member: The household member to interact with
            current_time: The current time
            
        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - bool: Whether to interact
                - Dict containing:
                    - 'reason': String explaining the decision
                    - 'tags': List of relevant tags
                    - 'context': Relevant context for the decision
        """
        try:
            # Initialize decision info
            decision_info = {
                'reason': '',
                'tags': ['household'],
                'context': {
                    'time': current_time,
                    'location': self.get_current_location_name(),
                    'energy_level': self.energy_level,
                    'relationship': self.get_relationship_with(member.name)
                }
            }

            # Check if we're already in a conversation
            if self.is_in_conversation:
                decision_info.update({
                    'reason': "Already in a conversation",
                    'tags': ['state_blocked']
                })
                print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Household interaction blocked with {member.name}: Already in conversation")
                return False, decision_info

            # Check for recent conversation cooldown
            recent_conversations = self.get_recent_memories('conversation', limit=5)
            if any(conv.get('participants', []).count(member.name) > 0 
                  and current_time - conv.get('time', 0) < 1  # 1 hour cooldown for household members
                  for conv in recent_conversations):
                decision_info.update({
                    'reason': "Recent conversation cooldown",
                    'tags': ['cooldown', 'recent_interaction']
                })
                print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Household interaction blocked with {member.name}: Recent conversation cooldown")
                return False, decision_info

            # Check if it's a good time for household interaction
            current_hour = current_time % 24
            if 6 <= current_hour <= 8 or 17 <= current_hour <= 21:  # Morning or evening hours
                decision_info.update({
                    'reason': "Good time for household interaction",
                    'tags': ['timing', 'household_routine']
                })
                print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Household interaction approved with {member.name}: Good timing")
                return True, decision_info

            # Default case: no compelling reason to interact
            decision_info.update({
                'reason': "Not a good time for household interaction",
                'tags': ['timing']
            })
            print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Household interaction blocked with {member.name}: Not a good time")
            return False, decision_info

        except Exception as e:
            error_info = {
                'reason': f"Error: {str(e)}",
                'tags': ['error'],
                'context': {
                    'error': str(e),
                    'time': current_time,
                    'location': self.get_current_location_name(),
                    'member': member.name if hasattr(member, 'name') else 'unknown'
                }
            }
            print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [ERROR] Error in household interaction decision: {str(e)}")
            return False, error_info

    def handle_household_interaction(self, current_time: int) -> str:
        """Handle interaction with household members at the current location."""
        try:
            # Get household members at current location
            household_members = [
                agent for agent in self.all_agents_list_for_perception
                if agent.name in [m['name'] for m in self.household_members]
                and agent.get_current_location_name() == self.get_current_location_name()
            ]
            
            if not household_members:
                return None

            # Check if we're already waiting for someone
            if self.waiting_for_conversation:
                # Check if the agent we're waiting for is now available
                for member in household_members:
                    if member.name == self.waiting_for_conversation:
                        # Check if they're available for conversation
                        should_interact, decision_info = self.should_interact_with_household_member(member, current_time)
                        if should_interact:
                            # Start the conversation
                            self.waiting_for_conversation = None
                            self.waiting_since = None
                            return self.handle_agent_conversation(self, [self, member])
                        else:
                            # Check if we've been waiting too long (e.g., 2 hours)
                            if current_time - self.waiting_since > 2:
                                self.waiting_for_conversation = None
                                self.waiting_since = None
                                return f"{self.name} gave up waiting for {member.name} to be available"
                            return f"{self.name} is waiting for {member.name} to be available"
                return None

            # Check if any household member is waiting for us
            for member in household_members:
                if member.waiting_for_conversation == self.name:
                    # Check if we're available for conversation
                    should_interact, decision_info = member.should_interact_with_household_member(self, current_time)
                    if should_interact:
                        # Start the conversation
                        member.waiting_for_conversation = None
                        member.waiting_since = None
                        return member.handle_agent_conversation(member, [member, self])
                    else:
                        return f"{member.name} is waiting for {self.name} to be available"

            # Normal conversation initiation
            for member in household_members:
                should_interact, decision_info = self.should_interact_with_household_member(member, current_time)
                if should_interact:
                    if not member.is_in_conversation:
                        return self.handle_agent_conversation(self, [self, member])
                    else:
                        # Start waiting for the member to be available
                        self.waiting_for_conversation = member.name
                        self.waiting_since = current_time
                        return f"{self.name} is waiting for {member.name} to be available"

            return None

        except Exception as e:
            print(f"Error in household interaction for {self.name}: {str(e)}")
            traceback.print_exc()
            return None

    def update_energy_from_action(self, action_str: str, context: Dict = None):
        """Update energy level based on structured action."""
        try:
            # Parse the action into structured format
            parsed_action = self.parse_structured_action(action_str)
            action = parsed_action.get('action', '').lower()
            location = parsed_action.get('location', '').lower()
            
            # Determine energy change based on structured action
            if location != 'stay':
                # Any movement to a new location costs travel energy
                energy_change = -ENERGY_COST_PER_HOUR_TRAVEL
            elif 'work' in action or 'supervise' in action or 'manage' in action:
                energy_change = -ENERGY_COST_WORK_HOUR
            elif 'sleep' in action:
                energy_change = ENERGY_GAIN_SLEEP
            elif 'nap' in action or 'rest' in action:
                energy_change = ENERGY_GAIN_NAP
            elif any(word in action for word in ['eat', 'meal', 'breakfast', 'lunch', 'dinner']):
                if 'home' in action or self.get_current_location_name().lower() == self.residence.lower():
                    energy_change = ENERGY_GAIN_HOME_MEAL
                else:
                    energy_change = ENERGY_GAIN_RESTAURANT_MEAL
            elif any(word in action for word in ['snack', 'coffee', 'drink']):
                energy_change = ENERGY_GAIN_SNACK
            elif any(word in action for word in ['talk', 'conversation', 'discuss']):
                energy_change = ENERGY_GAIN_CONVERSATION
            else:
                energy_change = -ENERGY_COST_PER_HOUR_IDLE
            
            # Update energy level
            self.energy_level = max(ENERGY_MIN, min(ENERGY_MAX, self.energy_level + energy_change))
            
            # Log the energy update
            self.record_memory('AGENT_STATE_UPDATE_EVENT', {
                'time': self.current_time,
                'energy_level': self.energy_level,
                'energy_change': energy_change,
                'action': action_str,
                'location': self.get_current_location_name(),
                'parsed_action': parsed_action
            })
            
        except Exception as e:
            print(f"Error updating energy from action: {str(e)}")
            traceback.print_exc()
            # Fallback to basic energy update
            self._fallback_energy_update(action_str)

    def initialize_personal_context(self, person_data: Dict):
        """Initialize personal context for the agent."""
        try:
            # Store personal data
            self.personal_info = person_data
            
            # Set current location to residence Location object
            if self.residence in self.locations:
                self.current_location = self.locations[self.residence]
            else:
                print(f"Warning: Residence {self.residence} not found in locations for {self.name}")
                # Create a temporary Location object for residence
                self.current_location = Location(
                    name=self.residence,
                    type="residence",
                    capacity=1
                )
                self.locations[self.residence] = self.current_location
            
            # Initialize other personal context data
            self.name = person_data.get('basics', {}).get('name', self.name)
            self.age = person_data.get('basics', {}).get('age', self.age)
            self.occupation = person_data.get('basics', {}).get('occupation', self.occupation)
            self.residence = person_data.get('basics', {}).get('residence', self.residence)
            self.workplace = person_data.get('basics', {}).get('workplace', self.workplace)
            self.work_schedule = person_data.get('basics', {}).get('income', {}).get('schedule', self.work_schedule)
            
        except Exception as e:
            print(f"Error initializing personal context for {self.name}: {str(e)}")
            traceback.print_exc()

    def _calculate_daily_wage(self) -> float:
        """Calculate daily wage based on income type and work schedule."""
        if not self.income_amount:
            return 0.0
            
        # Calculate daily income for wage calculation
        if self.income_type == 'annual':
            return self.income_amount / 365
        elif self.income_type == 'monthly':
            return self.income_amount / 30
        else:  # hourly
            work_hours = 8  # Standard work day
            if self.work_schedule:
                # Calculate actual work hours from schedule
                start_time = self.work_schedule.get('start', 9)
                end_time = self.work_schedule.get('end', 17)
                work_hours = max(0, end_time - start_time)
            return self.income_amount * work_hours

    def did_work_today(self, current_time: int) -> bool:
        """Check if the agent worked during the previous day."""
        # Simple check: if they have a workplace and work schedule, they worked
        # More sophisticated logic could track actual work attendance
        if not self.workplace or not self.work_schedule:
            return False
            
        # For now, assume they worked if they have work schedule
        # TODO: Could track actual work attendance from memory logs
        return True

    def get_social_context(self, nearby_agents: List[str]) -> Dict:
        """Get social context for interactions including relationships"""
        context = {
            'nearby_people': [],
            'household_members_present': [],
            'relationship_context': {}
        }
        
        for agent_name in nearby_agents:
            person_context = {
                'name': agent_name,
                'relationship': 'stranger'  # Default relationship
            }
            
            # First check relationships dictionary
            if agent_name in self.relationships:
                person_context['relationship'] = self.relationships[agent_name]
                
                # Add to relationship context
                context['relationship_context'][agent_name] = {
                    'type': self.relationships[agent_name],
                    'duration': 'unknown'  # Default duration
                }
            
            # Then check if this is a household member (this will override the relationship if found)
            for member in self.household_members:
                if member['name'] == agent_name:
                    person_context.update({
                        'relationship': member['relationship_type'],
                        'duration': member.get('relationship_duration', 'unknown'),
                        'living_arrangement': member.get('living_arrangement', 'same_household')
                    })
                    context['household_members_present'].append(agent_name)
                    
                    # Update relationship context with more details
                    context['relationship_context'][agent_name] = {
                        'type': member['relationship_type'],
                        'duration': member.get('relationship_duration', 'unknown'),
                        'living_arrangement': member.get('living_arrangement', 'same_household')
                    }
                    break
            
            context['nearby_people'].append(person_context)
            
        return context

    def build_social_interaction_context(self, nearby_agents: Union[List['Agent'], List[str]]) -> Dict:
        """Build unified context for social interactions.
        
        Args:
            nearby_agents: List of either Agent objects or agent names
            
        Returns:
            Dictionary containing social interaction context
        """
        try:
            # Convert string names to Agent objects if needed
            agent_objects = []
            for agent in nearby_agents:
                if isinstance(agent, str):
                    # Find the agent object from all_agents_list_for_perception
                    matching_agents = [a for a in self.all_agents_list_for_perception if a.name == agent]
                    if matching_agents:
                        agent_objects.append(matching_agents[0])
                else:
                    agent_objects.append(agent)
            
            return {
                'location': self.get_current_location_name(),
                'location_type': self.current_location.type if self.current_location else "unknown",
                'time': self.current_time,
                'nearby_agents': [{
                    'name': agent.name,
                    'relationship': self.get_relationship_with(agent.name),
                    'shared_history': self.get_recent_shared_activities(agent.name),
                    'current_activity': agent.current_activity,
                    'work_context': {
                        'workplace': agent.workplace if hasattr(agent, 'workplace') else None,
                        'schedule': agent.work_schedule if hasattr(agent, 'work_schedule') else None,
                        'occupation': agent.occupation if hasattr(agent, 'occupation') else None
                    },
                    'living_context': {
                        'is_household_member': agent.name in [m['name'] for m in self.household_members],
                        'residence': agent.residence,
                        'relationship_type': next((m['relationship_type'] for m in self.household_members if m['name'] == agent.name), None)
                    }
                } for agent in agent_objects],
                'personal_context': {
                    'energy': self.energy_level,
                    'current_activity': self.current_activity,
                    'work_status': {
                        'workplace': self.workplace,
                        'schedule': self.work_schedule,
                        'occupation': self.occupation
                    } if hasattr(self, 'work_schedule') else None,
                    'recent_activities': self.get_recent_activities(limit=3),
                    'daily_plan': self.daily_plan,
                    'meals_today': self.meals_today
                }
            }
        except Exception as e:
            print(f"Error building social context for {self.name}: {str(e)}")
            traceback.print_exc()
            return {}

    def get_recent_shared_activities(self, other_agent_name: str, time_window_hours: int = 24) -> List[str]:
        """Get recent shared activities with another agent.
        
        Args:
            other_agent_name: Name of the other agent
            time_window_hours: How far back to look for activities (default 24 hours)
            
        Returns:
            List of shared activity descriptions
        """
        try:
            # Get cutoff time
            cutoff_time = self.current_time - time_window_hours
            
            # Get memories of various types
            shared_activities = []
            
            # Get conversation memories
            conv_mems = self.memory_mgr.get_recent_memories(
                self.name,
                MEMORY_TYPES['CONVERSATION_LOG_EVENT'],
                limit=10
            )
            
            # Get activity memories
            activity_mems = self.memory_mgr.get_recent_memories(
                self.name,
                MEMORY_TYPES['ACTIVITY_EVENT'],
                limit=10
            )
            
            # Get location change memories
            location_mems = self.memory_mgr.get_recent_memories(
                self.name,
                MEMORY_TYPES['LOCATION_CHANGE_EVENT'],
                limit=10
            )
            
            # Process conversation memories
            for mem in conv_mems:
                data = mem.get('data', {})
                if (other_agent_name in data.get('participants', []) and 
                    data.get('time', 0) >= cutoff_time):
                    shared_activities.append({
                        'time': data.get('time', 0),
                        'type': 'conversation',
                        'content': data.get('content', ''),
                        'location': data.get('location', 'Unknown')
                    })
            
            # Process activity memories
            for mem in activity_mems:
                data = mem.get('data', {})
                if (other_agent_name in data.get('participants', []) and 
                    data.get('time', 0) >= cutoff_time):
                    shared_activities.append({
                        'time': data.get('time', 0),
                        'type': 'activity',
                        'content': data.get('description', ''),
                        'location': data.get('location', 'Unknown')
                    })
            
            # Process location memories
            for mem in location_mems:
                data = mem.get('data', {})
                if (other_agent_name in data.get('participants', []) and 
                    data.get('time', 0) >= cutoff_time):
                    shared_activities.append({
                        'time': data.get('time', 0),
                        'type': 'location',
                        'content': f"Went to {data.get('location', 'Unknown')}",
                        'location': data.get('location', 'Unknown')
                    })
            
            # Sort by time (most recent first)
            shared_activities.sort(key=lambda x: x['time'], reverse=True)
            
            # Return formatted activity descriptions
            return [
                f"{activity['type'].title()}: {activity['content']} at {activity['location']}"
                for activity in shared_activities
            ]
            
        except Exception as e:
            print(f"Error getting shared activities: {str(e)}")
            return []

    def get_recent_messages_from(self, other_agent_name: str, limit: int = 10) -> List[Dict]:
        """Get recent detailed messages/conversations from a specific person.
        
        Args:
            other_agent_name: Name of the other agent to get messages from
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with time, content, location, etc.
        """
        try:
            # Retrieve conversation memories
            conversation_mems = self.memory_manager.retrieve_memories(
                self.name,
                self.current_time,
                memory_type_key='CONVERSATION_LOG_EVENT',
                limit=limit * 2  # Get more to filter properly
            )
            
            messages = []
            for mem in conversation_mems:
                data = mem.get('data', {})
                participants = data.get('participants', [])
                
                # Check if this conversation involved the other agent
                if other_agent_name in participants:
                    message_info = {
                        'time': data.get('time', mem.get('simulation_time', 0)),
                        'content': data.get('content', ''),
                        'location': data.get('location', 'Unknown'),
                        'participants': participants,
                        'interaction_type': data.get('interaction_type', 'conversation'),
                        'relationship_context': data.get('relationship_context', {}),
                        'hours_ago': self.current_time - data.get('time', self.current_time)
                    }
                    messages.append(message_info)
            
            # Sort by time (most recent first) and limit
            messages.sort(key=lambda x: x['time'], reverse=True)
            return messages[:limit]
            
        except Exception as e:
            print(f"Error getting recent messages from {other_agent_name}: {str(e)}")
            return []

    def get_conversation_summary_with(self, other_agent_name: str, time_window_hours: int = 24) -> str:
        """Get a summary of recent conversations with a specific person.
        
        Args:
            other_agent_name: Name of the other agent
            time_window_hours: How many hours back to look
            
        Returns:
            String summary of recent conversations
        """
        try:
            messages = self.get_recent_messages_from(other_agent_name, limit=20)
            
            # Filter by time window
            cutoff_time = self.current_time - time_window_hours
            recent_messages = [msg for msg in messages if msg['time'] >= cutoff_time]
            
            if not recent_messages:
                return f"No recent conversations with {other_agent_name} in the last {time_window_hours} hours."
            
            # Build summary
            summary_parts = []
            summary_parts.append(f"Recent conversations with {other_agent_name}:")
            
            for msg in recent_messages:
                hours_ago = self.current_time - msg['time']
                time_desc = f"{hours_ago} hours ago" if hours_ago > 0 else "just now"
                location = msg['location']
                content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                
                summary_parts.append(f"- {time_desc} at {location}: {content_preview}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            print(f"Error getting conversation summary with {other_agent_name}: {str(e)}")
            return f"Error retrieving conversation history with {other_agent_name}"

    def should_interact(self, context: Dict) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if the agent should interact based on the provided context.
        
        Args:
            context: A dictionary containing:
                - 'other_agent': The other agent to interact with (Agent or str).
                - 'current_time': The current time.
                - 'encounter_info': Optional encounter information.
                - 'location': Current location name.
                - 'location_type': Type of current location.
                - 'nearby_agents': List of nearby agent names.
                - 'personal_context': Personal context information.
                - Any other relevant context.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - bool: Whether to interact
                - Dict containing:
                    - 'reason': String explaining the decision
                    - 'tags': List of relevant tags
                    - 'context': Relevant context for the decision
        """
        try:
            # Extract context information
            other_agent = context.get('other_agent')
            current_time = context.get('current_time')
            encounter_info = context.get('encounter_info', {})
            location = context.get('location')
            location_type = context.get('location_type')
            nearby_agents = context.get('nearby_agents', [])
            personal_context = context.get('personal_context', {})

            # Initialize decision info
            decision_info = {
                'reason': '',
                'tags': [],
                'context': {
                    'time': current_time,
                    'location': location,
                    'location_type': location_type,
                    'energy_level': self.energy_level
                }
            }

            # Get other agent name for logging
            other_agent_name = other_agent.name if isinstance(other_agent, Agent) else other_agent

            # Check if we're already in a conversation
            if self.is_in_conversation:
                decision_info.update({
                    'reason': "Already in a conversation",
                    'tags': ['state_blocked']
                })
                print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Interaction blocked with {other_agent_name}: Already in conversation")
                return False, decision_info

            # Check relationship
            relationship = self.get_relationship_with(other_agent_name)
            decision_info['context']['relationship'] = relationship

            # Check for household members
            if other_agent_name in self.household_members:
                decision_info.update({
                    'reason': "Household member",
                    'tags': ['household', 'family']
                })
                print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Interaction approved with {other_agent_name}: Household member")
                return True, decision_info

            # Check for social location opportunities
            if location_type in ['social', 'entertainment'] and relationship != 'stranger':
                decision_info.update({
                    'reason': f"Social location with {relationship}",
                    'tags': ['location_social', relationship]
                })
                print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Interaction approved with {other_agent_name}: Social location")
                return True, decision_info

            # Check for repeated encounters
            recent_encounters = self.get_recent_memories('encounter', limit=5)
            if any(enc.get('other_agent') == other_agent_name for enc in recent_encounters):
                decision_info.update({
                    'reason': "Repeated encounter",
                    'tags': ['relationship_history', 'frequent_encounter']
                })
                print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Interaction approved with {other_agent_name}: Repeated encounter")
                return True, decision_info

            # Check for recent conversation cooldown
            recent_conversations = self.get_recent_memories('conversation', limit=5)
            if any(conv.get('participants', []).count(other_agent_name) > 0 
                  and current_time - conv.get('time', 0) < 2  # 2 hour cooldown
                  for conv in recent_conversations):
                decision_info.update({
                    'reason': "Recent conversation cooldown",
                    'tags': ['cooldown', 'recent_interaction']
                })
                print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Interaction blocked with {other_agent_name}: Recent conversation cooldown")
                return False, decision_info

            # Default case: no compelling reason to interact
            decision_info.update({
                'reason': "No compelling reason to interact",
                'tags': ['no_reason']
            })
            print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [DEBUG] Interaction blocked with {other_agent_name}: No compelling reason")
            return False, decision_info

        except Exception as e:
            error_info = {
                'reason': f"Error: {str(e)}",
                'tags': ['error'],
                'context': {
                    'error': str(e),
                    'time': current_time,
                    'location': location,
                    'other_agent': other_agent_name if 'other_agent_name' in locals() else 'unknown'
                }
            }
            print(f"[Day {current_time // 24 + 1} | Hour {current_time % 24}] [Agent: {self.name}] [ERROR] Error in interaction decision: {str(e)}")
            print(f"Context: {json.dumps(error_info['context'], indent=2)}")
            return False, error_info

    def generate_social_interaction(self, context: Dict) -> Optional[str]:
        """Generate appropriate social interaction based on unified context."""
        try:
            # First check if we should interact
            should_interact, decision_info = self.should_interact(context)
            if not should_interact:
                print(f"[{self.name}] Not generating interaction: {decision_info['reason']}")
                return None

            # Add decision info to context for the LLM
            context['interaction_decision'] = decision_info

            # Build rich prompt that considers all factors
            interaction = self.model_manager.generate(
                self.prompt_manager.get_prompt(
                    "social_interaction",
                    name=self.name,
                    location=context['location'],
                    location_type=context['location_type'],
                    time=context['time'],
                    nearby_agents=context['nearby_agents'],
                    interaction_reason=decision_info['reason'],
                    personal_context=context['personal_context'],
                    decision_tags=decision_info['tags']
                )
            )

            if not interaction:
                print(f"[{self.name}] Failed to generate interaction content")
                return None

            # Clean the interaction content
            cleaned_interaction = self._clean_conversation_content(interaction)
            if not cleaned_interaction:
                print(f"[{self.name}] Generated interaction was empty after cleaning")
                return None

            print(f"[{self.name}] Generated interaction: {cleaned_interaction[:100]}...")
            return cleaned_interaction

        except Exception as e:
            print(f"[{self.name}] Error generating social interaction: {str(e)}")
            return None


    def _handle_solo_evening_routine(self, current_time: int) -> str:
        """Handle evening routine for people living alone"""
        try:
            # Get recent activities
            recent_activities = self.memory_manager.retrieve_memories(
                self.name,
                current_time,
                memory_type_key='ACTIVITY_EVENT',
                limit=3
            )
            
            # Get next day's schedule from daily plan
            next_day_schedule = self.daily_plan or "No specific plans for tomorrow"
            
            # Generate routine using prompt
            routine = self.model_manager.generate(
                self.prompt_manager.get_prompt(
                    "solo_evening_routine",
                    name=self.name,
                    location=self.get_current_location_name(),
                    time=current_time % 24,
                    energy=self.energy_level,
                    grocery_level=self.grocery_level,
                    recent_activities=str([m['data'].get('content', '') for m in recent_activities]),
                    next_day_schedule=next_day_schedule
                )
            )
            
            # Record the routine in memory
            self.memory_manager.add_memory(
                self.name,
                'CONVERSATION_LOG_EVENT',
                {
                    'content': routine,
                    'time': current_time,
                    'location': self.get_current_location_name(),
                    'household_members': [],
                    'relationship_context': {}
                }
            )
            
            return routine
            
        except Exception as e:
            print(f"Error in solo routine for {self.name}: {str(e)}")
            return "Error in solo routine"


    def handle_agent_conversation(self, initiator: 'Agent', participants: List['Agent']):
        """Handle a conversation between agents with improved synchronization."""
        try:
            # Step 1: Verify all household members are present
            expected_names = [m['name'] for m in self.household_members]
            present_names = [a.name for a in participants]
            if set(present_names) != set(expected_names):
                print(f"[DEBUG] Waiting for all household members to arrive. Expected: {expected_names}, Present: {present_names}")
                return None  # Someone hasn't joined yet

            # Step 2: Make sure nobody is already in conversation
            if any(p.is_in_conversation for p in participants):
                print(f"[DEBUG] One or more participants are already busy with another conversation.")
                return None

            # Sort participants to ensure consistent initiator
            sorted_participants = sorted(participants, key=lambda x: x.name)
            actual_initiator = sorted_participants[0]

            # Create conversation state
            conv_state = ConversationState()
            conv_state.participants = participants

            # Mark all participants as in conversation
            for participant in participants:
                participant.is_in_conversation = True
                participant.conversation_partners = [p.name for p in participants if p != participant]

            # Generate and process conversation turns
            max_turns = 5  # Limit conversation length
            turn_count = 0

            while turn_count < max_turns:
                # Generate next turn
                turn_content = actual_initiator.generate_conversation_turn(conv_state)
                
                # Add turn to conversation state
                conv_state.add_turn(actual_initiator.name, turn_content)
                
                # Check if conversation should end
                if actual_initiator.should_end_conversation(turn_content):
                    break
                
                # Switch initiator for next turn
                actual_initiator = next(p for p in participants if p != actual_initiator)
                turn_count += 1

            # Reset conversation state for all participants
            for participant in participants:
                participant.is_in_conversation = False
                participant.conversation_partners = []

            # Log conversation end
            print(f"[DEBUG] Conversation ended after {turn_count + 1} turns")

        except Exception as e:
            print(f"Error in conversation handling: {str(e)}")
            traceback.print_exc()
            
            # Ensure conversation state is reset even if there's an error
            for participant in participants:
                participant.is_in_conversation = False
                participant.conversation_partners = []

    def find_closest_food_location(self, current_hour: int) -> Optional[str]:
        """Find the closest food-serving location for immediate dining.
        
        Rules:
        - Coffee shops: breakfast and snacks (including beverages) ONLY
        - Local diners & fried chicken shops: any meal during open hours
        - Must physically visit to purchase and consume
        - STRICT FILTERING: No grocery stores during lunch/dinner times
        
        Args:
            current_hour: Current hour of the day (0-23)
            
        Returns:
            Name of the closest food-serving location or None if none found/open
        """
        if not self.town_map or not self.current_grid_position:
            return None
            
        # Get current location name for exclusion
        current_location_name = self.get_current_location_name()
            
        # Get all food-serving locations for immediate dining (NOT grocery stores)
        food_locations = []
        
        # Determine current meal context for better filtering
        current_meal_type = self.determine_meal_type(current_hour)
        is_lunch_time = 11 <= current_hour < 15
        is_dinner_time = 17 <= current_hour < 22
        
        for loc_name, location in self.locations.items():
            if loc_name == current_location_name:
                continue  # Skip current location
                
            # STRICT FILTERING: Never suggest grocery stores/markets for immediate dining
            if location.type in ['market', 'grocery', 'supermarket']:
                print(f"DEBUG FOOD: {self.name} - Excluding {loc_name} (grocery/market) for immediate dining")
                continue
                
            if location.is_open(current_hour):
                location_suitable = False
                
                # Apply meal-specific location filters
                if 'coffee' in loc_name.lower() or location.type == 'coffee_shop':
                    # Coffee shops: ONLY for breakfast and snacks, NEVER lunch/dinner
                    if (current_meal_type == 'breakfast') or (current_meal_type == 'snack' and not is_lunch_time and not is_dinner_time):
                        location_suitable = True
                        print(f"DEBUG FOOD: {self.name} - Including {loc_name} (coffee shop) for {current_meal_type} at {current_hour:02d}:00")
                    else:
                        print(f"DEBUG FOOD: {self.name} - Excluding {loc_name} (coffee shop) - cannot serve {current_meal_type} at {current_hour:02d}:00")
                        
                elif 'diner' in loc_name.lower() or 'fried chicken' in loc_name.lower() or location.type in ['restaurant', 'local_shop']:
                    # Restaurants: suitable for all meals when open
                    location_suitable = True
                    print(f"DEBUG FOOD: {self.name} - Including {loc_name} (restaurant) for {current_meal_type} at {current_hour:02d}:00")
                else:
                    # Other food-serving locations: include if open and not grocery
                    if location.type not in ['market', 'grocery', 'supermarket']:
                        location_suitable = True
                        print(f"DEBUG FOOD: {self.name} - Including {loc_name} ({location.type}) for {current_meal_type}")
                    
                if location_suitable:
                    food_locations.append((loc_name, location))
        
        if not food_locations:
            print(f"DEBUG FOOD: {self.name} - No suitable food locations found for {current_meal_type} at {current_hour:02d}:00")
            return None
        
        # Find closest location using grid coordinates
        closest_location = None
        min_distance = float('inf')
        
        for loc_name, location in food_locations:
            if hasattr(location, 'grid_coordinate') and location.grid_coordinate:
                distance = abs(self.current_grid_position[0] - location.grid_coordinate[0]) + \
                          abs(self.current_grid_position[1] - location.grid_coordinate[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_location = loc_name
        
        if closest_location:
            print(f"DEBUG FOOD: {self.name} - Closest appropriate food location: {closest_location} (distance: {min_distance})")
        else:
            print(f"DEBUG FOOD: {self.name} - No food locations with valid coordinates found")
            
        return closest_location

    def find_closest_grocery_store(self, current_hour: int) -> Optional[str]:
        """Find the closest grocery store for shopping trips.
        
        Grocery stores sell ingredients and supplies but NOT prepared food.
        Agents must purchase ingredients and cook at home.
        
        Args:
            current_hour: Current hour of the day (0-23)
            
        Returns:
            Name of the closest grocery store or None if none found/open
        """
        if not self.town_map or not self.current_grid_position:
            return None
            
        # Get current location name for exclusion
        current_location_name = self.get_current_location_name()
            
        # Get all grocery stores (NOT food-serving locations)
        grocery_stores = []
        
        for loc_name, location in self.locations.items():
            if loc_name == current_location_name:
                continue  # Skip current location
                
            # Only include locations that sell groceries/ingredients
            if location.type in ['market', 'grocery']:
                if location.is_open(current_hour):
                    grocery_stores.append((loc_name, location))
        
        if not grocery_stores:
            return None
        
        # Find closest based on grid distance
        closest_distance = float('inf')
        closest_store = None
        
        for store_name, store_location in grocery_stores:
            if hasattr(store_location, 'grid_coordinate') and store_location.grid_coordinate:
                distance = abs(self.current_grid_position[0] - store_location.grid_coordinate[0]) + \
                          abs(self.current_grid_position[1] - store_location.grid_coordinate[1])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_store = store_name
                    
        return closest_store

    def needs_groceries(self, current_hour: int) -> bool:
        """Determine if agent needs to buy groceries based on current stock and preferences.
        
        Args:
            current_hour: Current hour of the day (0-23)
            
        Returns:
            True if agent should buy groceries, False otherwise
        """
        # Basic need: grocery level is low
        if self.grocery_level < 30:
            return True
        
        # Check if any grocery stores are open (don't go if all closed)
        grocery_open = False
        for location in self.locations.values():
            if location.type in ['market', 'grocery'] and location.is_open(current_hour):
                grocery_open = True
                break
        
        if not grocery_open:
            return False
        
        # Proactive shopping: if grocery level is moderate but it's a good time to shop
        if self.grocery_level < 60:
            # Good times to shop: mid-morning (9-11) or mid-afternoon (14-16)
            shopping_time = (9 <= current_hour <= 11) or (14 <= current_hour <= 16)
            if shopping_time:
                return True
        
        return False

    def handle_grocery_shopping(self, current_time: int) -> str:
        """Handle grocery shopping trip.
        
        Args:
            current_time: Current simulation time in hours
            
        Returns:
            String describing the grocery shopping action
        """
        try:
            current_hour = current_time % 24
            current_location_name = self.get_current_location_name()
            
            # If already at a grocery store, make purchase
            if current_location_name in self.locations:
                location = self.locations[current_location_name]
                if location.type in ['market', 'grocery'] and location.is_open(current_hour):
                    # Calculate how much to buy based on available money
                    current_price = location.get_current_price(current_time)
                    available_money = self.money * 0.3  # Use up to 30% of money for groceries
                    
                    if available_money >= current_price:
                        # Calculate purchase amount
                        max_affordable = int(available_money / current_price)
                        purchase_amount = min(max_affordable, 5)  # Buy up to 5 units
                        
                        purchase_result = self.make_purchase(
                            location_name=current_location_name,
                            item_type='groceries',
                            item_description=f'{purchase_amount} units of groceries'
                        )
                        
                        if purchase_result['success']:
                            return f"Purchased groceries at {current_location_name}"
                        else:
                            reason = purchase_result.get('reason', 'Unknown error') if isinstance(purchase_result, dict) else str(purchase_result)
                            return f"Failed to purchase groceries at {current_location_name}: {reason}"
                    else:
                        return f"Cannot afford groceries at {current_location_name} (need ${current_price:.2f}, have ${available_money:.2f})"
            
            # If not at grocery store, find closest one
            grocery_store = self.find_closest_grocery_store(current_hour)
            if grocery_store and grocery_store != current_location_name:
                travel_result = self.start_travel_to(grocery_store)
                return f"Heading to {grocery_store} for grocery shopping: {travel_result}"
            
            return f"No grocery stores available at this time"
            
        except Exception as e:
            print(f"Error in grocery shopping for {self.name}: {str(e)}")
            return "Error handling grocery shopping"

    def should_end_conversation(self, turn_content: str) -> bool:
        """Determine if the conversation should end using structured LLM decision."""
        try:
            # Build context for conversation ending decision
            context = {
                'participants': [],  # Will be filled by caller if available
                'location': self.get_current_location_name(),
                'conversation_state': None,  # Will be filled by caller if available
                'time': getattr(self, 'current_time', 0) % 24,
                'daily_plan': getattr(self, 'daily_plan', 'No specific plans'),
                'conversation_duration': 5  # Default, will be filled by caller if available
            }
            
            # Generate structured decision
            decision = self.generate_structured_conversation_ending(context)
            
            # Log the decision for debugging
            if hasattr(self, 'memory_manager'):
                self.log_memory('CONVERSATION_LOG_EVENT', {
                    'decision': decision.get('continue_conversation', 'YES'),
                    'reasoning': decision.get('reasoning', ''),
                    'natural_thoughts': decision.get('natural_thoughts', ''),
                    'time': context['time'],
                    'location': context['location']
                })
            
            return decision.get('continue_conversation', 'YES') == 'NO'
            
        except Exception as e:
            print(f"Error in structured conversation ending for {self.name}: {e}")
            # Fallback to simple check
            end_phrases = ["goodbye", "see you", "bye", "have to go", "need to leave", "take care", "catch you later"]
            return any(phrase in turn_content.lower() for phrase in end_phrases)

    def _fallback_energy_update(self, action_str: str):
        """Fallback energy update using keyword detection with new energy system."""
        action_lower = action_str.lower()
        
        # Work activities: -20 energy
        if any(word in action_lower for word in ['work', 'working', 'meeting', 'project', 'office', 'job']):
            self.energy_level = max(0, self.energy_level - 20)
        # Food activities: use actual location detection instead of keywords
        elif any(meal in action_lower for meal in ['eat', 'meal', 'breakfast', 'lunch', 'dinner']):
            current_location_name = self.get_current_location_name()
            
            # Check actual location type for accurate energy calculation
            if current_location_name in self.locations:
                location = self.locations[current_location_name]
                location_type = location.type
                
                if location_type in ['local_shop'] and current_location_name in ['Fried Chicken Shop', 'Local Diner']:
                    # Full restaurant meals
                    self.energy_level = min(100, self.energy_level + 40)
                elif location_type in ['local_shop'] and current_location_name in ['The Coffee Shop']:
                    # Coffee shop snacks/beverages
                    self.energy_level = min(100, self.energy_level + 5)
                elif current_location_name == self.residence:
                    # Home-cooked meals
                    self.energy_level = min(100, self.energy_level + 20)
                else:
                    # Default to homemade if location unclear
                    self.energy_level = min(100, self.energy_level + 20)
            else:
                # Fallback to keyword detection if location not found
                if any(place in action_lower for place in ['restaurant', 'cafe', 'shop', 'store']):
                    self.energy_level = min(100, self.energy_level + 40)  # Restaurant meals
                elif any(home in action_lower for home in ['home', 'cook', 'make', 'prepare']):
                    self.energy_level = min(100, self.energy_level + 20)  # Homemade meals
                else:
                    self.energy_level = min(100, self.energy_level + 20)  # Default to homemade
        # Snacks and beverages
        elif any(snack in action_lower for snack in ['snack', 'beverage', 'drink', 'coffee', 'tea']):
            self.energy_level = min(100, self.energy_level + 5)
        # Sleep and rest: +50 energy
        elif any(rest in action_lower for rest in ['sleep', 'rest', 'bed', 'relax']):
            self.energy_level = min(100, self.energy_level + 50)
        # Naps: +10 energy (with time restrictions)
        elif 'nap' in action_lower:
            current_hour = self.current_time % 24
            if 11 <= current_hour <= 15:  # Naps only allowed 11 AM - 3 PM
                self.energy_level = min(100, self.energy_level + 10)
        # Travel energy is handled per-step in _perform_travel_step, so no change here
        elif any(word in action_lower for word in ['travel', 'walk', 'go to', 'head to', 'move']):
            pass  # Travel energy handled elsewhere
        else:
            # Default idle activity: small energy drain
            self.energy_level = max(0, self.energy_level - 3)

    def is_work_time(self) -> bool:
        """Check if it's currently work time."""
        if not hasattr(self, 'work_schedule') or not self.work_schedule:
            return False
            
        current_hour = getattr(self, 'current_time', 0) % 24
        start_time = self.work_schedule.get('start', 9)
        end_time = self.work_schedule.get('end', 17)
        
        return start_time <= current_hour < end_time

    def handle_food_needs(self, current_time: int) -> str:
        """Handle food needs for the agent."""
        try:
            print(f"DEBUG: {self.name} handling food needs at {current_time:02d}:00")
            
            # Check if it's meal time and what type
            is_meal_time, meal_type = self.is_meal_time(current_time)
            if not is_meal_time:
                return None
                
            print(f"DEBUG FOOD: {self.name} - meal time ({meal_type}) and can cook at home")
            
            # Check if we have groceries
            if self.grocery_level > 20:
                # Cook and eat at home
                self.grocery_level -= 20
                self.energy_level = min(100, self.energy_level + 20)
                
                # Record the activity using the correct memory type constant
                self.record_memory('ACTIVITY_EVENT', {
                    'time': current_time,
                    'agent_name': self.name,
                    'location': self.get_current_location_name(),
                    'activity_type_tag': 'dining',
                    'description': f'Cooked and ate {meal_type} at home',
                    'energy_gain': 20,
                    'grocery_cost': 20
                })
                
                # Record state update
                self.record_memory('AGENT_STATE_UPDATE_EVENT', {
                    'time': current_time,
                    'energy_level': self.energy_level,
                    'grocery_level': self.grocery_level,
                    'location': self.get_current_location_name(),
                    'action': f'ate {meal_type}'
                })
                
                return f"Made and ate {meal_type} at home using groceries"
            else:
                # Need to go get food
                food_location = self.find_closest_food_location(current_time)
                if food_location:
                    return f"Need to get {meal_type} at {food_location}"
                else:
                    return f"Need to get groceries for {meal_type}"
                    
        except Exception as e:
            print(f"Error handling food needs for {self.name}: {str(e)}")
            traceback.print_exc()
            return None

    def generate_conversation_turn(self, conv_state: 'ConversationState') -> str:
        """Generate a conversation turn using the conversation prompt."""
        try:
            current_hour = self.current_time % 24 if hasattr(self, 'current_time') else 0
            
            # Get recent conversation memories for context
            recent_memory_context = ""
            if len(conv_state.dialogue) > 0:
                # Use recent turns as context
                recent_turns = conv_state.dialogue[-3:]  # Last 3 turns
                recent_memory_context = "\n".join([
                    f"{turn['speaker']}: {turn['content']}" 
                    for turn in recent_turns
                ])
            
            context = {
                'name': self.name,
                'location': conv_state.location.name if conv_state.location else "unknown location",
                'location_type': conv_state.location.type if conv_state.location else "unknown",
                'time': current_hour,
                'energy_level': self.energy_level,
                'current_activity': getattr(self, 'current_activity', 'having a conversation'),
                'recent_interactions': self.get_recent_activities(limit=2),
                'relationships': {
                    name: self.get_relationship_with(name) 
                    for name in conv_state.participants if name != self.name
                },
                'previous_turns': "\n".join([
                    f"{turn['speaker']}: {turn['content']}" 
                    for turn in conv_state.dialogue[-5:]  # Last 5 turns for context
                ]) if conv_state.dialogue else "This is the start of the conversation.",
                'recent_memory_context': recent_memory_context
            }
            
            # Generate conversation turn using prompt
            prompt = self.prompt_manager.get_prompt('conversation_turn', **context)
            turn_content = self.model_manager.generate(prompt)
            
            return turn_content
            
        except Exception as e:
            print(f"Error generating conversation turn for {self.name}: {str(e)}")
            return "I should probably get going."

    def generate_household_conversation_turn(self, conv_state: 'ConversationState', current_time: int) -> str:
        """Generate a turn in a household conversation with improved timeout handling."""
        try:
            # Get recent activities and memories for context
            recent_activities = self.get_recent_activities(limit=3)
            recent_memories = self.get_recent_memories(memory_type="conversation", limit=3)
            
            # Build the prompt with household context
            prompt = self.prompt_manager.get_prompt(
                "household_coordination",
                name=self.name,
                time=current_time,
                location=self.get_current_location_name(),
                members=[m.name for m in conv_state.participants if m.name != self.name],
                relationships={m.name: self.get_relationship_with(m.name) for m in conv_state.participants if m.name != self.name},
                recent_activities=recent_activities,
                work_schedules={m.name: m.work_schedule for m in conv_state.participants if m.name != self.name},
                shared_meals=self._get_recent_household_conversations(current_time, conv_state.participants),
                personal_activities=recent_activities
            )
            
            # Add conversation history context
            if conv_state.dialogue:
                prompt += "\n\nPrevious conversation turns:\n"
                for turn in conv_state.get_last_turns(3):
                    prompt += f"{turn['speaker']}: {turn['content']}\n"
            
            # Add memory context
            if recent_memories:
                prompt += "\n\nRecent conversation memories:\n"
                for memory in recent_memories:
                    prompt += f"- {memory['content']}\n"
            
            # Generate response with timeout protection
            response = self.model_manager.generate_response(prompt, max_attempts=2, timeout=10)
            
            if not response:
                return f"{self.name}: (pauses thoughtfully) I'm not sure what to say right now."
            
            return response
            
        except Exception as e:
            print(f"Error generating household conversation turn for {self.name}: {str(e)}")
            return f"{self.name}: (smiles warmly) It's nice spending time together."

    def _clean_conversation_content(self, content: str) -> str:
        """Clean conversation content by removing meta-commentary and stage directions."""
        if not content:
            return ""
            
        # Remove common meta-commentary patterns
        meta_patterns = [
            r'\[Note:.*?\]',
            r'\(Note:.*?\)',
            r'\[This.*?\]',
            r'\(This.*?\)',
            r'\[Optional:.*?\]',
            r'\(Optional:.*?\)',
            r'Example:.*',
            r'Note that.*',
            r'Remember.*',
            r'Keep in mind.*'
        ]
        
        import re
        cleaned = content
        for pattern in meta_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove excessive stage directions but keep brief ones
        # Remove complex stage directions like *looks around thoughtfully while considering*
        cleaned = re.sub(r'\*[^*]{50,}\*', '', cleaned)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # If nothing meaningful remains, return empty string
        if len(cleaned) < 5 or cleaned.lower() in ['ok', 'yes', 'no', 'sure']:
            return ""
            
        return cleaned

    def _should_make_purchase(self, action_str: str) -> bool:
        """This method is deprecated. Use generate_structured_purchase_decision instead."""
        print(f"Warning: {self.name} using deprecated _should_make_purchase method. Use generate_structured_purchase_decision instead.")
        return False

    def _extract_purchase_intent(self, action_str: str) -> dict:
        """This method is deprecated. Use generate_structured_purchase_decision instead."""
        print(f"Warning: {self.name} using deprecated _extract_purchase_intent method. Use generate_structured_purchase_decision instead.")
        return {
            'item_type': 'misc',
            'item_description': 'various items',
            'reason': 'deprecated_method'
        }

    def parse_structured_action(self, action_response: str) -> Dict[str, str]:
        """Parse structured action response with hybrid natural language + structured format.
        
        Expected format:
        [Natural conversation/thoughts]
        
        LOCATION: [location name]
        ACTION: [action description] 
        REASONING: [reasoning]
        
        Returns:
            Dict containing parsed location, action, reasoning, and natural thoughts
        """
        parsed = {
            'location': None,
            'action': None,
            'reasoning': None,
            'natural_thoughts': None,
            'raw_response': action_response
        }
        
        # Handle None or empty response
        if not action_response:
            print(f"Warning: {self.name} received empty/None response from model")
            parsed['natural_thoughts'] = "No response from model"
            return parsed
        
        try:
            # Split response into natural thoughts and structured parts
            if 'LOCATION:' in action_response:
                parts = action_response.split('LOCATION:', 1)
                if len(parts) == 2:
                    parsed['natural_thoughts'] = parts[0].strip()
                    structured_part = 'LOCATION:' + parts[1]
                else:
                    structured_part = action_response
            else:
                # No structured part found, treat entire response as natural thoughts
                parsed['natural_thoughts'] = action_response.strip()
                return parsed
            
            # Parse structured part
            lines = structured_part.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('LOCATION:'):
                    parsed['location'] = line.replace('LOCATION:', '').strip()
                elif line.startswith('ACTION:'):
                    parsed['action'] = line.replace('ACTION:', '').strip()
                elif line.startswith('REASONING:'):
                    parsed['reasoning'] = line.replace('REASONING:', '').strip()
                    
        except Exception as e:
            print(f"Error parsing structured action for {self.name}: {e}")
            # If parsing fails, treat entire response as natural thoughts
            parsed['natural_thoughts'] = action_response.strip() if action_response else "Empty response"
            
        return parsed

    def generate_structured_action(self, context: Dict) -> Dict[str, str]:
        """Generate a structured action based on context."""
        try:
            # Ensure available_locations is in context
            if 'available_locations' not in context:
                available_locations = []
                for location_name, location in self.locations.items():
                    if location.is_open(context['time']):
                        available_locations.append(location_name)
                context['available_locations'] = '\n'.join(f"- {loc}" for loc in available_locations)
            
            # Get prompt from prompt manager
            prompt = self.prompt_mgr.get_prompt('structured_action', **context)
            
            # Generate structured response using model manager
            result = self.model_mgr.generate_structured(prompt, {
                'location': str,
                'action': str,
                'reasoning': str
            })
            
            return result if result else {
                'location': 'stay',
                'action': 'No action specified',
                'reasoning': 'Failed to generate structured action'
            }
            
        except Exception as e:
            print(f"Error generating structured action for {self.name}: {str(e)}")
            return {
                'location': 'stay',
                'action': 'Error occurred',
                'reasoning': str(e)
            }

    def parse_structured_purchase(self, purchase_response: str) -> Dict[str, str]:
        """Parse structured purchase response with hybrid natural language + structured format.
        
        Expected format:
        [Natural conversation/thoughts]
        
        PURCHASE: [YES/NO]
        ITEM_TYPE: [groceries/meal/beverages_and_snacks/misc]
        ITEM_DESCRIPTION: [description]
        REASONING: [reasoning]
        
        Returns:
            Dict containing parsed purchase decision and natural thoughts
        """
        parsed = {
            'purchase': None,
            'item_type': None,
            'item_description': None,
            'reasoning': None,
            'natural_thoughts': None,
            'raw_response': purchase_response
        }
        
        try:
            # Split response into natural thoughts and structured parts
            if 'PURCHASE:' in purchase_response:
                parts = purchase_response.split('PURCHASE:', 1)
                if len(parts) == 2:
                    parsed['natural_thoughts'] = parts[0].strip()
                    structured_part = 'PURCHASE:' + parts[1]
                else:
                    structured_part = purchase_response
            else:
                # No structured part found, treat entire response as natural thoughts
                parsed['natural_thoughts'] = purchase_response.strip()
                return parsed
            
            # Parse structured part
            lines = structured_part.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('PURCHASE:'):
                    parsed['purchase'] = line.replace('PURCHASE:', '').strip().upper()
                elif line.startswith('ITEM_TYPE:'):
                    parsed['item_type'] = line.replace('ITEM_TYPE:', '').strip()
                elif line.startswith('ITEM_DESCRIPTION:'):
                    parsed['item_description'] = line.replace('ITEM_DESCRIPTION:', '').strip()
                elif line.startswith('REASONING:'):
                    parsed['reasoning'] = line.replace('REASONING:', '').strip()
                    
        except Exception as e:
            print(f"Error parsing structured purchase for {self.name}: {e}")
            # If parsing fails, treat entire response as natural thoughts
            parsed['natural_thoughts'] = purchase_response.strip()
            
        return parsed

    def generate_structured_purchase_decision(self, context: Dict) -> Dict[str, str]:
        """Generate a structured purchase decision using hybrid natural language + structured output approach."""
        try:
            current_location_name = context['location']
            current_location = self.locations.get(current_location_name)
            
            if not current_location:
                print(f"Warning: {self.name} trying to make purchase decision at unknown location: {current_location_name}")
                return {'purchase': 'NO', 'reasoning': 'Unknown location'}
            
            # Get location information
            base_price = getattr(current_location, 'base_price', 0.0)
            current_price = current_location.get_current_price(context['time'])
            
            # Get recent purchase history - ONLY FROM CURRENT SIMULATION
            current_simulation_day = (context['time'] // 24) + 1
            
            # Debug: check what memories are being retrieved
            print(f"DEBUG MEMORY: {self.name} checking recent purchases for day {current_simulation_day}, time {context['time']}")
            
            recent_purchases = self.memory_manager.retrieve_memories(
                self.name, context['time'], 'FOOD_PURCHASE_EVENT', limit=5
            ) + self.memory_manager.retrieve_memories(
                self.name, context['time'], 'GROCERY_PURCHASE_EVENT', limit=5
            )
            
            # Debug: log what was retrieved
            print(f"DEBUG MEMORY: {self.name} retrieved {len(recent_purchases)} purchase memories")
            for i, purchase in enumerate(recent_purchases):
                data = purchase.get('data', {})
                purchase_time = data.get('time', 'unknown')
                location = data.get('location', 'unknown')
                print(f"DEBUG MEMORY: Purchase {i+1}: time={purchase_time}, location={location}")
            
            purchase_history_str = "No recent purchases"
            
            # CRITICAL FIX: Only show purchases from current simulation (filter by simulation time)
            # If we're on day 1 and it's the first lunch, there should be NO previous meal purchases
            valid_purchases = []
            if recent_purchases:
                for purchase in recent_purchases:
                    data = purchase.get('data', {})
                    purchase_time = data.get('time', -1)
                    
                    # Only include purchases from the current simulation run (not test data)
                    # If purchase time is from before current simulation started, ignore it
                    if purchase_time >= 0 and purchase_time <= context['time']:
                        valid_purchases.append(purchase)
                        print(f"DEBUG MEMORY: Including purchase from time {purchase_time}")
                    else:
                        print(f"DEBUG MEMORY: Excluding invalid purchase from time {purchase_time}")
            
            if valid_purchases:
                history_items = []
                for purchase in valid_purchases[-3:]:  # Last 3 valid purchases
                    data = purchase.get('data', {})
                    location = data.get('location', 'Unknown')
                    amount = data.get('amount', 0)
                    item_desc = data.get('item_description', 'items')
                    purchase_time = data.get('time', 0)
                    purchase_hour = purchase_time % 24
                    purchase_day = (purchase_time // 24) + 1
                    history_items.append(f"- Day {purchase_day} {purchase_hour:02d}:00 at {location}: ${amount:.2f} for {item_desc}")
                purchase_history_str = "\n".join(history_items)
                print(f"DEBUG MEMORY: Final purchase history: {purchase_history_str}")
            else:
                print(f"DEBUG MEMORY: No valid recent purchases found - this is expected for early simulation")
            
            # Calculate available money (keep some reserve)
            available_money = max(0, self.money - 30)  # Keep $30 as reserve
            
            # Get location offers
            location_offers = "No special offers"
            if hasattr(current_location, 'discounts') and current_location.discounts:
                discount_info = current_location.discounts
                current_day = (context['time'] // 24) + 1
                if current_day in discount_info.get('days', []):
                    discount_value = discount_info.get('value', 0)
                    location_offers = f"{discount_value}% discount available today!"
            
            # Generate structured prompt
            prompt = self.prompt_manager.get_prompt(
                "structured_purchase",
                name=context['name'],
                location=current_location_name,
                time=context['time'],
                energy_level=context['energy_level'],
                grocery_level=context['grocery_level'],
                money=context['money'],
                available_money=available_money,
                location_type=current_location.type,
                location_offers=location_offers,
                base_price=base_price,
                current_price=current_price,
                recent_purchase_history=purchase_history_str
            )
            
            # Get response from model
            response = self.model_manager.generate(prompt)
            
            # Check if response is valid before parsing
            if not response:
                print(f"Warning: {self.name} received empty/None response from model manager")
                return {
                    'purchase': 'NO',
                    'reasoning': 'Model response was empty or None',
                    'final_outcome': 'Purchase decision failed due to error'
                }
            
            # Parse the structured response
            parsed = self.parse_structured_purchase(response)
            
            # Validate and execute purchase if needed
            if parsed.get('purchase') == 'YES':
                item_type = parsed.get('item_type', 'misc')
                item_description = parsed.get('item_description', 'generic item')
                
                print(f"DEBUG: {self.name} purchase decision is YES, executing purchase")
                print(f"DEBUG: Item type: {item_type}, Description: {item_description}")
                print(f"DEBUG: Location: {current_location_name}")
                
                # Execute the purchase
                purchase_result = self.make_purchase(
                    location_name=current_location_name,
                    item_type=item_type,
                    item_description=item_description
                )
                
                print(f"DEBUG: Purchase result type: {type(purchase_result)}")
                print(f"DEBUG: Purchase result: {purchase_result}")
                
                # Log the purchase with appropriate memory type
                if item_type in ['groceries']:
                    memory_type = 'GROCERY_PURCHASE_EVENT'
                    self.log_memory(memory_type, {
                        'time': context['time'],
                        'location': current_location_name,
                        'items_list': [item_description],
                        'amount': purchase_result.get('amount', purchase_result.get('price', 0.0)) if isinstance(purchase_result, dict) else 0.0,
                        'used_discount': purchase_result.get('used_discount', False) if isinstance(purchase_result, dict) else False,
                        'purchase_result': purchase_result,
                        'structured_decision': True,
                        'natural_thoughts': parsed.get('natural_thoughts', ''),
                        'reasoning': parsed.get('reasoning', '')
                    })
                else:
                    memory_type = 'FOOD_PURCHASE_EVENT'
                    self.log_memory(memory_type, {
                        'time': context['time'],
                        'location': current_location_name,
                        'item_description': item_description,
                        'amount': purchase_result.get('amount', purchase_result.get('price', 0.0)) if isinstance(purchase_result, dict) else 0.0,
                        'used_discount': purchase_result.get('used_discount', False) if isinstance(purchase_result, dict) else False,
                        'purchase_result': purchase_result,
                        'structured_decision': True,
                        'natural_thoughts': parsed.get('natural_thoughts', ''),
                        'reasoning': parsed.get('reasoning', '')
                    })
                
                # Update parsed result with purchase outcome
                if purchase_result and isinstance(purchase_result, dict) and purchase_result.get('success'):
                    price_paid = purchase_result.get('price', purchase_result.get('amount', 0.0))
                    parsed['final_outcome'] = f"Successfully purchased {item_description} for ${price_paid:.2f}"
                    print(f"DEBUG: {self.name} structured purchase - bought {item_description} for ${price_paid:.2f}")
                else:
                    if isinstance(purchase_result, dict):
                        parsed['final_outcome'] = f"Purchase failed: {purchase_result.get('reason', 'Unknown error')}"
                        print(f"DEBUG: {self.name} structured purchase failed: {purchase_result.get('reason', '')}")
                        print(f"DEBUG: Full purchase result: {purchase_result}")
                    else:
                        parsed['final_outcome'] = f"Purchase failed: Invalid response type"
                        print(f"DEBUG: {self.name} structured purchase failed: Invalid response type - got {type(purchase_result)}")
                        print(f"DEBUG: Full purchase result: {purchase_result}")
            else:
                parsed['final_outcome'] = "Decided not to make a purchase"
                print(f"DEBUG: {self.name} structured purchase - decided not to buy anything")
                print(f"DEBUG: Purchase decision was: '{parsed.get('purchase')}'")
                print(f"DEBUG: Full parsed result: {parsed}")
            
            return parsed
            
        except Exception as e:
            print(f"Error in structured purchase decision for {self.name}: {e}")
            return {
                'purchase': 'NO',
                'reasoning': f'Error in decision process: {str(e)}',
                'final_outcome': 'Purchase decision failed due to error'
            }

    def parse_structured_conversation_ending(self, response: str) -> Dict[str, str]:
        """Parse structured conversation ending response."""
        result = {
            'continue_conversation': 'YES',  # Default to continue
            'reasoning': '',
            'natural_thoughts': ''
        }
        
        try:
            # Split into natural thoughts and structured parts
            if 'CONTINUE_CONVERSATION:' in response:
                parts = response.split('CONTINUE_CONVERSATION:', 1)
                result['natural_thoughts'] = parts[0].strip()
                structured_part = 'CONTINUE_CONVERSATION:' + parts[1]
            else:
                result['natural_thoughts'] = response
                return result
            
            # Parse CONTINUE_CONVERSATION
            if 'CONTINUE_CONVERSATION:' in structured_part:
                continue_match = structured_part.split('CONTINUE_CONVERSATION:')[1].split('\n')[0].strip()
                if continue_match.upper() in ['YES', 'NO']:
                    result['continue_conversation'] = continue_match.upper()
            
            # Parse REASONING
            if 'REASONING:' in structured_part:
                reasoning_match = structured_part.split('REASONING:')[1].split('\n')[0].strip()
                result['reasoning'] = reasoning_match
                
        except Exception as e:
            print(f"Error parsing conversation ending response: {e}")
            
        return result

    def generate_structured_conversation_ending(self, context: Dict) -> Dict[str, str]:
        """Generate a structured conversation ending decision."""
        try:
            # Format conversation history
            conversation_history = ""
            if 'conversation_state' in context:
                conv_state = context['conversation_state']
                history_lines = []
                for turn in conv_state.dialogue[-5:]:  # Last 5 turns
                    history_lines.append(f"{turn['speaker']}: {turn['content']}")
                conversation_history = "\n".join(history_lines)
            
            # Calculate conversation duration
            duration = context.get('conversation_duration', 5)
            
            # Build prompt
            prompt = self.prompt_manager.get_prompt(
                'structured_conversation_ending',
                name=self.name,
                participants=", ".join(context.get('participants', [])),
                location=context.get('location', 'Unknown'),
                conversation_history=conversation_history,
                time=context.get('time', 0),
                energy_level=self.energy_level,
                current_plans=context.get('daily_plan', 'No specific plans'),
                conversation_duration=duration
            )
            
            # Generate and parse response
            response = self.model_manager.generate(prompt)
            return self.parse_structured_conversation_ending(response)
            
        except Exception as e:
            print(f"Error in generate_structured_conversation_ending for {self.name}: {e}")
            return {'continue_conversation': 'NO', 'reasoning': 'Error in decision making'}

    def parse_structured_location_visit(self, response: str) -> Dict[str, str]:
        """Parse structured location visit response."""
        result = {
            'visit_location': 'NO',
            'visit_duration': 'NONE',
            'action_plan': 'none',
            'reasoning': '',
            'natural_thoughts': ''
        }
        
        try:
            # Split into natural thoughts and structured parts
            if 'VISIT_LOCATION:' in response:
                parts = response.split('VISIT_LOCATION:', 1)
                result['natural_thoughts'] = parts[0].strip()
                structured_part = 'VISIT_LOCATION:' + parts[1]
            else:
                result['natural_thoughts'] = response
                return result
            
            # Parse VISIT_LOCATION
            if 'VISIT_LOCATION:' in structured_part:
                visit_match = structured_part.split('VISIT_LOCATION:')[1].split('\n')[0].strip()
                if visit_match.upper() in ['YES', 'NO']:
                    result['visit_location'] = visit_match.upper()
            
            # Parse VISIT_DURATION
            if 'VISIT_DURATION:' in structured_part:
                duration_match = structured_part.split('VISIT_DURATION:')[1].split('\n')[0].strip()
                if duration_match.upper() in ['QUICK', 'REGULAR', 'NONE']:
                    result['visit_duration'] = duration_match.upper()
            
            # Parse ACTION_PLAN
            if 'ACTION_PLAN:' in structured_part:
                action_match = structured_part.split('ACTION_PLAN:')[1].split('\n')[0].strip()
                result['action_plan'] = action_match
            
            # Parse REASONING
            if 'REASONING:' in structured_part:
                reasoning_match = structured_part.split('REASONING:')[1].split('\n')[0].strip()
                result['reasoning'] = reasoning_match
                
        except Exception as e:
            print(f"Error parsing location visit response: {e}")
            
        return result

    def generate_structured_location_visit(self, context: Dict) -> Dict[str, str]:
        """Generate a structured location visit decision during travel."""
        try:
            # Build prompt
            prompt = self.prompt_manager.get_prompt(
                'structured_location_visit',
                name=self.name,
                destination=context.get('destination', 'Unknown'),
                encountered_location=context.get('encountered_location', 'Unknown'),
                location_type=context.get('location_type', 'unknown'),
                time=context.get('time', 0),
                energy_level=self.energy_level,
                money=context.get('money', 0),
                urgency_level=context.get('urgency_level', 'moderate'),
                location_offers=context.get('location_offers', 'None'),
                current_plans=context.get('daily_plan', 'No specific plans')
            )
            
            # Generate and parse response
            response = self.model_manager.generate(prompt)
            return self.parse_structured_location_visit(response)
            
        except Exception as e:
            print(f"Error in generate_structured_location_visit for {self.name}: {e}")
            return {'visit_location': 'NO', 'reasoning': 'Error in decision making'}

    def parse_structured_activity_classification(self, response: str) -> Dict[str, str]:
        """Parse a structured activity classification from the response."""
        try:
            # Get prompt for activity classification
            prompt = self.prompt_mgr.get_prompt('activity_classification', response=response)
            
            # Generate structured classification
            classification = self.model_mgr.generate_structured(prompt, {
                'activity_type': str,
                'description': str,
                'energy_impact': str
            })
            
            return classification if classification else {
                'activity_type': 'unknown',
                'description': response,
                'energy_impact': 'neutral'
            }
            
        except Exception as e:
            print(f"Error parsing activity classification: {str(e)}")
            return {
                'activity_type': 'unknown',
                'description': response,
                'energy_impact': 'neutral'
            }

    def parse_llm_plan_to_activity(self, plan_content: str) -> List[Dict[str, Any]]:
        """Parse the LLM-generated plan into structured activities."""
        try:
            print(f"\n[DEBUG] Parsing plan content for activities: {plan_content[:200]}...")
            
            # If plan_content is a dictionary with a schedule, use that directly
            if isinstance(plan_content, dict) and 'schedule' in plan_content:
                schedule = plan_content['schedule']
                activities = []
                for time, description in schedule.items():
                    # Try to determine activity type and location from description
                    activity_type = 'other'
                    location = 'unspecified'
                    
                    # Determine activity type
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ['wake', 'morning routine', 'shower', 'dress']):
                        activity_type = 'morning_routine'
                    elif any(word in desc_lower for word in ['breakfast', 'lunch', 'dinner', 'eat']):
                        activity_type = 'meal'
                    elif any(word in desc_lower for word in ['work', 'shift']):
                        activity_type = 'work'
                    elif any(word in desc_lower for word in ['travel', 'commute', 'go to']):
                        activity_type = 'travel'
                    elif any(word in desc_lower for word in ['rest', 'relax', 'unwind']):
                        activity_type = 'rest'
                    elif any(word in desc_lower for word in ['social', 'interact', 'visit']):
                        activity_type = 'social'
                    
                    # Determine location
                    if 'home' in desc_lower:
                        location = 'home'
                    elif 'work' in desc_lower or 'shop' in desc_lower:
                        location = 'workplace'
                    elif 'market' in desc_lower:
                        location = 'Local Market'
                    
                    # Determine energy impact
                    energy_impact = 'moderate'
                    if activity_type in ['work', 'travel']:
                        energy_impact = 'high'
                    elif activity_type in ['rest', 'meal']:
                        energy_impact = 'low'
                    
                    activity = {
                        'time': time,
                        'type': activity_type,
                        'description': description,
                        'location': location,
                        'energy_impact': energy_impact
                    }
                    activities.append(activity)
                
                print(f"[DEBUG] Successfully parsed {len(activities)} activities from schedule")
                return activities
            
            # If not a dictionary, try to parse from text
            print("[DEBUG] Attempting to parse activities from text format")
            activities = []
            for line in str(plan_content).split('\n'):
                if ' - ' in line and not line.startswith('Activities:'):
                    try:
                        parts = line.strip().split(' - ')
                        if len(parts) >= 3:
                            activity = {
                                'time': parts[0].strip(),
                                'type': parts[1].strip(),
                                'description': parts[2].strip(),
                                'location': parts[3].strip() if len(parts) > 3 else 'unspecified',
                                'energy_impact': parts[4].strip() if len(parts) > 4 else 'moderate'
                            }
                            activities.append(activity)
                    except Exception as e:
                        print(f"[DEBUG] Error parsing activity line: {line}")
                        continue
            
            print(f"[DEBUG] Successfully parsed {len(activities)} activities from text")
            return activities
                
        except Exception as e:
            print(f"[DEBUG] Error parsing plan to activities: {str(e)}")
            traceback.print_exc()
            return []

    def add_activity(self, activity: Dict[str, Any]) -> None:
        """Add an activity to the agent's memory.
        
        Args:
            activity: Dictionary containing activity details
        """
        try:
            print(f"\n[DEBUG] Adding activity: {activity}")
            
            # Ensure required fields
            if not all(k in activity for k in ['time', 'activity_type', 'description']):
                print("[DEBUG] Activity missing required fields")
                return
                
            # Log as activity event
            self.record_memory('ACTIVITY_EVENT', {
                'time': activity['time'],
                'activity_type': activity['activity_type'],
                'activity_description': activity['description'],
                'location': activity.get('location', self.get_current_location_name()),
                'energy_impact': activity.get('energy_impact', 'neutral')
            })
            
            print("[DEBUG] Activity logged successfully")
            
        except Exception as e:
            print(f"[DEBUG] Error adding activity: {str(e)}")
            traceback.print_exc()

    def needs_food(self, current_time: int) -> bool:
        """Check if agent needs food based on energy level and meal timing.
        
        Returns True if:
        - Energy level is critically low (≤ 30)
        - It's meal time and haven't eaten that meal yet
        - Work priority is respected - won't seek food if should be at work
        """
        try:
            current_hour = current_time % 24
            
            # CRITICAL: Do not seek food if it's work time and agent should be at work
            if self.is_work_time():
                workplace_name = getattr(self, 'workplace', None)
                current_location_name = self.get_current_location_name()
                
                # If it's work time and agent is not at workplace, prioritize getting to work
                # BUT make exception for critically low energy (≤ 30)
                if workplace_name and current_location_name != workplace_name and self.energy_level > 30:
                    print(f"DEBUG FOOD: {self.name} - work time priority over food (should be at {workplace_name}, currently at {current_location_name})")
                    return False
            
            # Check energy level - this should be the primary trigger for urgent food needs
            if self.energy_level <= 30:  # Very low energy always triggers urgent food need
                print(f"DEBUG FOOD: {self.name} - urgent food need due to low energy ({self.energy_level})")
                return True

            # Determine current meal type
            meal_type = self.determine_meal_type(current_hour)
            
            # If not meal time and energy is not critical, no need for food
            if not meal_type:
                return False
                
            # Check if we've already handled the current meal type
            if meal_type in self.meals_today and self.meals_today[meal_type].get('handled', False):
                print(f"DEBUG FOOD: {self.name} - {meal_type} already handled today")
                return False
                
            # Check if it's meal time and we haven't eaten this meal yet
            is_meal_time, detected_meal_type = self.is_meal_time(current_time)
            if is_meal_time and detected_meal_type == meal_type and not self.meals_today[meal_type].get('handled', False):
                # If at home and have groceries, can eat at home (always counts as needing food)
                if self.get_current_location_name() == self.residence and self.grocery_level >= 10:
                    print(f"DEBUG FOOD: {self.name} - meal time ({meal_type}) and can cook at home")
                    return True
                    
                # If not at home, only seek food if energy is getting low (60 or below) or already traveling to food location
                if self.energy_level <= 60:
                    print(f"DEBUG FOOD: {self.name} - meal time ({meal_type}) and energy getting low ({self.energy_level})")
                    return True
                    
                # Or if already traveling to a food location for this meal
                if self.is_traveling:
                    destination = getattr(self, 'travel_destination', None)
                    if destination and destination in self.locations:
                        dest_location = self.locations[destination]
                        if (dest_location.type in ['restaurant', 'local_shop'] or 
                            'diner' in destination.lower() or 
                            'fried chicken' in destination.lower() or
                            ('coffee' in destination.lower() and meal_type in ['breakfast', 'snack'])):
                            print(f"DEBUG FOOD: {self.name} - already traveling to food location {destination} for {meal_type}")
                            return False  # Already being handled
            
            return False
            
        except Exception as e:
            print(f"Error checking food needs for {self.name}: {str(e)}")
            return False

    def is_meal_time(self, current_time: int) -> tuple[bool, Optional[str]]:
        """Check if current time is during a meal period"""
        hour = current_time % 24
        
        # Define meal time windows
        meal_times = {
            'breakfast': range(6, 9),    # 6-8 AM
            'lunch': range(11, 14),      # 11 AM - 1 PM
            'dinner': range(17, 20)      # 5-7 PM
        }
        
        # Check which meal time it is
        for meal_type, time_range in meal_times.items():
            if hour in time_range:
                return True, meal_type
                
        return False, None

    def determine_meal_type(self, current_hour: int) -> Optional[str]:
        """Determine what type of meal is appropriate for the current hour.
        
        Args:
            current_hour: Hour of the day (0-23)
            
        Returns:
            Meal type string or None if not meal time
        """
        # Use is_meal_time logic but with hour parameter
        meal_times = {
            'breakfast': range(6, 9),    # 6-8 AM
            'lunch': range(11, 14),      # 11 AM - 1 PM
            'dinner': range(17, 20)      # 5-7 PM
        }
        
        # Check which meal time it is
        for meal_type, time_range in meal_times.items():
            if current_hour in time_range:
                return meal_type
        
        # Check for snack times (outside main meal windows but when awake)
        if 9 <= current_hour < 11 or 14 <= current_hour < 17 or 20 <= current_hour <= 22:
            return 'snack'
                
        return None

    def make_meal(self, meal_type: str) -> Dict[str, Any]:
        """Make and eat a meal at home using groceries during appropriate meal times only.
        
        Args:
            meal_type: Type of meal to make
            
        Returns:
            Dictionary with success status and details
        """
        try:
            # Check if at home
            if self.get_current_location_name() != self.residence:
                return {'success': False, 'reason': 'Not at home'}
            
            # Check if have enough groceries
            grocery_cost = 20  # Cost in grocery points to make a meal
            if self.grocery_level < grocery_cost:
                return {'success': False, 'reason': 'Not enough groceries'}
            
            # CRITICAL: Only allow cooking during appropriate meal times
            current_time = getattr(self, 'current_time', 0)
            current_hour = current_time % 24
            current_meal_type = self.determine_meal_type(current_hour)
            
            # Restrict cooking to current meal time only
            if meal_type != current_meal_type:
                if current_meal_type:
                    return {'success': False, 'reason': f'Cannot cook {meal_type} at {current_hour:02d}:00 - this is {current_meal_type} time'}
                else:
                    return {'success': False, 'reason': f'Cannot cook {meal_type} at {current_hour:02d}:00 - not a meal time'}
            
            # Check if already eaten this meal today
            if self.meals_today[meal_type]['handled']:
                return {'success': False, 'reason': f'Already had {meal_type} today'}
            
            # Consume groceries and gain energy immediately (no meal prep, immediate consumption)
            self.grocery_level = max(0, self.grocery_level - grocery_cost)
            energy_gain = 20  # Home-cooked meals give +20 energy
            self.energy_level = min(100, self.energy_level + energy_gain)
            
            # Update meals_today status
            self.meals_today[meal_type] = {'handled': True, 'method': 'home_cooking'}
            
            # Log meal preparation and consumption as a single event
            self.log_memory('ACTIVITY_EVENT', {
                'time': current_time,
                'activity_type_tag': ACTIVITY_TYPES['DINING'],
                'description': f'Cooked and ate {meal_type} at home',
                'location': self.residence,
                'energy_gain': energy_gain,
                'grocery_cost': grocery_cost
            })
            
            return {
                'success': True, 
                'energy_gain': energy_gain,
                'grocery_cost': grocery_cost,
                'message': f'Successfully made and ate {meal_type} at home'
            }
            
        except Exception as e:
            print(f"Error making meal for {self.name}: {e}")
            return {'success': False, 'reason': f'Error: {str(e)}'}

    def make_purchase(self, location_name: str, item_type: str, item_description: str) -> Dict[str, Any]:
        """Make a purchase at a location.
        
        Args:
            location_name: Name of the location to purchase from
            item_type: Type of item (meal, groceries, etc.)
            item_description: Description of what's being purchased
            
        Returns:
            Dictionary with success status and details
        """
        try:
            # Get location
            if location_name not in self.locations:
                return {'success': False, 'reason': 'Location not found'}
            
            location = self.locations[location_name]
            
            # Check if location is open
            current_hour = getattr(self, 'current_time', 0) % 24
            if not location.is_open(current_hour):
                return {'success': False, 'reason': 'Location is closed'}
            
            # Get current price
            current_time = getattr(self, 'current_time', 0)
            price = location.get_current_price(current_time)
            
            # Check if have enough money
            if self.money < price:
                return {'success': False, 'reason': f'Not enough money (need ${price:.2f}, have ${self.money:.2f})'}
            
            # Make purchase
            self.money -= price
            
            # Apply effects based on item type
            energy_gain = 0
            grocery_gain = 0
            
            if item_type == 'meal' or item_type == 'food':
                # Restaurant meals give +40 energy
                energy_gain = 40
                self.energy_level = min(100, self.energy_level + energy_gain)
                
                # Update meals_today status for food purchases
                # Determine meal type based on current time
                current_hour = current_time % 24
                meal_type = self.determine_meal_type(current_hour)
                if meal_type and meal_type in self.meals_today:
                    self.meals_today[meal_type] = {'handled': True, 'method': 'dining_out'}
                    
            elif item_type == 'groceries':
                # Groceries replenish grocery level
                grocery_gain = 50
                self.grocery_level = min(100, self.grocery_level + grocery_gain)
            elif item_type == 'beverages_and_snacks':
                # Coffee and snacks give +5 energy
                energy_gain = 5
                self.energy_level = min(100, self.energy_level + energy_gain)
                
                # Update meals_today for snacks if it's snack time
                current_hour = current_time % 24
                meal_type = self.determine_meal_type(current_hour)
                if meal_type == 'snack' and 'snack' in self.meals_today:
                    self.meals_today['snack'] = {'handled': True, 'method': 'dining_out'}
            
            # Log purchase with correct memory type based on item being purchased
            if item_type in ['meal', 'food', 'beverages_and_snacks']:
                memory_type = 'FOOD_PURCHASE_EVENT'
            elif item_type == 'groceries':
                memory_type = 'GROCERY_PURCHASE_EVENT'
            else:
                memory_type = 'GENERIC_EVENT'  # Fallback for other purchases
                
            self.log_memory(memory_type, {
                'time': current_time,
                'location': location_name,
                'item_type': item_type,
                'item_description': item_description,
                'amount': price,  # Changed from 'price' to 'amount' to match memory validation
                'energy_gain': energy_gain,
                'grocery_gain': grocery_gain
            })
            
            # Record in metrics if available
            if hasattr(self, 'metrics') and self.metrics:
                # Calculate discount information for metrics
                current_day = (current_time // 24) + 1
                base_price = getattr(location, 'base_price', price)
                discount_applied = False
                discount_value = 0.0
                
                # Check for discount (specifically for Fried Chicken Shop Wed/Thu discount)
                if hasattr(location, 'discounts') and location.discounts and location_name == "Fried Chicken Shop":
                    if current_day in [3, 4]:  # Wednesday and Thursday
                        discount_rate = location.discounts.get('rate', 0.2)  # Default 20%
                        discount_value = base_price * discount_rate
                        discount_applied = True
                
                # Prepare detailed purchase data for metrics
                purchase_details = {
                    'base_price': base_price,
                    'final_price': price,
                    'quantity': 1,  # Each purchase is 1 item/meal
                    'used_discount': discount_applied,
                    'discount_value': discount_value,
                    'item_type': item_type,
                    'item_description': item_description,
                    'energy_gain': energy_gain,
                    'grocery_gain': grocery_gain
                }
                
                # Fix parameter order: agent_name, location, time, day, amount, details
                self.metrics.record_purchase(
                    self.name,              # agent_name (FIXED ORDER)
                    location_name,          # location (FIXED ORDER)
                    current_time,           # time
                    current_day,            # day (FIXED ORDER)
                    price,                  # amount (final price paid)
                    purchase_details        # details with all required data
                )
            
            return {
                'success': True,
                'price': price,
                'amount': price,  # Add amount field for compatibility
                'energy_gain': energy_gain,
                'grocery_gain': grocery_gain,
                'message': f'Successfully purchased {item_description} at {location_name} for ${price:.2f}'
            }
            
        except Exception as e:
            print(f"Error making purchase for {self.name}: {e}")
            return {'success': False, 'reason': f'Error: {str(e)}'}

    def _perform_travel_step(self) -> Tuple[Optional[str], bool, Optional[Dict[str, Any]]]:
        """Execute one step of travel along the path.
        
        Returns:
            Tuple[Optional[str], bool, Optional[Dict[str, Any]]]: 
            - Message about the travel step
            - Whether travel is continuing
            - Encounter information if any (None if no encounter)
        """
        try:
            if not self.is_traveling or not self.travel_path or self.travel_step_index >= len(self.travel_path):
                self.clear_invalid_travel_state()
                return None, False, None

            # Get current and next coordinates
            current_coord = self.travel_path[self.travel_step_index]
            next_coord = self.travel_path[self.travel_step_index + 1] if self.travel_step_index + 1 < len(self.travel_path) else None

            if not next_coord:
                # Pre-update tracker before completing travel
                if self.shared_tracker and self.travel_destination and self.town_map:
                    destination_coord = self.town_map.get_coordinates_for_location(self.travel_destination)
                    self.shared_tracker.update_agent_position(
                        self.name,
                        self.travel_destination,
                        destination_coord,
                        self.current_time
                    )
                print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Reached destination: {self.travel_destination}. Energy: {self.energy_level}")
                return self._complete_travel(), False, None

            # Move to next coordinate
            self.travel_step_index += 1
            self.current_grid_position = next_coord

            # Update shared location tracker with travel status
            if self.shared_tracker:
                self.shared_tracker.update_agent_position(
                    self.name,
                    f"Traveling to {self.travel_destination}",
                    next_coord,
                    self.current_time
                )

            # Check if we've reached a location at this coordinate
            if self.town_map:
                location_name = self.town_map.get_location_name_at_coord(next_coord)
                if location_name and location_name == self.travel_destination:
                    # Pre-update tracker before completing travel
                    if self.shared_tracker:
                        self.shared_tracker.update_agent_position(
                            self.name,
                            self.travel_destination,
                            next_coord,
                            self.current_time
                        )
                    print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Reached destination: {self.travel_destination}. Energy: {self.energy_level}")
                    return self._complete_travel(), False, None
                
                # Check for mid-travel location encounter
                if location_name and location_name != self.travel_destination:
                    # Found a location during travel
                    location = self.locations.get(location_name)
                    if location and location.is_open(self.current_time):
                        # Build context for location visit decision
                        visit_context = {
                            'destination': self.travel_destination,
                            'encountered_location': location_name,
                            'location_type': location.type,
                            'time': self.current_time,
                            'energy_level': self.energy_level,
                            'money': self.money,
                            'urgency_level': 'high' if self.is_work_time() else 'moderate',
                            'location_offers': f"Current price: ${location.get_current_price(self.current_time):.2f}",
                            'daily_plan': self.daily_plan if self.daily_plan else 'No specific plans'
                        }
                        
                        # Get structured decision about visiting
                        visit_decision = self.generate_structured_location_visit(visit_context)
                        
                        if visit_decision.get('visit_location') == 'YES':
                            # Agent decides to visit - update tracker to show they're at this location
                            if self.shared_tracker:
                                self.shared_tracker.update_agent_position(
                                    self.name,
                                    location_name,
                                    next_coord,
                                    self.current_time
                                )
                            print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Decided to visit {location_name} while traveling to {self.travel_destination}. Energy: {self.energy_level}")
                            
                            # Handle quick stop vs regular visit
                            visit_duration = visit_decision.get('visit_duration', 'QUICK')
                            if visit_duration == 'QUICK':
                                # Record quick stop activity
                                self.record_memory('ACTIVITY_EVENT', {
                                    'type': 'QUICK_STOP',
                                    'location': location_name,
                                    'description': f"Quick stop at {location_name}",
                                    'duration': 'QUICK',
                                    'action_plan': visit_decision.get('action_plan', '')
                                })
                                # Record location event
                                self.record_memory('LOCATION_EVENT', {
                                    'type': 'VISIT',
                                    'location': location_name,
                                    'description': f"Quick stop at {location_name}",
                                    'duration': 'QUICK'
                                })
                                # Update current activity
                                self.current_activity = 'QUICK_STOP'
                            
                            return f"{self.name} decided to visit {location_name} while traveling to {self.travel_destination}", True, {
                                'type': 'location',
                                'location_name': location_name,
                                'location_type': location.type,
                                'visit_duration': visit_duration,
                                'action_plan': visit_decision.get('action_plan', ''),
                                'reasoning': visit_decision.get('reasoning', ''),
                                'activity_type': 'QUICK_STOP' if visit_duration == 'QUICK' else 'REGULAR_VISIT'
                            }
                        else:
                            # Agent decides to continue travel - ensure tracker shows travel status
                            if self.shared_tracker:
                                self.shared_tracker.update_agent_position(
                                    self.name,
                                    f"Traveling to {self.travel_destination}",
                                    next_coord,
                                    self.current_time
                                )
                            print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Passing by {location_name} while traveling to {self.travel_destination}. Energy: {self.energy_level}")
                            return f"{self.name} passed by {location_name} while traveling to {self.travel_destination}", True, {
                                'type': 'location',
                                'location_name': location_name,
                                'location_type': location.type,
                                'visit_duration': 'NONE',
                                'reasoning': visit_decision.get('reasoning', 'Decided to continue travel')
                            }

            # Check for mid-travel agent encounter
            if self.shared_tracker:
                agents_at_coord = self.shared_tracker.get_agents_at_coordinate(next_coord)
                if agents_at_coord:
                    # Found agents during travel
                    encounter_info = {
                        'type': 'agent',
                        'agent_names': agents_at_coord
                    }
                    
                    # Check if we should interact with any of the encountered agents
                    should_interact = False
                    other_agent = None
                    for agent_name in agents_at_coord:
                        # Find the agent object
                        for agent in self.all_agents_list_for_perception:
                            if agent.name == agent_name:
                                if self.should_interact_with(agent, self.current_time, encounter_info):
                                    should_interact = True
                                    other_agent = agent
                                    break
                        if should_interact:
                            break
                    
                    if should_interact and other_agent:
                        # Update tracker to show interaction
                        if self.shared_tracker:
                            self.shared_tracker.update_agent_position(
                                self.name,
                                f"Interacting with {other_agent.name}",
                                next_coord,
                                self.current_time
                            )
                        print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Encountered {other_agent.name} while traveling. Energy: {self.energy_level}")
                        # Let the simulation handle the conversation
                        return f"{self.name} decided to interact with {other_agent.name} while traveling to {self.travel_destination}", True, {
                            'type': 'agent',
                            'agent_names': [other_agent.name],
                            'should_interact': True,
                            'other_agent': other_agent
                        }
                    else:
                        # Continue travel without interaction
                        if self.shared_tracker:
                            self.shared_tracker.update_agent_position(
                                self.name,
                                f"Traveling to {self.travel_destination}",
                                next_coord,
                                self.current_time
                            )
                        print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Passing by {', '.join(agents_at_coord)} while traveling to {self.travel_destination}. Energy: {self.energy_level}")
                        return f"{self.name} passed by {', '.join(agents_at_coord)} while traveling to {self.travel_destination}", True, encounter_info

            # Calculate progress
            total_steps = len(self.travel_path) - 1
            current_step = self.travel_step_index
            progress = (current_step / total_steps) * 100

            # Update energy based on travel
            self.energy_level = max(0, self.energy_level - self.energy_cost_per_hour_travel)

            # Generate travel update message
            travel_message = f"{self.name} is traveling to {self.travel_destination}. Progress: {progress:.1f}%"
            print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] {travel_message}. Energy: {self.energy_level}")
            
            return travel_message, True, None

        except Exception as e:
            print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [ERROR] Error in travel step: {str(e)}")
            print(f"Context: {json.dumps({
                'current_location': self.get_current_location_name(),
                'target_location': self.travel_destination,
                'energy_level': self.energy_level,
                'travel_step': self.travel_step_index,
                'path_length': len(self.travel_path) if self.travel_path else 0,
                'is_traveling': self.is_traveling,
                'current_coord': self.current_grid_position
            }, indent=2)}")
            traceback.print_exc()
            self.clear_invalid_travel_state()
            return None, False, None

    def _complete_travel(self) -> str:
        """Complete the travel and update agent's location."""
        try:
            if not self.travel_destination or not self.town_map:
                return self.clear_invalid_travel_state()

            # Get the destination location
            destination_coord = self.town_map.get_coordinates_for_location(self.travel_destination)
            if not destination_coord:
                return self.clear_invalid_travel_state()

            # Calculate energy cost for travel
            path_length = len(self.travel_path)
            travel_cost = self.calculate_travel_cost(path_length)
            self.energy_level = max(0, self.energy_level - travel_cost['energy'])
            
            # Log energy update
            self.record_memory('ENERGY_UPDATE_EVENT', {
                'time': self.current_time,
                'previous_energy': self.energy_level + travel_cost['energy'],
                'current_energy': self.energy_level,
                'change': -travel_cost['energy'],
                'reason': f'Travel to {self.travel_destination}',
                'location': self.travel_destination
            })

            # Update agent's location
            if self.travel_destination in self.locations:
                self.current_location = self.locations[self.travel_destination]
                self.current_grid_position = destination_coord

                # Update shared location tracker
                if self.shared_tracker:
                    self.shared_tracker.update_agent_position(
                        self.name,
                        self.travel_destination,
                        destination_coord,
                        self.current_time
                    )

                # Clear travel state
                self.is_traveling = False
                self.travel_destination = None
                self.travel_path = []
                self.travel_step_index = 0

                return f"{self.name} has arrived at {self.travel_destination}"

            return self.clear_invalid_travel_state()

        except Exception as e:
            print(f"Error completing travel for {self.name}: {str(e)}")
            traceback.print_exc()
            return self.clear_invalid_travel_state()

    def get_current_location_name(self) -> str:
        """Get the name of the current location."""
        try:
            if self.current_location is None:
                return self.residence  # Return residence as fallback
            
            if isinstance(self.current_location, str):
                return self.current_location  # Return string directly if it's a string
            
            if hasattr(self.current_location, 'name'):
                return self.current_location.name  # Return name if it's a Location object
            
            return self.residence  # Return residence as final fallback
            
        except Exception as e:
            print(f"Error getting current location name for {self.name}: {str(e)}")
            return self.residence  # Return residence as error fallback

    def get_recent_activities(self, limit: int = 3) -> List[str]:
        """Get the agent's most recent activities.
        
        Args:
            limit: Maximum number of activities to return
            
        Returns:
            List[str]: List of recent activity descriptions
        """
        if hasattr(self, 'memory_manager') and self.memory_manager:
            try:
                # Query recent memories using retrieve_memories
                recent_memories = self.memory_manager.retrieve_memories(
                    self.name,
                    self.current_time if hasattr(self, 'current_time') else 0,
                    memory_type_key='ACTION_RAW_OUTPUT',
                    limit=limit
                )
                
                # Extract activity descriptions from memories
                activities = []
                for memory in recent_memories:
                    if isinstance(memory, dict):
                        # Try to get content from various possible memory formats
                        if 'data' in memory:
                            content = memory['data'].get('content', '')
                        else:
                            content = memory.get('content', memory.get('description', memory.get('action', '')))
                        if content:
                            activities.append(content)
                    elif isinstance(memory, str):
                        activities.append(memory)
                
                # If no memories found, return basic status
                if not activities:
                    current_location = self.get_current_location_name()
                    if self.is_traveling:
                        return [f"Currently traveling to {self.travel_destination}"]
                    else:
                        return [f"At {current_location}"]
                
                return activities
                
            except Exception as e:
                print(f"Error getting recent activities for {self.name}: {str(e)}")
                return [f"At {self.get_current_location_name()}"]  # Fallback to basic location info
        
        return ["No recent activities available"]

    def create_daily_plan(self, current_hour: int, context: Dict[str, Any]) -> Optional[str]:
        """Create a daily plan for the agent."""
        try:
            print(f"\n[DEBUG] Creating daily plan for {self.name} at {current_hour}:00")
            print(f"[DEBUG] Current location: {self.get_current_location_name()}")
            print(f"[DEBUG] Energy level: {self.energy_level}")
            
            # Add current state to context
            context.update({
                'current_location': self.get_current_location_name(),
                'energy_level': self.energy_level,
                'money': self.money,
                'grocery_level': self.grocery_level,
                'work_schedule': self.work_schedule,
                'household_members': [m['name'] for m in self.household_members] if self.household_members else []
            })
            
            # Get prompt for daily plan
            prompt = self.prompt_mgr.get_prompt('daily_plan', **context)
            print(f"[DEBUG] Daily plan prompt length: {len(prompt)} characters")
            
            # Generate structured plan
            print("[DEBUG] Generating structured plan...")
            plan_result = self.model_mgr.generate_structured(prompt, {
                'schedule': str,
                'reasoning': str,
                'energy_considerations': str
            })
            
            if plan_result:
                print("[DEBUG] Plan generated successfully")
                print(f"[DEBUG] Plan content: {plan_result}")
                
                # Format the plan into a string
                plan_str = f"Schedule: {plan_result.get('schedule', '')}\n"
                plan_str += f"Reasoning: {plan_result.get('reasoning', '')}\n"
                plan_str += f"Energy Considerations: {plan_result.get('energy_considerations', '')}"
                
                # Store the raw plan result for later use
                self.daily_plan = plan_str
                
                # Parse activities from the plan
                print("[DEBUG] Parsing activities from plan...")
                activities = self.parse_llm_plan_to_activity(plan_str)
                if activities:
                    print(f"[DEBUG] Parsed {len(activities)} activities from plan")
                    for activity in activities:
                        print(f"[DEBUG] Adding activity: {activity}")
                        self.add_activity(activity)
                else:
                    print("[DEBUG] No activities parsed from plan")
                
                # Log the planning event
                print("[DEBUG] Logging planning event...")
                self.record_memory('PLANNING_EVENT', {
                    'time': current_hour,
                    'content': plan_str,
                    'plan_type': 'daily_plan',
                    'location': self.get_current_location_name(),
                    'energy_level': self.energy_level,
                    'money': self.money,
                    'grocery_level': self.grocery_level
                })
                print("[DEBUG] Planning event logged successfully")
                
                # Log system event for plan creation
                self.record_memory('SYSTEM_EVENT', {
                    'time': current_hour,
                    'event_type': 'daily_plan_created',
                    'content': f"Created new daily plan at {current_hour}:00",
                    'location': self.get_current_location_name()
                })
                
                return plan_str
            else:
                print("[DEBUG] Failed to generate plan")
                return None
                
        except Exception as e:
            print(f"[DEBUG] Error in create_daily_plan: {str(e)}")
            traceback.print_exc()
            return None

    def update_state(self):
        """Update agent state and log changes."""
        state_data = {
            'time': self.current_time,
            'location': self.get_current_location_name(),
            'energy': self.energy_level,
            'traveling': self.is_traveling
        }
        self.log_memory('AGENT_STATE_UPDATE_EVENT', state_data)

    def start_travel_to(self, target_location_name: str, desired_arrival_time: Optional[int] = None) -> str:
        """Start traveling to a target location."""
        try:
            # Create travel memory data
            memory_data = {
                'target_location': target_location_name,
                'start_location': self.get_current_location_name(),
                'start_time': self.current_time,
                'desired_arrival_time': desired_arrival_time,
                'status': 'started'
            }
            
            # Record as TRAVEL_EVENT
            self.record_memory('TRAVEL_EVENT', memory_data)
            
            # Update travel state
            self.is_traveling = True
            self.travel_destination = target_location_name
            self.travel_start_location = self.get_current_location_name()
            
            # Find path to destination
            if self.town_map:
                start_coord = self.current_grid_position
                end_coord = self.town_map.get_coordinates_for_location(target_location_name)
                
                if start_coord and end_coord:
                    self.travel_path = self.town_map.find_path(start_coord, end_coord)
                    self.travel_step_index = 0
                    
                    if not self.travel_path:
                        return f"Could not find path to {target_location_name}"
                    
                    return f"Starting travel to {target_location_name}"
                else:
                    return f"Could not find coordinates for {target_location_name}"
            else:
                return "No town map available for pathfinding"
            
        except Exception as e:
            print(f"Error starting travel: {str(e)}")
            traceback.print_exc()
            return "Error starting travel"

    def _calculate_movement_speed(self) -> int:
        """Calculate movement speed in steps per hour.
        
        Returns:
            int: Fixed at 15 steps per hour for all agents
        """
        return 15  # Fixed speed regardless of energy level
    
    def calculate_travel_cost(self, path_length: int) -> Dict[str, int]:
        """Calculate the energy and time cost for traveling a path."""
        try:
            # Calculate energy cost based on path length and energy cost per step
            energy_cost = path_length * ENERGY_COST_PER_STEP
            
            # Calculate time cost based on movement speed
            movement_speed = self._calculate_movement_speed()
            time_cost = (path_length + movement_speed - 1) // movement_speed
            
            return {
                'energy': energy_cost,
                'time': time_cost
            }
            
        except Exception as e:
            print(f"Error calculating travel cost: {str(e)}")
            return {'energy': 0, 'time': 0}

    def calculate_grocery_purchase_amount(self, available_money: float, price_per_unit: float) -> float:
        """Calculate how much money to spend on groceries based on available funds.
        
        Args:
            available_money: Total money available
            price_per_unit: Base price for groceries
            
        Returns:
            float: Amount to spend on groceries
        """
        # Spend up to 25% of available money on groceries, but at least the base price
        max_spend = available_money * 0.25
        return max(min(max_spend, price_per_unit * 2), price_per_unit) if available_money >= price_per_unit else 0

    def _clean_action_for_meal_system(self, action_str: str, current_time: int) -> str:
        """Clean agent action strings that mention unsupported meal preparation like 'packed lunch'.
        
        Args:
            action_str: The raw action string from the agent
            current_time: Current simulation time
            
        Returns:
            Cleaned action string that works with the meal system
        """
        try:
            current_hour = current_time % 24
            is_meal_time_result, meal_type = self.is_meal_time(current_time)
            
            # Check for problematic "packed lunch" or advance meal prep mentions
            problematic_patterns = [
                "packed lunch", "pack lunch", "packed meal", "pack meal",
                "bring lunch", "brought lunch", "lunch from home",
                "prepared lunch", "made lunch earlier", "lunch I packed"
            ]
            
            action_lower = action_str.lower()
            has_problematic_pattern = any(pattern in action_lower for pattern in problematic_patterns)
            
            if has_problematic_pattern:
                print(f"DEBUG MEAL SYSTEM: {self.name} mentioned unsupported packed lunch - redirecting")
                
                # If it's meal time, redirect to proper meal handling
                if is_meal_time_result and meal_type:
                    if self.energy_level <= 35:
                        # Critical energy - must eat at restaurant
                        return f"My energy is critically low ({self.energy_level}/100) and I need a proper {meal_type}. I should go to a restaurant like the Fried Chicken Shop or Local Diner for substantial nourishment."
                    elif self.grocery_level > 20:
                        # Can cook at home
                        return f"It's {meal_type} time and I have groceries. I'll cook a proper {meal_type} at home to restore my energy."
                    else:
                        # Low groceries - eat out
                        return f"It's {meal_type} time but my groceries are low. I should visit a restaurant for {meal_type}."
                
                # If not meal time, redirect the action appropriately
                elif "lunch" in action_lower and 11 <= current_hour <= 14:
                    # It's lunch time but they mentioned packed lunch
                    if self.energy_level <= 35:
                        return f"I'm at lunch time with critically low energy ({self.energy_level}/100). I need to go to a restaurant for a proper meal."
                    else:
                        return f"It's lunch time. I should either cook at home if I have groceries, or visit a restaurant for lunch."
                
                else:
                    # Non-meal time - redirect to current activity
                    return f"I should focus on my current activities. The meal system only allows cooking during proper meal times."
            
            return action_str
            
        except Exception as e:
            print(f"Error cleaning action for meal system: {e}")
            return action_str

    def generate_contextual_action(self, simulation: 'Simulation', current_time: int) -> str:
        """Generate a contextual action based on current state and time."""
        try:
            # Get context for the agent
            if hasattr(simulation, 'get_agent_context'):
                context = simulation.get_agent_context(self)
            else:
                # Fallback to basic context if simulation object is not properly passed
                context = {
                    'name': self.name,
                    'age': self.personal_info.get('age', 25),
                    'occupation': self.personal_info.get('occupation', 'unknown'),
                    'residence': self.residence,
                    'current_location': self.get_current_location_name(),
                    'energy_level': self.energy_level,
                    'money': self.money,
                    'grocery_level': self.grocery_level,
                    'daily_income': self._calculate_daily_wage(),
                    'work_schedule_start': self.work_schedule.get('start', 9) if self.work_schedule else 9,
                    'work_schedule_end': self.work_schedule.get('end', 17) if self.work_schedule else 17,
                    'workplace': self.workplace,
                    'current_time': current_time,
                    'recent_activities': self.get_recent_activities(),
                    'available_locations': [loc.name for loc in simulation.locations.values()] if hasattr(simulation, 'locations') else []
                }
            
            # If no daily plan, determine default activity
            if not self.daily_plan:
                return self._determine_default_activity(current_time)
            
            # Extract current hour activity from daily plan
            current_hour = current_time % 24
            plan_lower = self.daily_plan.lower()
            
            # Check for specific hour mentions
            hour_patterns = [
                f"{current_hour:02d}:00",
                f"{current_hour}:00",
                f"{current_hour} am" if current_hour < 12 else f"{current_hour-12 if current_hour > 12 else current_hour} pm",
                f"at {current_hour}"
            ]
            
            for pattern in hour_patterns:
                if pattern in plan_lower:
                    # Extract the sentence containing this time
                    sentences = self.daily_plan.split('.')
                    for sentence in sentences:
                        if pattern in sentence.lower():
                            return f"Following daily plan: {sentence.strip()}"
            
            # Fallback: general time-based activities
            if 6 <= current_hour <= 10:
                if 'breakfast' in plan_lower or 'morning' in plan_lower:
                    return "Following daily plan: Morning routine and breakfast"
            elif 11 <= current_hour <= 14:
                if 'lunch' in plan_lower:
                    return "Following daily plan: Lunch time activities"
            elif 17 <= current_hour <= 20:
                if 'dinner' in plan_lower or 'evening' in plan_lower:
                    return "Following daily plan: Evening routine and dinner"
            
            # If no specific activity found, return general plan-based activity
            return "Following daily plan activities"
                
        except Exception as e:
            print(f"Error generating contextual action for {self.name}: {str(e)}")
            traceback.print_exc()
            return f"{self.name} continuing with normal activities"

    def _format_recent_conversations(self, conversations: List[Dict]) -> str:
        """Format recent conversations for inclusion in household context."""
        conversation_summary = ""
        for conversation in conversations:
            conversation_summary += f"{conversation['speaker']}: {conversation['content']}\n"
        return conversation_summary

    def _get_recent_household_conversations(self, current_time: int, household_members: List['Agent']) -> List[Dict]:
        """Get recent household conversations from memory to continue ongoing dialogues."""
        try:
            participant_names = [member.name for member in household_members] + [self.name]
            recent_conversations = []
            
            # Retrieve conversation memories from the last 6 hours
            memories = self.memory_manager.retrieve_memories(
                self.name, 
                current_time, 
                'CONVERSATION_LOG_EVENT', 
                limit=10
            )
            
            for memory in memories:
                # Check if this is a household conversation
                if (memory.get('location') == self.residence and 
                    'participants' in memory and 
                    any(participant in memory['participants'] for participant in participant_names)):
                    
                    # Check if conversation is recent enough (within 6 hours)
                    memory_time = memory.get('time', 0)
                    if current_time - memory_time <= 6:
                        recent_conversations.append({
                            'time': memory_time,
                            'content': memory.get('content', ''),
                            'participants': memory.get('participants', []),
                            'speaker': memory.get('speaker', self.name)
                        })
            
            # Sort by time (most recent first)
            recent_conversations.sort(key=lambda x: x['time'], reverse=True)
            
            return recent_conversations[:5]  # Return max 5 most recent conversations
            
        except Exception as e:
            print(f"Error retrieving recent household conversations for {self.name}: {str(e)}")
            return []

    def get_relationship_with(self, other_agent_name: str) -> str:
        """Get the relationship with another agent, returning a default if not found."""
        try:
            if hasattr(self, 'relationships') and self.relationships:
                return self.relationships.get(other_agent_name, 'acquaintance')
            else:
                # If no relationships dict, try to infer from household members
                if hasattr(self, 'household_members'):
                    household_member_names = []
                    if isinstance(self.household_members, list):
                        household_member_names = [member.get('name') for member in self.household_members if isinstance(member, dict) and 'name' in member]
                    elif isinstance(self.household_members, dict):
                        household_member_names = list(self.household_members.keys())
                    
                    if other_agent_name in household_member_names:
                        return 'household member'
                
                return 'acquaintance'
        except Exception as e:
            print(f"Error getting relationship with {other_agent_name}: {str(e)}")
            return 'acquaintance'

    def analyze_conversation_outcomes(self, conversation_content: str, participants: List[str], current_time: int) -> Dict[str, Any]:
        """Analyze conversation outcomes and extract relevant information."""
        try:
            print(f"\n[DEBUG] Analyzing conversation outcomes for {self.name}")
            print(f"[DEBUG] Participants: {participants}")
            print(f"[DEBUG] Current time: {current_time}")
            
            # Get prompt for conversation analysis
            context = {
                'conversation_content': conversation_content,
                'participants': participants,
                'current_time': current_time,
                'agent_name': self.name,
                'current_location': self.get_current_location_name()
            }
            
            prompt = self.prompt_mgr.get_prompt('conversation_analysis', **context)
            print(f"[DEBUG] Analysis prompt length: {len(prompt)} characters")
            
            # Generate structured analysis
            analysis_result = self.model_mgr.generate_structured(prompt, {
                'schedule_changes': str,
                'relationship_impact': str,
                'emotional_state': str
            })
            
            if analysis_result:
                print(f"[DEBUG] Conversation analysis result: {analysis_result}")
                
                # Log the analysis
                self.record_memory('PLANNING_EVENT', {
                    'time': current_time,
                    'content': f"Conversation Analysis:\nSchedule Changes: {analysis_result.get('schedule_changes', '')}\nRelationship Impact: {analysis_result.get('relationship_impact', '')}\nEmotional State: {analysis_result.get('emotional_state', '')}",
                    'plan_type': 'conversation_outcome',
                    'location': self.get_current_location_name()
                })
                print("[DEBUG] Logged conversation analysis")
                
                return analysis_result
            else:
                print("[DEBUG] Failed to generate conversation analysis")
                return {}
                
        except Exception as e:
            print(f"[DEBUG] Error in analyze_conversation_outcomes: {str(e)}")
            traceback.print_exc()
            return {}

    def update_schedule_from_conversation(self, conversation_outcome: Dict[str, Any], current_time: int) -> str:
        """Update agent's schedule based on conversation outcomes."""
        try:
            print(f"\n[DEBUG] Updating schedule from conversation for {self.name}")
            print(f"[DEBUG] Conversation outcome: {conversation_outcome}")
            
            if not conversation_outcome or 'schedule_changes' not in conversation_outcome:
                print("[DEBUG] No schedule changes to process")
                return "No schedule changes needed."
            
            # Get current plan
            current_plan = self.daily_plan if hasattr(self, 'daily_plan') else None
            print(f"[DEBUG] Current plan: {current_plan}")
            
            # Generate updated plan
            context = {
                'current_plan': current_plan,
                'schedule_changes': conversation_outcome['schedule_changes'],
                'current_time': current_time,
                'current_location': self.get_current_location_name()
            }
            
            prompt = self.prompt_mgr.get_prompt('schedule_update', **context)
            print(f"[DEBUG] Schedule update prompt length: {len(prompt)} characters")
            
            # Generate updated schedule
            updated_schedule = self.model_mgr.generate_structured(prompt, {
                'updated_schedule': str,
                'reasoning': str,
                'time_impact': str
            })
            
            if updated_schedule:
                print(f"[DEBUG] Updated schedule: {updated_schedule}")
                
                # Format the updated plan
                plan_str = f"Updated Schedule: {updated_schedule.get('updated_schedule', '')}\n"
                plan_str += f"Reasoning: {updated_schedule.get('reasoning', '')}\n"
                plan_str += f"Time Impact: {updated_schedule.get('time_impact', '')}"
                
                # Log the schedule update
                self.record_memory('PLANNING_EVENT', {
                    'time': current_time,
                    'content': plan_str,
                    'plan_type': 'schedule_update',
                    'location': self.get_current_location_name()
                })
                print("[DEBUG] Logged schedule update")
                
                # Update the daily plan
                self.daily_plan = updated_schedule['updated_schedule']
                return f"Schedule updated: {updated_schedule['updated_schedule']}"
            else:
                print("[DEBUG] Failed to generate updated schedule")
                return "Failed to update schedule."
                
        except Exception as e:
            print(f"[DEBUG] Error in update_schedule_from_conversation: {str(e)}")
            traceback.print_exc()
            return "Error updating schedule."

    def clear_invalid_travel_state(self) -> str:
        """Clear invalid travel states where agent is marked as traveling but has no valid destination."""
        try:
            if not self.is_traveling:
                return ""  # Not traveling, nothing to clear
            
            # Check if travel destination is valid
            destination = getattr(self, 'travel_destination', None)
            if not destination or destination not in self.locations:
                print(f"DEBUG TRAVEL: {self.name} clearing invalid travel state - destination: {destination}")
                self.is_traveling = False
                self.travel_destination = None
                self.travel_path = []
                self.travel_step_index = 0
                return f"Cleared invalid travel state for {self.name}"
            
            # Check if travel path is corrupted
            if hasattr(self, 'travel_path') and self.travel_path:
                travel_step_index = getattr(self, 'travel_step_index', 0)
                if travel_step_index >= len(self.travel_path):
                    print(f"DEBUG TRAVEL: {self.name} clearing corrupted travel path - step {travel_step_index} >= path length {len(self.travel_path)}")
                    self.is_traveling = False
                    self.travel_destination = None
                    self.travel_path = []
                    self.travel_step_index = 0
                    return f"Cleared corrupted travel path for {self.name}"
            
            return ""  # Travel state is valid
            
        except Exception as e:
            print(f"Error clearing travel state for {self.name}: {str(e)}")
            # Force clear everything if there's an error
            self.is_traveling = False
            self.travel_destination = None
            self.travel_path = []
            self.travel_step_index = 0
            return f"Force cleared travel state for {self.name} due to error"

    def execute_current_schedule(self, current_time: int) -> str:
        """Execute the appropriate action based on current time and active plan."""
        try:
            # First, clear any invalid travel states
            travel_fix = self.clear_invalid_travel_state()
            if travel_fix:
                print(f"TRAVEL FIX: {travel_fix}")
            
            current_hour = current_time % 24
            active_plan = self.get_current_active_plan(current_time)
            
            print(f"DEBUG SCHEDULE: {self.name} executing schedule at {current_hour}:00")
            print(f"DEBUG SCHEDULE: Active plan type: {active_plan['type']}, source: {active_plan['source']}")
            print(f"DEBUG SCHEDULE: Current location: {self.get_current_location_name()}, is_traveling: {self.is_traveling}")
            
            # Priority 1: Work obligations (ENHANCED - with energy consideration)
            if self.is_work_time():
                workplace = getattr(self, 'workplace', None)
                current_location = self.get_current_location_name()
                
                print(f"DEBUG SCHEDULE: {self.name} - it's work time! Workplace: {workplace}, Current: {current_location}")
                
                if workplace and current_location != workplace:
                    # Check energy level for work urgency
                    if self.energy_level > 20:  # Has energy to travel to work
                        if workplace in self.locations and not self.is_traveling:
                            self.start_travel_to(workplace)
                            return f"{self.name} URGENT: Work time started - traveling to {workplace}"
                        elif self.is_traveling and getattr(self, 'travel_destination', None) != workplace:
                            # Wrong travel destination during work time
                            print(f"DEBUG SCHEDULE: {self.name} redirecting travel to workplace")
                            self.is_traveling = False  # Stop current travel
                            self.start_travel_to(workplace)
                            return f"{self.name} REDIRECTED: Changing travel destination to {workplace} for work"
                        else:
                            return f"{self.name} should go to work at {workplace} but travel issues prevent it"
                    else:
                        return f"{self.name} needs energy before going to work (energy: {self.energy_level})"
                
                elif workplace and current_location == workplace:
                    self.current_activity = f"Working at {workplace}"
                    return f"{self.name} continuing work at {workplace}"
                else:
                    return f"{self.name} work time but no valid workplace configured"
            
            # Priority 2: Shared plans (from conversations)
            if active_plan['type'] == 'shared':
                self.current_activity = active_plan['next_action']
                
                # Try to extract specific actions from shared plan
                plan_content = active_plan['content'].lower()
                if 'meet at' in plan_content or 'go to' in plan_content:
                    # Extract location if mentioned
                    for location_name in self.locations.keys():
                        if location_name.lower() in plan_content:
                            current_location = self.get_current_location_name()
                            if current_location != location_name and not self.is_traveling:
                                self.start_travel_to(location_name)
                                return f"{self.name} following shared plan - traveling to {location_name}"
                            elif current_location == location_name:
                                return f"{self.name} at {location_name} as planned"
                
                return f"{self.name} following shared plan: {active_plan['content'][:50]}..."
            
            # Priority 3: Personal daily plan
            if active_plan['content']:
                # Extract current hour activity from daily plan
                current_activity = self._extract_current_hour_activity(active_plan['content'], current_hour)
                if current_activity:
                    return f"Following daily plan: {current_activity}"
            
            # Priority 4: Basic needs and default activities
            return self._determine_default_activity(current_time)
            
        except Exception as e:
            print(f"Error executing schedule for {self.name}: {str(e)}")
            return f"{self.name} continuing with normal activities"

    def _extract_current_hour_activity(self, daily_plan: str, current_hour: int) -> str:
        """Extract what the agent should be doing at the current hour from their daily plan."""
        try:
            if not daily_plan:
                return ""
            
            # Look for time-specific activities in the daily plan
            plan_lower = daily_plan.lower()
            
            # Check for specific hour mentions
            hour_patterns = [
                f"{current_hour:02d}:00",
                f"{current_hour}:00",
                f"{current_hour} am" if current_hour < 12 else f"{current_hour-12 if current_hour > 12 else current_hour} pm",
                f"at {current_hour}"
            ]
            
            for pattern in hour_patterns:
                if pattern in plan_lower:
                    # Extract the sentence containing this time
                    sentences = daily_plan.split('.')
                    for sentence in sentences:
                        if pattern in sentence.lower():
                            return sentence.strip()
            
            # Fallback: general time-based activities
            if 6 <= current_hour <= 10:
                if 'breakfast' in plan_lower or 'morning' in plan_lower:
                    return "Morning routine and breakfast"
            elif 11 <= current_hour <= 14:
                if 'lunch' in plan_lower:
                    return "Lunch time activities"
            elif 17 <= current_hour <= 20:
                if 'dinner' in plan_lower or 'evening' in plan_lower:
                    return "Evening routine and dinner"
            
            return ""
            
        except Exception as e:
            print(f"Error extracting current hour activity: {str(e)}")
            return ""

    def _determine_default_activity(self, current_time: int) -> str:
        """Determine a default activity when no specific plan is available."""
        try:
            current_hour = current_time % 24
            
            # Check basic needs first
            if self.energy_level < 30:
                return f"{self.name} resting to recover energy (energy: {self.energy_level})"
            
            if self.grocery_level < 20:
                return f"{self.name} considering grocery shopping (groceries: {self.grocery_level}%)"
            
            # Time-based default activities
            if 6 <= current_hour <= 10:
                return f"{self.name} morning routine activities"
            elif 11 <= current_hour <= 14:
                return f"{self.name} midday activities"
            elif 17 <= current_hour <= 20:
                return f"{self.name} evening routine activities"
            elif 21 <= current_hour <= 23:
                return f"{self.name} winding down for the day"
            else:
                return f"{self.name} nighttime rest"
                
        except Exception as e:
            print(f"Error determining default activity: {str(e)}")
            return f"{self.name} continuing with normal activities"

    def log_conversation_memory(self, content: str, participants: List[str], time: int, location: str) -> None:
        """Record a conversation memory."""
        try:
            # Create conversation memory data
            memory_data = {
                'content': content,
                'participants': participants,
                'time': time,
                'location': location,
                'topic': self._extract_conversation_topic(content),
                'sentiment': self._extract_sentiment(content)
            }
            
            # Record as CONVERSATION_EVENT
            self.record_memory('CONVERSATION_EVENT', memory_data)
            
        except Exception as e:
            print(f"Error logging conversation memory: {str(e)}")
            traceback.print_exc()

    def make_purchase(self, location_name: str, item_type: str, item_description: str) -> Dict[str, Any]:
        """Record a purchase event."""
        try:
            # Create purchase memory data
            memory_data = {
                'location': location_name,
                'item_type': item_type,
                'item_description': item_description,
                'time': self.current_time,
                'items': [item_description]  # Store as list for consistency
            }
            
            # Record as PURCHASE_EVENT
            self.record_memory('PURCHASE_EVENT', memory_data)
            
            return memory_data
            
        except Exception as e:
            print(f"Error recording purchase: {str(e)}")
            traceback.print_exc()
            return {}

    def make_meal(self, meal_type: str) -> Dict[str, Any]:
        """Record a meal preparation event."""
        try:
            # Create activity memory data
            memory_data = {
                'activity_type': 'DINING',
                'meal_type': meal_type,
                'location': self.get_current_location_name(),
                'time': self.current_time,
                'description': f"Preparing and eating {meal_type} at home"
            }
            
            # Record as ACTIVITY_EVENT
            self.record_memory('ACTIVITY_EVENT', memory_data)
            
            return memory_data
            
        except Exception as e:
            print(f"Error recording meal: {str(e)}")
            traceback.print_exc()
            return {}

    def update_state(self):
        """Update agent state and record state changes."""
        try:
            # Create state update memory data
            memory_data = {
                'energy_level': self.energy_level,
                'grocery_level': self.grocery_level,
                'money': self.money,
                'current_location': self.get_current_location_name(),
                'current_activity': self.current_activity,
                'time': self.current_time
            }
            
            # Record as AGENT_STATE_UPDATE_EVENT
            self.record_memory('AGENT_STATE_UPDATE_EVENT', memory_data)
            
        except Exception as e:
            print(f"Error updating state: {str(e)}")
            traceback.print_exc()

    def create_daily_plan(self, current_hour: int, context: Dict[str, Any]) -> Optional[str]:
        """Create a daily plan and record it."""
        try:
            # Create planning memory data
            memory_data = {
                'time': current_hour,
                'context': context,
                'plan_type': 'daily',
                'status': 'created'
            }
            
            # Record as PLANNING_EVENT
            self.record_memory('PLANNING_EVENT', memory_data)
            
            # Rest of the planning logic...
            
        except Exception as e:
            print(f"Error creating daily plan: {str(e)}")
            traceback.print_exc()
            return None

    def handle_household_interaction(self, current_time: int) -> str:
        """Handle household interaction and record it."""
        try:
            # Create interaction memory data
            memory_data = {
                'time': current_time,
                'location': self.get_current_location_name(),
                'interaction_type': 'household',
                'participants': self.household_members
            }
            
            # Record as INTERACTION_EVENT
            self.record_memory('INTERACTION_EVENT', memory_data)
            
            # Rest of the interaction logic...
            
        except Exception as e:
            print(f"Error handling household interaction: {str(e)}")
            traceback.print_exc()
            return "Error handling household interaction"

    def _resume_travel_after_encounter(self, encounter_info: Dict[str, Any]) -> str:
        """Resume travel after an encounter, updating state and memory."""
        try:
            # Log the encounter in memory
            self.record_memory('TRAVEL_ENCOUNTER_EVENT', {
                'time': self.current_time,
                'encounter_type': encounter_info['type'],
                'location_name': encounter_info.get('location_name'),
                'location_type': encounter_info.get('location_type'),
                'visit_duration': encounter_info.get('visit_duration', 'NONE'),
                'action_plan': encounter_info.get('action_plan', ''),
                'reasoning': encounter_info.get('reasoning', ''),
                'current_energy': self.energy_level,
                'destination': self.travel_destination
            })

            # If this was a location encounter and we decided to visit
            if encounter_info['type'] == 'location' and encounter_info.get('visit_duration') != 'NONE':
                # Calculate energy cost for the visit
                visit_duration = encounter_info.get('visit_duration', 'QUICK')
                duration_multiplier = {
                    'QUICK': 0.5,  # 5-10 minutes
                    'REGULAR': 1.0  # 15-30 minutes
                }.get(visit_duration, 0.5)
                
                visit_energy_cost = int(ENERGY_COST_PER_HOUR_IDLE * duration_multiplier)
                self.energy_level = max(0, self.energy_level - visit_energy_cost)
                
                # Log energy update for visit
                self.record_memory('ENERGY_UPDATE_EVENT', {
                    'time': self.current_time,
                    'previous_energy': self.energy_level + visit_energy_cost,
                    'current_energy': self.energy_level,
                    'change': -visit_energy_cost,
                    'reason': f'Visit to {encounter_info["location_name"]}',
                    'location': encounter_info['location_name']
                })

            return f"{self.name} continuing travel to {self.travel_destination}"

        except Exception as e:
            print(f"Error resuming travel for {self.name}: {str(e)}")
            return f"{self.name} continuing travel to {self.travel_destination}"

    def _perform_travel_step(self) -> Tuple[Optional[str], bool, Optional[Dict[str, Any]]]:
        """Execute one step of travel along the path.
        
        Returns:
            Tuple[Optional[str], bool, Optional[Dict[str, Any]]]: 
            - Message about the travel step
            - Whether travel is continuing
            - Encounter information if any (None if no encounter)
        """
        try:
            if not self.is_traveling or not self.travel_path or self.travel_step_index >= len(self.travel_path):
                self.clear_invalid_travel_state()
                return None, False, None

            # Get current and next coordinates
            current_coord = self.travel_path[self.travel_step_index]
            next_coord = self.travel_path[self.travel_step_index + 1] if self.travel_step_index + 1 < len(self.travel_path) else None

            if not next_coord:
                # Pre-update tracker before completing travel
                if self.shared_tracker and self.travel_destination and self.town_map:
                    destination_coord = self.town_map.get_coordinates_for_location(self.travel_destination)
                    self.shared_tracker.update_agent_position(
                        self.name,
                        self.travel_destination,
                        destination_coord,
                        self.current_time
                    )
                return self._complete_travel(), False, None

            # Move to next coordinate
            self.travel_step_index += 1
            self.current_grid_position = next_coord

            # Update shared location tracker with travel status
            if self.shared_tracker:
                self.shared_tracker.update_agent_position(
                    self.name,
                    f"Traveling to {self.travel_destination}",
                    next_coord,
                    self.current_time
                )

            # Check if we've reached a location at this coordinate
            if self.town_map:
                location_name = self.town_map.get_location_name_at_coord(next_coord)
                if location_name and location_name == self.travel_destination:
                    # Pre-update tracker before completing travel
                    if self.shared_tracker:
                        self.shared_tracker.update_agent_position(
                            self.name,
                            self.travel_destination,
                            next_coord,
                            self.current_time
                        )
                    return self._complete_travel(), False, None
                
                # Check for mid-travel location encounter
                if location_name and location_name != self.travel_destination:
                    # Found a location during travel
                    location = self.locations.get(location_name)
                    if location and location.is_open(self.current_time):
                        # Build context for location visit decision
                        visit_context = {
                            'destination': self.travel_destination,
                            'encountered_location': location_name,
                            'location_type': location.type,
                            'time': self.current_time,
                            'energy_level': self.energy_level,
                            'money': self.money,
                            'urgency_level': 'high' if self.is_work_time() else 'moderate',
                            'location_offers': f"Current price: ${location.get_current_price(self.current_time):.2f}",
                            'daily_plan': self.daily_plan if self.daily_plan else 'No specific plans'
                        }
                        
                        # Get structured decision about visiting
                        visit_decision = self.generate_structured_location_visit(visit_context)
                        
                        if visit_decision.get('visit_location') == 'YES':
                            # Agent decides to visit - update tracker to show they're at this location
                            if self.shared_tracker:
                                self.shared_tracker.update_agent_position(
                                    self.name,
                                    location_name,
                                    next_coord,
                                    self.current_time
                                )
                            print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Decided to visit {location_name} while traveling to {self.travel_destination}. Energy: {self.energy_level}")
                            
                            # Handle quick stop vs regular visit
                            visit_duration = visit_decision.get('visit_duration', 'QUICK')
                            if visit_duration == 'QUICK':
                                # Record quick stop activity
                                self.record_memory('ACTIVITY_EVENT', {
                                    'type': 'QUICK_STOP',
                                    'location': location_name,
                                    'description': f"Quick stop at {location_name}",
                                    'duration': 'QUICK',
                                    'action_plan': visit_decision.get('action_plan', '')
                                })
                                # Record location event
                                self.record_memory('LOCATION_EVENT', {
                                    'type': 'VISIT',
                                    'location': location_name,
                                    'description': f"Quick stop at {location_name}",
                                    'duration': 'QUICK'
                                })
                                # Update current activity
                                self.current_activity = 'QUICK_STOP'
                            
                            return f"{self.name} decided to visit {location_name} while traveling to {self.travel_destination}", True, {
                                'type': 'location',
                                'location_name': location_name,
                                'location_type': location.type,
                                'visit_duration': visit_decision.get('visit_duration', 'QUICK'),
                                'action_plan': visit_decision.get('action_plan', ''),
                                'reasoning': visit_decision.get('reasoning', ''),
                                'activity_type': 'QUICK_STOP' if visit_duration == 'QUICK' else 'REGULAR_VISIT'
                            }
                        else:
                            # Agent decides to continue travel - ensure tracker shows travel status
                            if self.shared_tracker:
                                self.shared_tracker.update_agent_position(
                                    self.name,
                                    f"Traveling to {self.travel_destination}",
                                    next_coord,
                                    self.current_time
                                )
                            print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Passing by {location_name} while traveling to {self.travel_destination}. Energy: {self.energy_level}")
                            return f"{self.name} passed by {location_name} while traveling to {self.travel_destination}", True, {
                                'type': 'location',
                                'location_name': location_name,
                                'location_type': location.type,
                                'visit_duration': 'NONE',
                                'reasoning': visit_decision.get('reasoning', 'Decided to continue travel')
                            }

            # Check for mid-travel agent encounter
            if self.shared_tracker:
                agents_at_coord = self.shared_tracker.get_agents_at_coordinate(next_coord)
                if agents_at_coord:
                    # Found agents during travel
                    encounter_info = {
                        'type': 'agent',
                        'agent_names': agents_at_coord
                    }
                    
                    # Check if we should interact with any of the encountered agents
                    should_interact = False
                    other_agent = None
                    for agent_name in agents_at_coord:
                        # Find the agent object
                        for agent in self.all_agents_list_for_perception:
                            if agent.name == agent_name:
                                if self.should_interact_with(agent, self.current_time, encounter_info):
                                    should_interact = True
                                    other_agent = agent
                                    break
                        if should_interact:
                            break
                    
                    if should_interact and other_agent:
                        # Update tracker to show interaction
                        if self.shared_tracker:
                            self.shared_tracker.update_agent_position(
                                self.name,
                                f"Interacting with {other_agent.name}",
                                next_coord,
                                self.current_time
                            )
                        print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Encountered {other_agent.name} while traveling. Energy: {self.energy_level}")
                        # Let the simulation handle the conversation
                        return f"{self.name} decided to interact with {other_agent.name} while traveling to {self.travel_destination}", True, {
                            'type': 'agent',
                            'agent_names': [other_agent.name],
                            'should_interact': True,
                            'other_agent': other_agent
                        }
                    else:
                        # Continue travel without interaction
                        if self.shared_tracker:
                            self.shared_tracker.update_agent_position(
                                self.name,
                                f"Traveling to {self.travel_destination}",
                                next_coord,
                                self.current_time
                            )
                        print(f"[Day {self.current_time // 24 + 1} | Hour {self.current_time % 24}] [Agent: {self.name}] [DEBUG] Passing by {', '.join(agents_at_coord)} while traveling to {self.travel_destination}. Energy: {self.energy_level}")
                        return f"{self.name} passed by {', '.join(agents_at_coord)} while traveling to {self.travel_destination}", True, encounter_info

            # Calculate progress
            total_steps = len(self.travel_path) - 1
            current_step = self.travel_step_index
            progress = (current_step / total_steps) * 100

            # Generate travel update message
            travel_message = f"{self.name} is traveling to {self.travel_destination}. Progress: {progress:.1f}%"

            return travel_message, True, None

        except Exception as e:
            print(f"Error in travel step for {self.name}: {str(e)}")
            traceback.print_exc()
            self.clear_invalid_travel_state()
            return None, False, None

# Global shared location tracker for real-time agent position updates during parallel execution
class SharedLocationTracker:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SharedLocationTracker, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self.agent_positions = {}  # agent_name -> (location_name, grid_coordinate, timestamp)
            self.location_occupants = defaultdict(list)  # location_name -> [agent_names]
            self.grid_occupants = defaultdict(list)  # grid_coordinate -> [agent_names]
            self.update_lock = threading.Lock()
            self._initialized = True
    
    def update_agent_position(self, agent_name: str, location_name: str, grid_coordinate: Optional[Tuple[int, int]], timestamp: int):
        """Update an agent's position in real-time"""
        with self.update_lock:
            # Remove agent from old positions
            if agent_name in self.agent_positions:
                old_location, old_coord, _ = self.agent_positions[agent_name]
                if old_location in self.location_occupants:
                    if agent_name in self.location_occupants[old_location]:
                        self.location_occupants[old_location].remove(agent_name)
                if old_coord and old_coord in self.grid_occupants:
                    if agent_name in self.grid_occupants[old_coord]:
                        self.grid_occupants[old_coord].remove(agent_name)
            
            # Add agent to new position
            self.agent_positions[agent_name] = (location_name, grid_coordinate, timestamp)
            if location_name:
                if agent_name not in self.location_occupants[location_name]:
                    self.location_occupants[location_name].append(agent_name)
            if grid_coordinate:
                if agent_name not in self.grid_occupants[grid_coordinate]:
                    self.grid_occupants[grid_coordinate].append(agent_name)
    
    def get_agents_at_location(self, location_name: str) -> List[str]:
        """Get all agents currently at a specific location"""
        with self.update_lock:
            return list(self.location_occupants.get(location_name, []))
    
    def get_agents_at_coordinate(self, grid_coordinate: Tuple[int, int]) -> List[str]:
        """Get all agents currently at a specific grid coordinate"""
        with self.update_lock:
            return list(self.grid_occupants.get(grid_coordinate, []))
    
    def get_agent_position(self, agent_name: str) -> Optional[Tuple[str, Optional[Tuple[int, int]], int]]:
        """Get an agent's current position"""
        with self.update_lock:
            return self.agent_positions.get(agent_name)
    
    def clear(self):
        """Clear all position data"""
        with self.update_lock:
            self.agent_positions.clear()
            self.location_occupants.clear()
            self.grid_occupants.clear()

# Global instance
shared_location_tracker = SharedLocationTracker()

class SharedMemoryBuffer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SharedMemoryBuffer, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the shared memory buffer."""
        self.buffer = []
        self.memory_types = {
            'CONVERSATION_LOG_EVENT': 'conversation_log_event',
            'AGENT_STATE_UPDATE_EVENT': 'agent_state_update_event',
            'PLANNING_EVENT': 'planning_event',
            'ACTIVITY_EVENT': 'activity_event',
            'TRAVEL_EVENT': 'travel_event',
            'INTERACTION_EVENT': 'interaction_event'
        }
        self._write_lock = threading.Lock()

    def push(self, memory_event: Union[Dict[str, Any], MemoryEvent]) -> None:
        """Add a memory event to the buffer."""
        with self._write_lock:
            # Convert MemoryEvent to dict if necessary
            if isinstance(memory_event, MemoryEvent):
                memory_event = memory_event.to_dict()
            
            # Validate memory type
            if memory_event.get('type') not in self.memory_types:
                print(f"Warning: Invalid memory type: {memory_event.get('type')}")
                return

            # Add timestamp if not present
            if 'timestamp' not in memory_event:
                memory_event['timestamp'] = time_module.time()

            # Add version and hash for conflict prevention
            memory_event['version'] = self._get_memory_version(memory_event)
            memory_event['content_hash'] = self._hash_memory_content(memory_event)

            # Check for duplicates
            if not self._is_duplicate_memory(memory_event):
                self.buffer.append(memory_event)

    def flush_to_agents(self, agents: List['Agent']) -> None:
        """Distribute buffered memories to relevant agents."""
        with self._write_lock:
            for event in self.buffer:
                # Handle conversation events
                if event['type'] == 'CONVERSATION_LOG_EVENT':
                    participants = event['data'].get('participants', [])
                    for name in participants:
                        agent = next((a for a in agents if a.name == name), None)
                        if agent:
                            agent.record_memory(event['type'], event['data'])

                # Handle state update events
                elif event['type'] == 'AGENT_STATE_UPDATE_EVENT':
                    agent_name = event['data'].get('agent_name')
                    if agent_name:
                        agent = next((a for a in agents if a.name == agent_name), None)
                        if agent:
                            agent.record_memory(event['type'], event['data'])

                # Handle planning events
                elif event['type'] == 'PLANNING_EVENT':
                    agent_name = event['data'].get('agent_name')
                    if agent_name:
                        agent = next((a for a in agents if a.name == agent_name), None)
                        if agent:
                            agent.record_memory(event['type'], event['data'])

                # Handle activity events
                elif event['type'] == 'ACTIVITY_EVENT':
                    agent_name = event['data'].get('agent_name')
                    if agent_name:
                        agent = next((a for a in agents if a.name == agent_name), None)
                        if agent:
                            agent.record_memory(event['type'], event['data'])

            # Clear the buffer after distribution
            self.buffer.clear()

    def _get_memory_version(self, memory_event: Dict[str, Any]) -> int:
        """Get version number for memory to prevent conflicts."""
        try:
            # Get existing memories of this type
            existing_memories = [m for m in self.buffer if m['type'] == memory_event['type']]
            
            # Start with version 1 if no existing memories
            if not existing_memories:
                return 1
                
            # Increment version from last memory
            last_version = max(m.get('version', 0) for m in existing_memories)
            return last_version + 1
            
        except Exception as e:
            print(f"Error getting memory version: {str(e)}")
            return 1

    def _hash_memory_content(self, data: Dict[str, Any]) -> str:
        """Generate a hash of memory content for duplicate detection."""
        try:
            import hashlib
            import json
            
            # Create a copy of the data to modify
            data_copy = data.copy()
            
            # Create a stable string representation of the data
            content_str = json.dumps({
                'type': data_copy.get('type', ''),
                'time': data_copy.get('time', 0),
                'location': data_copy.get('location', ''),
                'participants': sorted(data_copy.get('participants', [])),
                'content': data_copy.get('content', '')
            }, sort_keys=True)
            
            # Generate hash
            return hashlib.md5(content_str.encode()).hexdigest()
            
        except Exception as e:
            print(f"Error hashing memory content: {str(e)}")
            return ''

    def _is_duplicate_memory(self, memory_event: Dict[str, Any]) -> bool:
        """Check if this memory is a duplicate of an existing one."""
        try:
            content_hash = memory_event.get('content_hash', '')
            if not content_hash:
                return False
                
            # Check for duplicates in buffer
            for memory in self.buffer:
                if memory.get('content_hash') == content_hash:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error checking for duplicate memory: {str(e)}")
            return False

    def get_recent_memories(self, memory_type: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories of a specific type."""
        with self._write_lock:
            if memory_type:
                memories = [m for m in self.buffer if m['type'] == memory_type]
            else:
                memories = self.buffer.copy()
            
            return sorted(memories, key=lambda x: x.get('timestamp', 0), reverse=True)[:limit]

# Create singleton instance
shared_memory_buffer = SharedMemoryBuffer()