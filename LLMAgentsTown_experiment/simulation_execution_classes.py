# Standard library imports
import os
import json
import random
import threading
import time
import traceback
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Union, 
    TYPE_CHECKING, TypeVar, Generic
)

# Third-party imports
import requests

# Local imports
from menu_validator import MenuValidator
from simulation_types import (
    MemoryType, MemoryEvent, ActivityType,
    ACTIVITY_TYPES, MEMORY_TYPES,
    TimeManager, EnergySystem, GrocerySystem,
    MemoryManagerInterface,
    MAX_CONVERSATION_TURNS, CONVERSATION_COOLDOWN_HOURS,
    # Grocery constants
    GROCERY_THRESHOLD_LOW,
    # Meal period function
    get_meal_period
)
from simulation_constants import (
    # Energy constants
    ENERGY_MAX, ENERGY_MIN, ENERGY_DECAY_PER_HOUR,
    ENERGY_COST_WORK_HOUR, ENERGY_COST_PER_STEP,
    ENERGY_GAIN_RESTAURANT_MEAL,
    ENERGY_GAIN_SNACK, ENERGY_GAIN_HOME_MEAL,
    ENERGY_GAIN_NAP, ENERGY_THRESHOLD_LOW
)
from simulation_constants import (
    Result, SimulationError, AgentError, LocationError,
    MemoryError, MetricsError, ThreadSafeBase
)
if TYPE_CHECKING:
    from llm_deepseek_manager import ModelManager
    from memory_manager import MemoryManagerInterface
    from metrics_manager import StabilityMetricsManager
    from shared_trackers import LocationLockManager, SharedLocationTracker
    from prompt_manager import PromptManager
from memory_manager import MemoryManager
from shared_trackers import SharedResourceManager

class SimulationSettings:
    """Centralized simulation settings management."""
    _instance = None
    _settings = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SimulationSettings, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config: Dict[str, Any]) -> None:
        """Initialize settings with configuration."""
        with cls._lock:
            cls._settings = config
            cls._initialized = True

    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """Get all settings."""
        with cls._lock:
            return cls._settings.copy() if cls._settings else {}

    @classmethod
    def get_setting(cls, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        with cls._lock:
            return cls._settings.get(key, default) if cls._settings else default

    @classmethod
    def reset(cls) -> None:
        """Reset settings to initial state."""
        with cls._lock:
            cls._settings = None
            cls._initialized = False

class TownMap:
    """Manages the town's map and location relationships."""
    
    def __init__(self, world_locations_data: Dict[str, List[int]], travel_paths_data: List[List[List[int]]]):
        # Validate input data
        if not world_locations_data:
            raise ValueError("world_locations_data cannot be empty")
        if not travel_paths_data:
            raise ValueError("travel_paths_data cannot be empty")
            
        # Store data with coordinates converted to tuples
        self.locations = {name: tuple(coord) for name, coord in world_locations_data.items()}
        self.paths = [[tuple(coord) for coord in path] for path in travel_paths_data]
        
        # Calculate grid bounds from data
        all_x = [coord[0] for coord in self.locations.values()] + [coord[0] for path in self.paths for coord in path]
        all_y = [coord[1] for coord in self.locations.values()] + [coord[1] for path in self.paths for coord in path]
        
        self.min_x = min(all_x)
        self.max_x = max(all_x)
        self.min_y = min(all_y)
        self.max_y = max(all_y)
        
        # Calculate grid size
        self.grid_size = (self.max_x - self.min_x + 1, self.max_y - self.min_y + 1)
        
        print(f"\nMap bounds: X({self.min_x} to {self.max_x}), Y({self.min_y} to {self.max_y})")
        print(f"Grid size: {self.grid_size}")
        
        # Build path graph and validate paths
        self._build_path_graph()
        
        # Validate all locations are reachable
        self._validate_connectivity()

    def _build_path_graph(self):
        """Build the path graph from travel paths."""
        self.path_graph = defaultdict(set)
        self.valid_coordinates = set()
        
        # First collect all coordinates from paths
        for path in self.paths:
            for coord in path:
                self.valid_coordinates.add(coord)
        
        # Then build the graph
        for path in self.paths:
            for i in range(len(path) - 1):
                coord1 = path[i]
                coord2 = path[i + 1]
                self.path_graph[coord1].add(coord2)
                self.path_graph[coord2].add(coord1)

    def _is_valid_coordinate(self, coord: Tuple[int, int]) -> bool:
        """Check if a coordinate is on a valid path."""
        return coord in self.valid_coordinates

    def _validate_connectivity(self):
        """Validate that all locations are reachable from each other."""
        location_coords = list(self.locations.values())
        unreachable_pairs = []
        
        # First validate that all locations are on valid paths
        invalid_locations = []
        for name, coord in self.locations.items():
            if coord not in self.valid_coordinates:
                invalid_locations.append((name, coord))
        
        if invalid_locations:
            print("\nError: Found locations not on valid paths:")
            for name, coord in invalid_locations:
                print(f"  Location '{name}' at {coord} is not on any path")
            raise ValueError("All locations must be on valid paths")
        
        # Then check connectivity and path lengths
        print("\nChecking connectivity and path lengths between locations:")
        for i, start in enumerate(location_coords):
            for end in location_coords[i+1:]:
                result = self.find_shortest_path(start, end)
                if result:
                    path, length = result
                    print(f"  Path from {start} to {end}: length {length}")
                else:
                    unreachable_pairs.append((start, end))
        
        if unreachable_pairs:
            print("\nWarning: Found unreachable location pairs:")
            for start, end in unreachable_pairs:
                print(f"  No path found between {start} and {end}")

    def find_path(self, start_coord: Tuple[int, int], end_coord: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path between two coordinates using BFS."""
        if not self._is_valid_coordinate(start_coord) or not self._is_valid_coordinate(end_coord):
            return None
            
        # If start and end are the same, return single point
        if start_coord == end_coord:
            return [start_coord]

        # Initialize BFS with distance tracking
        queue = deque([(start_coord, [start_coord], 0)])  # (coord, path, distance)
        visited = {start_coord: 0}  # coord -> distance
        best_path = None
        best_distance = float('inf')
        
        while queue:
            current, path, distance = queue.popleft()
            
            # If we found a path but it's longer than our best so far, skip
            if distance >= best_distance:
                continue
            
            # Check all adjacent coordinates
            for next_coord in self.path_graph[current]:
                if next_coord == end_coord:
                    # Found a path to the end
                    new_path = path + [end_coord]
                    new_distance = distance + 1
                    if new_distance < best_distance:
                        best_path = new_path
                        best_distance = new_distance
                    continue
                
                # Only explore if we haven't seen this coordinate or found a shorter path to it
                if next_coord not in visited or distance + 1 < visited[next_coord]:
                    visited[next_coord] = distance + 1
                    queue.append((next_coord, path + [next_coord], distance + 1))
        
        if best_path:
            print(f"Found path from {start_coord} to {end_coord} with length {best_distance}")
            return best_path
        
        return None

    def get_path_length(self, path: List[Tuple[int, int]]) -> int:
        """Calculate the length of a path in grid units."""
        if not path:
            return 0
        return len(path) - 1  # Number of steps between coordinates

    def find_shortest_path(self, start_coord: Tuple[int, int], end_coord: Tuple[int, int]) -> Optional[Tuple[List[Tuple[int, int]], int]]:
        """Find the shortest path and its length between two coordinates."""
        path = self.find_path(start_coord, end_coord)
        if path:
            return path, self.get_path_length(path)
        return None

    def _is_on_path(self, coord: Tuple[int, int]) -> bool:
        """Check if a coordinate is on any defined path."""
        return coord in self.valid_coordinates

    def get_coordinates_for_location(self, location_name: str) -> Optional[Tuple[int, int]]:
        """Get coordinates for a location by name."""
        return self.locations.get(location_name)

    def get_location_name_at_coord(self, coord: Tuple[int, int]) -> Optional[str]:
        """Get location name at given coordinates."""
        for name, loc_coord in self.locations.items():
            if loc_coord == coord:
                return name
        return None

class Location:
    """Represents a location in the town simulation."""
    
    def __init__(self, name: str, location_type: str, coordinates: Tuple[int, int], 
                 hours: Optional[Dict[str, Any]] = None,
                 prices: Optional[Dict[str, float]] = None, discounts: Optional[Dict[str, Any]] = None):
        """Initialize a location."""
        self.name = name
        self.location_type = location_type
        self.coordinates = coordinates
        self.hours = hours or {'always_open': False, 'open': 7, 'close': 22}
        self.prices = prices or {}
        self.discounts = discounts or {}
        
    def is_open(self, current_time: int) -> bool:
        """Check if the location is open at the given time."""
        try:
            if self.hours.get('always_open', False):
                return True
                
            open_hour = self.hours.get('open', 0)
            close_hour = self.hours.get('close', 24)
            
            # Handle overnight hours (e.g., 22:00 - 06:00)
            if close_hour < open_hour:
                return current_time >= open_hour or current_time < close_hour
            else:
                return open_hour <= current_time < close_hour
                
        except Exception as e:
            print(f"Error checking if {self.name} is open: {str(e)}")
            return False
    
    def get_discounts(self) -> Dict[str, Any]:
        """Get available discounts for this location."""
        return self.discounts.copy()
    
    def __str__(self) -> str:
        return f"Location({self.name}, type={self.location_type}, coords={self.coordinates})"
    
    def __repr__(self) -> str:
        return self.__str__()

# Type hints for circular imports
if TYPE_CHECKING:
    from main_simulation import Location

class Agent:
    """Agent class representing a person in the town simulation."""
    
    def __init__(self, name: str, config: Dict[str, Any], memory_mgr: Optional[MemoryManagerInterface] = None):
        """Initialize an agent with their configuration."""
        self.name = name
        self.memory_mgr = memory_mgr
        
        # Basic information
        self.age = config.get('age')
        self.occupation = config.get('occupation')
        self.residence = config.get('residence')
        self.workplace = config.get('workplace')
        
        # Income information
        income_data = config.get('income_info', {})
        self.income_info = {
            'type': income_data.get('type', 'hourly'),
            'amount': income_data.get('amount', 0.0),
            'schedule': income_data.get('schedule', {
                'type': 'full_time',
                'work_hours': {
                    'start': 9,
                    'end': 17
                },
                'days': [1, 2, 3, 4, 5]
            })
        }
        
        # Relationship information
        relationships = config.get('relationships', {})
        self.relationship_status = relationships.get('status', 'single')
        self.relationships = relationships  # Store full relationship config for relationship detection
        self.household_members = []
        self.spouse = None
        self.best_friends = []
        self.dating = []
        
        # Process household members (people who live together)
        if 'household_members' in relationships:
            for member in relationships['household_members']:
                if isinstance(member, dict):
                    self.household_members.append(member.get('name'))
                    if member.get('relationship_type') == 'spouse':
                        self.spouse = member.get('name')
        
        # Process spouse if directly specified
        if 'spouse' in relationships:
            spouse_data = relationships['spouse']
            if isinstance(spouse_data, dict):
                self.spouse = spouse_data.get('name')
                if self.spouse and self.spouse not in self.household_members:
                    self.household_members.append(self.spouse)
        
        # Process best friends (people who don't live together)
        if 'best_friends' in relationships:
            for friend in relationships['best_friends']:
                if isinstance(friend, dict):
                    self.best_friends.append(friend.get('name'))
        
        # Process dating relationships (people who don't live together)
        if 'dating' in relationships:
            for partner in relationships['dating']:
                if isinstance(partner, dict):
                    self.dating.append(partner.get('name'))
        
        # Initialize tracking variables
        self.daily_income = 0.0
        self.daily_expenses = 0.0

        # Calculate initial money based on 3 days of wages from config
        daily_wage = 0.0
        income_type = self.income_info.get('type', 'hourly')
        income_amount = self.income_info.get('amount', 0.0)
        
        if income_type == 'hourly':
            schedule = self.income_info.get('schedule', {})
            work_hours = schedule.get('work_hours', {'start': 9, 'end': 17})
            hours_per_day = work_hours.get('end', 17) - work_hours.get('start', 9)
            daily_wage = income_amount * hours_per_day
        elif income_type == 'daily':
            daily_wage = income_amount
        elif income_type == 'weekly':
            daily_wage = income_amount / 5.0
        elif income_type == 'monthly':
            daily_wage = income_amount / 20.0
        elif income_type == 'salary':
            daily_wage = income_amount / 365.0
        
        self.money = daily_wage * 3

        self.current_activity = 'idle'
        self.current_location = None
        self.daily_plan = None
        self.plan_processed = False
        self.last_income_update = 0  # Initialize last_income_update
        
        # Initialize systems
        self.energy_system = EnergySystem()
        self.grocery_system = GrocerySystem()
            
        # Initialize relationships
        self.relationships = {}
        self.conversation_history = defaultdict(list)
        
        # Initialize location tracking
        self.location_history = []
        self.visit_counts = defaultdict(int)
        
        # Initialize financial tracking
        self.purchase_history = []
        self.earning_history = []
        
        # Initialize activity tracking
        self.activity_history = []
        self.current_activity_start_time = None
        
        # Initialize plan tracking
        self.plan_history = []
        self.current_plan = None
        self.plan_success_rate = 0.0
        
        # Initialize interrupted travel state for resumption after conversations
        self.interrupted_travel = None  # Stores {target_location, original_activity, reason} when travel is interrupted
        
        # Initialize state tracking
        self.state_history = []
        self.last_state_update = None
        
        # Initialize metrics tracking
        self.metrics = {
            'energy_levels': [],
            'grocery_levels': [],
            'financial_states': [],
            'activities': [],
            'location_visits': []
        }

    def get_current_location_name(self) -> str:
        """Get the current location name."""
        if self.current_location:
            return self.current_location.name
        return "Unknown"
        
    def update_state(self, state_update: Dict[str, Any]) -> None:
        """Update agent's state."""
        if 'energy_level' in state_update:
            self.energy_system.set_energy(self.name, state_update['energy_level'])
        if 'grocery_level' in state_update:
            self.grocery_system.set_grocery_level(self.name, state_update['grocery_level'])
        if 'money' in state_update:
            self.money = state_update['money']
        if 'current_activity' in state_update:
            self.current_activity = state_update['current_activity']
            self._update_activity_flags(state_update['current_activity'])
        if 'current_location' in state_update:
            self.current_location = state_update['current_location']

    def _update_activity_flags(self, activity: str) -> None:
        """Update activity-related flags."""
        self.current_activity = activity

    def can_afford_purchase(self, amount: float) -> bool:
        """Check if agent can afford a purchase."""
        return self.money >= amount

    def record_purchase(self, amount: float, location: str, items: List[Dict[str, Any]], has_discount: bool = False) -> Optional[Dict[str, Any]]:
        """Record a purchase and update money. Assumes affordability has been checked."""
        try:
            # Update money
            self.money -= amount
            self.daily_expenses += amount
            
            # Create purchase record
            purchase_record = {
                'amount': amount,
                'location': location,
                'items': items,
                'has_discount': has_discount,
                'remaining_money': self.money,
                'day': TimeManager.get_current_day(),
                'time': TimeManager.get_current_hour()
            }
            
            # Record purchase in memory as a state update
            self.record_memory(
                memory_type=MemoryType.STATE_UPDATE.value,
                content={
                    'type': 'purchase',
                    'purchase_record': purchase_record
                }
            )
            
            return purchase_record
            
        except Exception as e:
            print(f"Error recording purchase for {self.name}: {str(e)}")
            traceback.print_exc()
            return None

    def get_financial_status(self) -> Dict[str, Any]:
        """Get current financial status."""
        return {
            'money': self.money,
            'daily_income': self.daily_income,
            'daily_expenses': self.daily_expenses,
            'income_type': self.income_info.get('type', 'hourly'),
            'income_amount': self.income_info.get('amount', 0.0),
            'last_income_update': self.last_income_update
        }

    def store_plan(self, plan: Dict[str, Any], current_time: int, current_day: int) -> None:
        """Store a plan for the agent."""
        self.daily_plan = plan
        self.plan_history.append({
            'plan': plan,
            'time': current_time,
            'day': current_day
        })

    def update_plan_slice(self, new_activities: List[Dict[str, Any]], start_hour: int):
        """
        Updates specific hours in the agent's daily plan without destroying non-conflicting activities.
        Only replaces activities that actually conflict with the new commitments.
        """
        if not self.daily_plan or 'activities' not in self.daily_plan:
            # If there's no plan, create one with the new activities
            self.daily_plan = {'activities': new_activities}
            return

        original_activities = self.daily_plan.get('activities', [])
        
        # Get the specific hours that will be replaced by new activities
        commitment_hours = {act.get('time') for act in new_activities}
        
        # Keep all original activities that DON'T conflict with commitment hours
        preserved_activities = [act for act in original_activities if act.get('time') not in commitment_hours]
        
        # Combine preserved activities with new commitments and sort by time
        self.daily_plan['activities'] = sorted(preserved_activities + new_activities, key=lambda x: x.get('time', 0))
        
        # Record that the plan was updated with specific hours
        self.memory_mgr.add_memory(
            agent_name=self.name,
            memory_data={
                'reason': f'Plan updated with commitments for hours: {sorted(commitment_hours)}',
                'new_activities': new_activities,
                'preserved_activities_count': len(preserved_activities)
            },
            memory_type=MemoryType.PLAN_UPDATE.value
        )

    def record_memory(self, memory_type: str, content: Dict[str, Any], timestamp: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> Result[None]:
        """Record a memory directly using the memory manager."""
        try:
            self.memory_mgr.add_memory(
                agent_name=self.name,
                memory_data=content,
                memory_type=memory_type,
                timestamp=timestamp
            )
            return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error recording memory: {str(e)}")

    def get_recent_messages_from(self, other_agent_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation messages with a specific agent."""
        try:
            if not self.memory_mgr:
                return []

            # Get recent conversation memories
            conversation_memories = self.memory_mgr.get_recent_memories(
                agent_name=self.name,
                memory_type=MemoryType.CONVERSATION.value,
                limit=limit * 2  # Get more to filter, since not all will involve the target agent
            )
            
            # Filter for conversations that include the target agent
            relevant_conversations = []
            for memory in conversation_memories:
                content = memory.get('content', {})
                participants = content.get('participants', [])
                
                # Check if the other agent was a participant in this conversation
                if other_agent_name in participants:
                    # Transform the memory structure to what PromptManager expects
                    transformed_message = {
                        'content': content.get('dialogue', ''),
                        'participants': participants,
                        'location': content.get('location', 'unknown'),
                        'turn_number': content.get('turn_number', 0),
                        'day': memory.get('day', 0),
                        'hour': memory.get('hour', 0)
                    }
                    relevant_conversations.append(transformed_message)
            
            # Return the most recent conversations, limited to the requested number
            return relevant_conversations[:limit]
                
        except Exception as e:
            print(f"Error getting recent messages from {other_agent_name}: {str(e)}")
            traceback.print_exc()
            return []

    def can_perform_action(self, energy_cost: float) -> bool:
        """Check if agent has enough energy to perform an action."""
        result = self.energy_system.can_perform_action(self.name, int(energy_cost))
        return result.success and result.value

    def clear_interrupted_travel(self):
        """Clear interrupted travel state when no longer needed."""
        if hasattr(self, 'interrupted_travel'):
            self.interrupted_travel = None

class PlanExecutor:
    """Executes plans for agents in the simulation."""
    
    def __init__(self, location_lock_mgr: 'LocationLockManager', conversation_mgr: 'ConversationManager', memory_mgr: MemoryManagerInterface, metrics_mgr: 'StabilityMetricsManager', location_tracker: 'SharedLocationTracker', agents: Dict[str, 'Agent'], locations: Dict[str, 'Location'], prompt_mgr: 'PromptManager', model_mgr: 'ModelManager', config_data: Dict[str, Any]):
        """Initialize the PlanExecutor."""
        self.location_lock_mgr = location_lock_mgr
        self.conversation_mgr = conversation_mgr
        self.memory_mgr = memory_mgr
        self.metrics_mgr = metrics_mgr
        self.location_tracker = location_tracker
        self.agents = agents
        self.locations = locations
        self.prompt_mgr = prompt_mgr
        self.model_mgr = model_mgr
        self.config_data = config_data
        self.menu_validator = MenuValidator(config_data)  # Add menu validation
        self._active_agent_activities = {}
        self.agent_plans = {}
        self.agent_activities = {}
        self._lock = threading.Lock()

    def get_agent_activity(self, agent_name: str) -> str:
        """Get the current activity of an agent."""
        return self.agent_activities.get(agent_name, 'idle')

    def update_agent_activity(self, agent_name: str, activity: str) -> None:
        """Update the current activity of an agent."""
        with self._lock:
            self.agent_activities[agent_name] = activity

    def _handle_work(self, agent: 'Agent', location: 'Location', plan: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """Handle work at a location."""
        try:
            # Extract work description from plan if provided
            work_description = plan.get('description', 'Working') if plan else 'Working'

            # Check if location matches agent's workplace
            if location.name != agent.workplace:
                return False, {'error': 'Location is not agent\'s workplace'}
            
            # Check if workplace is open
            if not location.is_open(TimeManager.get_current_hour()):
                return False, {'error': 'Workplace is closed'}
            
            # Calculate energy cost for working using centralized system
            result = agent.energy_system.calculate_energy_cost('work', 1)
            energy_cost = result.value if result.success else ENERGY_COST_WORK_HOUR
            if not agent.can_perform_action(energy_cost):
                return False, {'error': 'Not enough energy to work'}
            
            # Get energy before work for detailed logging
            energy_before_work = agent.energy_system.get_energy(agent.name)
            
            # Update energy
            result = agent.energy_system.update_energy(agent.name, -energy_cost)
            if not result.success:
                print(f"Failed to update energy for {agent.name}: {result.error}")
                return False, {'error': f'Failed to update energy: {result.error}'}
            
            # Log work energy cost
            energy_after_work = agent.energy_system.get_energy(agent.name)
            print(f"[ENERGY] {agent.name} work cost: {energy_before_work} → {energy_after_work} (work -{energy_cost})")
            
            # Calculate money earned based on income type using the same logic as Agent initialization
            money_earned = 0.0
            income_type = agent.income_info.get('type', 'hourly')
            income_amount = agent.income_info.get('amount', 0.0)
            
            if income_type == 'hourly':
                # For hourly workers, pay the hourly rate directly
                money_earned = income_amount
            elif income_type == 'monthly':
                # For monthly workers, calculate hourly rate based on work schedule
                schedule = agent.income_info.get('schedule', {})
                work_hours = schedule.get('work_hours', {'start': 9, 'end': 17})
                hours_per_day = work_hours.get('end', 17) - work_hours.get('start', 9)
                # Calculate hourly rate: monthly salary / (20 work days * hours per day)
                hourly_rate = income_amount / (20.0 * hours_per_day)
                money_earned = hourly_rate
            else:
                # For any other income types, default to 0 (shouldn't happen with current config)
                print(f"[WARNING] {agent.name} has unsupported income type: {income_type}")
                money_earned = 0.0
            
            # Update agent's money and daily income tracking
            agent.money += money_earned
            agent.daily_income += money_earned
            
            # Log the wage calculation for debugging
            print(f"[WAGE] {agent.name} earned ${money_earned:.2f} (income_type: {income_type}, amount: {income_amount})")
            
            # Record work in memory
            work_data = {
                'type': 'work',
                'location': location.name,
                'description': work_description,
                'energy_cost': energy_cost,
                'money_earned': money_earned,
                'timestamp': {
                    'day': TimeManager.get_current_day(),
                    'hour': TimeManager.get_current_hour()
                }
            }
            
            # Add memory using MemoryType.WORK
            self.memory_mgr.add_memory(
                agent_name=agent.name, 
                memory_data=work_data, 
                memory_type=MemoryType.WORK.value
            )
            
            # Update agent's activity
            agent.current_activity = 'working'
            
            return True, work_data
            
        except Exception as e:
            print(f"Error handling work for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False, {'error': str(e)}

    def execute(self, agent: 'Agent', plan: Dict[str, Any], current_time: int) -> Tuple[bool, Dict[str, Any]]:
        """Execute a plan for an agent.
        
        ENERGY THRESHOLD SYSTEM:
        - If agent energy <= ENERGY_THRESHOLD_LOW, trigger emergency replan for food
        - If agent energy <= 0, force agent to residence for recovery
        - Emergency replans take priority over all other activities
        - This prevents agents from getting stuck when they can't travel due to low energy
        - During sleep hours (23-6), normal sleep energy recovery applies
        """
        try:
            # ZERO: ENERGY THRESHOLD CHECKS - Check for critical energy levels before any execution
            current_energy = agent.energy_system.get_energy(agent.name)
            
            # FIRST: Handle automatic sleep hours (23-6) when agent is at residence
            # Sleep takes priority over energy checks during sleep hours
            if self._is_sleep_hour(current_time):
                print(f"[SLEEP] {agent.name} entering automatic sleep system at hour {current_time}")
                return self._handle_automatic_sleep(agent, current_time)
            
            # Check for critical energy failure (0 or below) - only outside sleep hours
            if current_energy <= 0:
                print(f"[CRITICAL] {agent.name} has {current_energy} energy - forcing return to residence for recovery")
                return self._handle_critical_energy_failure(agent, current_time)
            
            # Check for low energy threshold - trigger emergency replan
            if current_energy <= ENERGY_THRESHOLD_LOW:
                print(f"[EMERGENCY] {agent.name} has {current_energy} energy (≤{ENERGY_THRESHOLD_LOW}) - triggering emergency food replan")
                return False, {
                    'error': f'Critical energy level ({current_energy}/{ENERGY_MAX}). Immediate food needed.',
                    'replan_needed': True,
                    'reason_code': 'CRITICAL_ENERGY_LOW',
                    'priority': 'emergency'
                }
            
            # If no plan is provided and it's not sleep time, return success (agent is idle)
            if not plan:
                return True, {'status': 'No plan provided, agent is idle'}
            
            # SECOND: Check if agent has interrupted travel to resume
            if hasattr(agent, 'interrupted_travel') and agent.interrupted_travel:
                print(f"[DEBUG {agent.name}] Resuming interrupted travel to {agent.interrupted_travel['target_location']}")
                
                target_location_name = agent.interrupted_travel['target_location']
                original_activity = agent.interrupted_travel['original_activity']
                
                # Check if agent has reached the target already
                current_position = self.location_tracker.get_agent_position(agent.name)
                current_location_name = current_position[0] if current_position else agent.residence
                
                if current_location_name == target_location_name:
                    # Agent is already at target, proceed with original action
                    print(f"[DEBUG {agent.name}] Already at target {target_location_name}, executing original action")
                    agent.interrupted_travel = None  # Clear interrupted state
                    success, result = self._handle_action_at_location(agent, original_activity, current_time)
                    if success:
                        self.update_agent_activity(agent.name, original_activity['action'])
                        self._check_for_social_opportunity_at_location(agent, self.locations[target_location_name])
                    return success, result
                else:
                    # Continue travel to target
                    path = self._find_path_to_target(agent, target_location_name)
                    if not path:
                        agent.interrupted_travel = None  # Clear invalid interrupted state
                        return False, {'error': f"No valid path found to resume travel to {target_location_name}"}
                    
                    # Resume travel
                    if not self._execute_travel(agent, path, original_activity, current_time):
                        # Travel interrupted again (conversation), keep interrupted state
                        print(f"[DEBUG {agent.name}] Travel interrupted again during resumption")
                        return True, {'status': "Travel resumption interrupted for conversation."}
                    else:
                        # Travel completed, execute original action
                        agent.interrupted_travel = None  # Clear interrupted state
                        success, result = self._handle_action_at_location(agent, original_activity, current_time)
                        if success:
                            self.update_agent_activity(agent.name, original_activity['action'])
                            self._check_for_social_opportunity_at_location(agent, self.locations[target_location_name])
                        return success, result
            
            # THIRD: Normal plan execution if no interrupted travel
            if not plan or 'activities' not in plan:
                return False, {'error': "No valid plan found"}
            
            # DEBUG: Log all agents' plan contents to understand the issue
            print(f"[DEBUG {agent.name}] Hour {current_time} - Plan has {len(plan.get('activities', []))} activities")
            
            # Find activity for current hour
            current_activity = next((act for act in plan['activities'] if act.get('time') == current_time), None)
            if not current_activity:
                # DEBUG: Show what activities exist around this time
                print(f"[DEBUG {agent.name}] No activity found for hour {current_time}!")
                nearby_activities = [act for act in plan.get('activities', []) if abs(act.get('time', -1) - current_time) <= 2]
                print(f"[DEBUG {agent.name}] Nearby activities (±2 hours): {nearby_activities}")
                print(f"[DEBUG {agent.name}] All activity times: {[act.get('time') for act in plan.get('activities', [])]}")
                return True, {'status': f"No activity scheduled for hour {current_time}"} # Not a failure
                
            print(f"[DEBUG {agent.name}] Found activity for hour {current_time}: {current_activity}")
            
            target_location_name = current_activity.get('target')
            if not target_location_name:
                return False, {'error': "No target location specified in activity"}
                
            target_location = self.locations.get(target_location_name)
            if not target_location:
                return False, {'error': f"Target location {target_location_name} not found"}
                
            # --- Travel Logic ---
            current_position = self.location_tracker.get_agent_position(agent.name)
            current_location_name = current_position[0] if current_position else agent.residence
            
            if current_location_name != target_location_name:
                print(f"{agent.name} needs to travel from {current_location_name} to {target_location_name}")
                path = self._find_path_to_target(agent, target_location_name)
                if not path:
                    return False, {'error': f"No valid path found to {target_location_name}"}
                    
                # If travel is interrupted for a conversation, it returns False.
                # This is a successful use of the time slot, so we return True.
                if not self._execute_travel(agent, path, current_activity, current_time):
                    # Travel was interrupted - the interrupted state should already be stored in _execute_travel
                    # Just ensure we have the state stored correctly
                    if not hasattr(agent, 'interrupted_travel') or not agent.interrupted_travel:
                        # Fallback: store interrupted state if it wasn't stored in _execute_travel
                        agent.interrupted_travel = {
                            'target_location': target_location_name,
                            'original_activity': current_activity,
                            'reason': 'conversation_interruption_fallback'
                        }
                    return True, {'status': "Travel interrupted for conversation."}
            
            # --- Action Logic (Now correctly runs after travel logic) ---
            success, result = self._handle_action_at_location(agent, current_activity, current_time)
            
            if success:
                print(f"{agent.name} successfully completed activity '{current_activity['action']}' at {target_location_name}")
                self.update_agent_activity(agent.name, current_activity['action'])
                self._check_for_social_opportunity_at_location(agent, target_location)
            else:
                print(f"Failed to complete activity at {target_location_name}: {result.get('error', 'Unknown error')}")

            return success, result
                        
        except Exception as e:
            print(f"Error executing plan for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False, {'error': str(e)}

    def _is_sleep_hour(self, current_time: int) -> bool:
        """Check if current time is during sleep hours (23-6)."""
        return current_time == 23 or (0 <= current_time <= 6)

    def _handle_critical_energy_failure(self, agent: 'Agent', current_time: int) -> Tuple[bool, Dict[str, Any]]:
        """Handle critical energy failure by forcing agent to residence for recovery."""
        try:
            print(f"[CRITICAL] Handling energy failure for {agent.name} at hour {current_time}")
            
            # Get current location
            current_position = self.location_tracker.get_agent_position(agent.name)
            current_location_name = current_position[0] if current_position else None
            
            # If no current position is tracked, assume agent is at residence
            if not current_location_name:
                current_location_name = agent.residence
            
            # If agent is already at residence, just rest there
            if current_location_name == agent.residence:
                print(f"[CRITICAL] {agent.name} already at residence - forcing rest")
                
                # Clear any interrupted travel state
                if hasattr(agent, 'interrupted_travel'):
                    agent.interrupted_travel = None
                
                # Only clear daily plan if agent has no valid emergency replan
                # Check if the current plan has food-related activities that could restore energy
                has_food_plan = False
                if agent.daily_plan and 'activities' in agent.daily_plan:
                    for activity in agent.daily_plan['activities']:
                        if (activity.get('action') == 'eat' and 
                            activity.get('time', -1) >= current_time):
                            has_food_plan = True
                            break
                
                if not has_food_plan:
                    # Clear any daily plan to prevent further execution attempts
                    agent.daily_plan = None
                    print(f"[CRITICAL] {agent.name} has no valid food plan - cleared daily plan")
                else:
                    print(f"[CRITICAL] {agent.name} has valid food plan - keeping plan for emergency execution")
                
                # Give minimal energy recovery only outside sleep hours to prevent complete stuckness
                # During sleep hours, let the sleep system handle energy recovery
                if not self._is_sleep_hour(current_time):
                    minimal_recovery = ENERGY_GAIN_NAP  # Small energy boost for resting at home
                    result = agent.energy_system.update_energy(agent.name, minimal_recovery)
                    if result.success:
                        print(f"[CRITICAL] {agent.name} gained {minimal_recovery} energy from resting at home (now at {agent.energy_system.get_energy(agent.name)})")
                else:
                    print(f"[CRITICAL] {agent.name} at residence during sleep hours - sleep system will handle energy recovery")
                
                # Record the critical energy failure
                self.memory_mgr.add_memory(
                    agent_name=agent.name,
                    memory_data={
                        'type': 'critical_energy_failure',
                        'reason': 'Energy depleted to 0, forced rest at residence',
                        'location': agent.residence,
                        'timestamp': {
                            'day': TimeManager.get_current_day(),
                            'hour': current_time
                        }
                    },
                    memory_type=MemoryType.FAILED_ACTION.value
                )
                
                agent.current_activity = 'forced_rest'
                return True, {
                    'status': f'Critical energy failure - {agent.name} resting at residence until sleep time',
                    'action': 'forced_rest',
                    'location': agent.residence,
                    'reason': 'energy_depleted'
                }
            
            # Agent is not at residence - force travel to residence
            print(f"[CRITICAL] {agent.name} at {current_location_name}, forcing travel to {agent.residence}")
            
            # Clear any interrupted travel state
            if hasattr(agent, 'interrupted_travel'):
                agent.interrupted_travel = None
            
            # Only clear daily plan if agent has no valid emergency replan
            # Check if the current plan has food-related activities that could restore energy
            has_food_plan = False
            if agent.daily_plan and 'activities' in agent.daily_plan:
                for activity in agent.daily_plan['activities']:
                    if (activity.get('action') == 'eat' and 
                        activity.get('time', -1) >= current_time):
                        has_food_plan = True
                        break
            
            if not has_food_plan:
                # Clear any daily plan to prevent further execution attempts
                agent.daily_plan = None
                print(f"[CRITICAL] {agent.name} has no valid food plan - cleared daily plan")
            else:
                print(f"[CRITICAL] {agent.name} has valid food plan - keeping plan for emergency execution")
            
            # Force travel to residence (ignore energy costs for critical failure)
            # Get residence coordinates from the agent's town map
            residence_coords = None
            if hasattr(agent, 'town_map') and agent.town_map:
                residence_coords = agent.town_map.get_coordinates_for_location(agent.residence)
            
            if not residence_coords:
                # Fallback: try to get coordinates from the location tracker or config
                print(f"[WARNING] Could not find coordinates for {agent.residence} from town map, using fallback")
                # For now, just update the location tracker with the residence name
                # The coordinates will be handled by the location tracker
                residence_coords = (0, 0)  # Default coordinates
            
            # Update location tracker to residence immediately
            self.location_tracker.update_agent_position(
                agent.name, 
                agent.residence, 
                residence_coords, 
                TimeManager.get_current_hour()
            )
            
            # Also update agent's current location for consistency
            agent.current_location = agent.residence
            
            # Record the critical energy failure and forced travel
            self.memory_mgr.add_memory(
                agent_name=agent.name,
                memory_data={
                    'type': 'critical_energy_failure',
                    'reason': 'Energy depleted to 0, forced travel to residence',
                    'from_location': current_location_name,
                    'to_location': agent.residence,
                    'timestamp': {
                        'day': TimeManager.get_current_day(),
                        'hour': current_time
                    }
                },
                memory_type=MemoryType.FAILED_ACTION.value
            )
            
            agent.current_activity = 'forced_travel'
            return True, {
                'status': f'Critical energy failure - {agent.name} forced to {agent.residence} for recovery',
                'action': 'forced_travel',
                'location': agent.residence,
                'reason': 'energy_depleted'
            }
            
        except Exception as e:
            print(f"Error handling critical energy failure for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False, {'error': str(e)}

    def _handle_automatic_sleep(self, agent: 'Agent', current_time: int) -> Tuple[bool, Dict[str, Any]]:
        """Handle automatic sleep during hours 23-6 when agent is at residence."""
        try:
            # Get agent's current location
            current_position = self.location_tracker.get_agent_position(agent.name)
            current_location_name = current_position[0] if current_position else agent.residence
            
            # Check if agent is at their residence
            if current_location_name != agent.residence:
                # Agent is not at residence during sleep hours - force them home
                print(f"[SLEEP FORCE] {agent.name} is not at residence ({agent.residence}) during sleep hour {current_time}. Currently at: {current_location_name}. Forcing travel home.")

                # Force agent to go home immediately for sleep
                self.location_tracker.update_agent_position(agent.name, agent.residence, 'forced_sleep_travel')
                print(f"[SLEEP FORCE] {agent.name} forced to travel home to {agent.residence} for sleep")

                # Update current location for sleep processing
                current_location_name = agent.residence
            
            # Agent is at residence - handle automatic sleep
            # During all sleep hours (23-6), set energy to 100 to compensate for decay and ensure max energy

            if current_time == 23:
                # Starting sleep at 23:00
                sleep_type = 'begin_sleep'
                description = f"Beginning sleep at {agent.residence} to rest and reset energy for tomorrow."
            else:
                # Continuing sleep during hours 0-6
                sleep_type = 'continue_sleep'
                description = f"Continuing to sleep peacefully at {agent.residence} and recovering energy."

            # Set energy to maximum during sleep (compensates for natural decay + ensures full recovery)
            result = agent.energy_system.set_energy(agent.name, ENERGY_MAX)
            if not result.success:
                print(f"[WARNING] Failed to set energy for {agent.name} during sleep: {result.error}")
            else:
                current_energy_after = agent.energy_system.get_energy(agent.name)
                print(f"[SLEEP] {agent.name} energy set to {ENERGY_MAX} during sleep (now at {current_energy_after})")
            
            # Record sleep activity in memory
            sleep_data = {
                'type': 'automatic_sleep',
                'location': agent.residence,
                'sleep_type': sleep_type,
                'energy_set_to': ENERGY_MAX,
                'timestamp': {
                    'day': TimeManager.get_current_day(),
                    'hour': current_time
                }
            }
            
            # Add memory using MemoryType.REST
            self.memory_mgr.add_memory(
                agent_name=agent.name, 
                memory_data=sleep_data, 
                memory_type=MemoryType.REST.value
            )
            
            # Update agent's activity
            agent.current_activity = 'sleeping'
            
            return True, {
                'status': description,
                'action': 'sleep',
                'location': agent.residence,
                'sleep_type': sleep_type,
                'energy_set_to': ENERGY_MAX
            }
            
        except Exception as e:
            print(f"Error handling automatic sleep for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False, {'error': str(e)}

    def _find_path_to_target(self, agent: 'Agent', target_location: str) -> Optional[List[Tuple[int, int]]]:
        """Find a path to the target location."""
        try:
            # Get current position
            current_position = self.location_tracker.get_agent_position(agent.name)
            if not current_position:
                print(f"No current position found for {agent.name}")
                return None
            
            current_location, current_coord, _ = current_position
            
            # Get target coordinates from agent's town map
            target_coord = None
            if hasattr(agent, 'town_map') and agent.town_map:
                target_coord = agent.town_map.get_coordinates_for_location(target_location)
            
            if not target_coord:
                print(f"No coordinates found for {target_location}")
                return None
            
            # Find path
            path = None
            if hasattr(agent, 'town_map') and agent.town_map:
                path = agent.town_map.find_path(current_coord, target_coord)
            
            return path
            
        except Exception as e:
            print(f"Error finding path for {agent.name}: {str(e)}")
            return None

    def _execute_travel(self, agent: 'Agent', path: List[Tuple[int, int]], current_activity: dict, current_time: int) -> bool:
        """Execute travel along a path."""
        try:
            if not path:
                return False
            
            # CRITICAL: Check for zero energy before attempting travel
            current_energy = agent.energy_system.get_energy(agent.name)
            if current_energy <= 0:
                print(f"[CRITICAL] {agent.name} has {current_energy} energy - cannot travel, forcing to residence")
                return False  # This will trigger the critical energy failure handler
            
            # Calculate energy cost for travel
            travel_cost_result = agent.energy_system.calculate_energy_cost('travel', len(path))
            if not travel_cost_result.success:
                print(f"Failed to calculate travel cost: {travel_cost_result.error}")
                return False
            
            energy_cost = travel_cost_result.value
            
            # Check if agent has enough energy
            if not agent.can_perform_action(energy_cost):
                print(f"{agent.name} doesn't have enough energy for travel ({energy_cost} needed)")
                return False
            
            # Get current location
            current_position = self.location_tracker.get_agent_position(agent.name)
            if not current_position:
                print(f"No current position found for {agent.name}")
                return False
                
            current_location, _, _ = current_position
            
            # Execute travel step by step
            steps_taken = 0
            last_named_location = current_location
            
            for step in path:
                # Get energy before travel step for detailed logging
                energy_before_step = agent.energy_system.get_energy(agent.name)
                
                # Update energy (cost per step)
                step_energy_cost = energy_cost / len(path)
                if not agent.can_perform_action(step_energy_cost):
                    print(f"{agent.name} ran out of energy during travel")
                    return False
                result = agent.energy_system.update_energy(agent.name, -step_energy_cost)
                if not result.success:
                    print(f"Failed to update energy for {agent.name}: {result.error}")
                    return False
                
                # Log travel energy cost
                energy_after_step = agent.energy_system.get_energy(agent.name)
                print(f"[ENERGY] {agent.name} travel step {steps_taken + 1}/{len(path)}: {energy_before_step} → {energy_after_step} (travel -{step_energy_cost:.1f})")
                
                steps_taken += 1
                
                # Update tracker with current coordinate for emergent encounters
                current_location_name = last_named_location
                if hasattr(agent, 'town_map') and agent.town_map:
                    location_name = agent.town_map.get_location_name_at_coord(step)
                    if location_name:
                        current_location_name = location_name
                self.location_tracker.update_agent_position(agent.name, current_location_name, step, TimeManager.get_current_hour())

                # Check for social opportunities on the path
                if self._check_for_social_opportunity_on_path(agent, step, current_location_name):
                    # If a conversation starts, the agent pauses travel for this hour.
                    print(f"{agent.name} paused travel to have a conversation.")
                    
                    # Store interrupted travel state for resumption
                    # We need to get the target location from the current activity
                    target_location_name = current_activity.get('target') if current_activity else None
                    if target_location_name:
                        agent.interrupted_travel = {
                            'target_location': target_location_name,
                            'original_activity': current_activity,
                            'reason': 'conversation_interruption',
                            'interrupted_at_step': steps_taken,
                            'total_steps': len(path)
                        }
                        print(f"[DEBUG {agent.name}] Stored interrupted travel state: target={target_location_name}, step={steps_taken}/{len(path)}")
                    
                    return False # Travel did not complete

                # If we've reached a named location, update state
                if current_location_name != last_named_location:
                    print(f"{agent.name} reached {current_location_name} after {steps_taken} steps")
                    last_named_location = current_location_name
                    
            # Record travel in memory
            final_pos = self.location_tracker.get_agent_position(agent.name)
            final_location_name = final_pos[0] if final_pos else last_named_location

            self.memory_mgr.add_memory(
                agent_name=agent.name,
                memory_data={
                    'type': 'travel',
                    'from': current_location,
                    'to': final_location_name,
                    'steps': steps_taken,
                    'energy_cost': energy_cost,
                    'timestamp': TimeManager.get_current_hour()
                },
                memory_type=MemoryType.TRAVEL.value
            )
            
            return True
            
        except Exception as e:
            print(f"Error during travel: {str(e)}")
            return False

    def _get_nearby_agents_at_location(self, agent: 'Agent', location_name: str) -> Set[str]:
        """Get names of agents at a named location."""
        nearby_agents = set()
        for other_agent_name in self.location_tracker.get_agents_at_location(location_name):
            if other_agent_name != agent.name:
                nearby_agents.add(other_agent_name)
        return nearby_agents

    def _get_nearby_agents_on_path(self, agent: 'Agent', coord: Tuple[int, int]) -> Set[str]:
        """Get names of agents at a specific coordinate."""
        nearby_agents = set()
        for other_agent_name in self.location_tracker.get_agents_at_coordinate(coord):
            if other_agent_name != agent.name:
                nearby_agents.add(other_agent_name)
        return nearby_agents

    def _find_and_initiate_conversation(self, agent: 'Agent', potential_participant_names: Set[str], context: Dict[str, Any]) -> bool:
        """Shared logic to find an available agent and start a conversation."""
        for participant_name in potential_participant_names:
            participant = self.agents.get(participant_name)
            if not participant:
                continue

            # Perform "Ready Check" on the participant.
            participant_activity = self.get_agent_activity(participant.name)
            if participant_activity in ['resting', 'working', 'socializing']:
                continue  # This person is busy, check the next one.

            # Perform "Cooldown Check".
            if not self.conversation_mgr.can_converse(agent.name, participant.name):
                continue  # They've talked too recently, check the next one.

            # Found a valid partner. Start the conversation and return True.
            print(f"[INFO] Emergent social opportunity: {agent.name} is starting a conversation with {participant.name} at {context.get('location', 'an unknown place')}.")
            self.conversation_mgr.handle_conversation(
                initiator=agent,
                participants=[participant],
                memory_mgr=self.memory_mgr,
                plan_executor=self,
                prompt_mgr=self.prompt_mgr,
                model_mgr=self.model_mgr,
                context=context
            )
            return True # Conversation started
        return False # No conversation was started

    def _check_for_social_opportunity_at_location(self, agent: 'Agent', location: 'Location'):
        """After an action, check for nearby agents at a named location."""
        try:
            if agent.current_activity in ['resting', 'working', 'socializing']:
                print(f"[DEBUG] Social check skipped for {agent.name} - busy with {agent.current_activity}")
                return

            nearby_agents_names = self._get_nearby_agents_at_location(agent, location.name)
            if not nearby_agents_names:
                print(f"[DEBUG] No nearby agents found for {agent.name} at {location.name}")
                return

            print(f"[DEBUG] Social opportunity check for {agent.name} at {location.name} - found nearby agents: {nearby_agents_names}")
            
            context = {
                'location': location.name,
                'location_type': location.location_type,
                'current_topic': 'general chat',
                'valid_locations': list(self.locations.keys())
            }
            
            conversation_started = self._find_and_initiate_conversation(agent, nearby_agents_names, context)
            if conversation_started:
                print(f"[DEBUG] Conversation successfully initiated for {agent.name}")
            else:
                print(f"[DEBUG] No conversation initiated for {agent.name} - all agents busy or on cooldown")
                
        except Exception as e:
            print(f"[ERROR] Error during social opportunity check at location for {agent.name}: {str(e)}")
            traceback.print_exc()

    def _check_for_social_opportunity_on_path(self, agent: 'Agent', coord: Tuple[int, int], location_name: str) -> bool:
        """During travel, check for nearby agents at a specific coordinate."""
        try:
            if agent.current_activity in ['resting', 'working', 'socializing']:
                return False

            nearby_agents_names = self._get_nearby_agents_on_path(agent, coord)
            # Don't try to talk if there are more than 2 people (crowd) or no one
            if not (1 <= len(nearby_agents_names) <= 2):
                 return False

            context = {
                'location': f"On the path near {location_name}",
                'location_type': 'path',
                'current_topic': 'passing by',
                'valid_locations': list(self.locations.keys())
            }
            return self._find_and_initiate_conversation(agent, nearby_agents_names, context)
        except Exception as e:
            print(f"[ERROR] Error during social opportunity check on path for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False

    def _handle_action_at_location(self, agent: 'Agent', plan: dict, current_time: int) -> Tuple[bool, Dict[str, Any]]:
        """Handle an action at a location based on the agent's plan."""
        # CRITICAL: Check for zero energy before attempting any action
        current_energy = agent.energy_system.get_energy(agent.name)
        if current_energy <= 0:
            print(f"[CRITICAL] {agent.name} has {current_energy} energy - cannot perform action, forcing to residence")
            return False, {'error': 'Energy depleted to 0', 'replan_needed': True, 'reason_code': 'CRITICAL_ENERGY_ZERO'}
        
        # Get the current location name before acquiring the lock
        current_position = self.location_tracker.get_agent_position(agent.name)
        if not current_position:
            return False, {'error': 'Agent has no current position'}
        current_location_name, _, _ = current_position
            
        # Lock the specific location to ensure thread-safe state modifications
        with self.location_lock_mgr.location_lock(current_location_name):
            try:
                # Get the Location object
                location = self.location_tracker.get_location(current_location_name)
                if not location:
                    # Try to get from agent's locations
                    if hasattr(agent, 'locations') and current_location_name in agent.locations:
                        location = agent.locations[current_location_name]
                    else:
                        return False, {'error': f'Location {current_location_name} not found'}
                
                # Check if location is open at current time
                if not location.is_open(TimeManager.get_current_hour()):
                    return False, {'error': 'Location is closed'}
                
                # Get the action from the plan
                action = plan.get('action', '').lower()
                
                # Handle different action types
                if action == 'work':
                    return self._handle_work(agent, location, plan)
                elif action == 'eat':
                    # Eating is now a two-step process for dining out: purchase, then consume.
                    if location.name == agent.residence:
                        # Eating at home is a single check-and-consume action.
                        return self._handle_eat(agent, location)
                    else:
                        # First, handle the purchase of the meal.
                        purchase_success, purchase_result = self._handle_purchase(agent, location, plan)
                        if not purchase_success:
                            # If purchase fails (e.g., can't afford), propagate the failure.
                            return False, purchase_result
                        
                        # If purchase succeeds, then handle the consumption (energy gain).
                        return self._handle_eat(agent, location)
                elif action == 'rest':
                    return self._handle_rest(agent, location)
                elif action == 'idle':
                    return self._handle_idle(agent, location)
                elif action == 'shop':
                    return self._handle_purchase(agent, location, plan)
                elif action == 'go_to':
                    # Travel-only action - agent has already traveled to the target location
                    # This is a successful completion of a movement activity
                    agent.current_activity = 'traveling'
                    return True, {
                        'status': f'Successfully traveled to {location.name}',
                        'action': 'go_to',
                        'location': location.name
                    }
                elif action == 'socialize':
                    # This is now a passive action. Socializing is handled emergently
                    # by _check_for_social_opportunity_at_location after any action.
                    return True, {'status': 'Passive socializing time, observing surroundings.'}
                else:
                    return False, {'error': f'Unknown action type: {action}'}
                    
            except Exception as e:
                print(f"Error handling action at location for {agent.name}: {str(e)}")
                traceback.print_exc()
                return False, {'error': str(e)}

    def _handle_eat(self, agent: 'Agent', location: 'Location') -> Tuple[bool, Dict[str, Any]]:
        """
        Handle eating at a location with consistent energy gains based on meal type.
        Uses the meal type from the recent purchase for accurate energy calculation.
        """
        try:
            current_time = TimeManager.get_current_hour()
            if not location.is_open(current_time):
                return False, {'error': 'Location is closed'}
            
            # Scenario 1: Dining out (Restaurant or similar shop)
            # This path assumes the purchase has already been successfully handled.
            if location.location_type in ['restaurant', 'local_shop']:
                # Get the meal type from the most recent purchase to ensure consistency
                recent_purchase = None
                if agent.purchase_history:
                    # Find the most recent purchase at this location
                    for purchase in reversed(agent.purchase_history):
                        if (purchase.get('location') == location.name and 
                            purchase.get('timestamp', {}).get('day') == TimeManager.get_current_day() and
                            purchase.get('timestamp', {}).get('hour') == current_time):
                            recent_purchase = purchase
                            break
                
                # Determine energy gain based on meal type
                energy_gain = ENERGY_GAIN_RESTAURANT_MEAL  # Default fallback
                meal_type = 'meal'  # Default fallback
                
                if recent_purchase and 'items' in recent_purchase:
                    # Get the meal type from the purchase record
                    for item in recent_purchase['items']:
                        item_meal_type = item.get('meal_type', 'meal')
                        if item_meal_type == 'snack':
                            energy_gain = ENERGY_GAIN_SNACK
                            meal_type = 'snack'
                        elif item_meal_type in ['breakfast', 'lunch', 'dinner']:
                            energy_gain = ENERGY_GAIN_RESTAURANT_MEAL
                            meal_type = item_meal_type
                        break  # Use the first item's meal type
                
                # Apply energy gain
                energy_before_meal = agent.energy_system.get_energy(agent.name)
                result = agent.energy_system.update_energy(agent.name, energy_gain)
                if not result.success:
                    print(f"Failed to update energy for {agent.name}: {result.error}")
                    return False, {'error': f'Failed to update energy: {result.error}'}
                
                # Log meal energy gain
                energy_after_meal = agent.energy_system.get_energy(agent.name)
                print(f"[ENERGY] {agent.name} restaurant meal: {energy_before_meal} → {energy_after_meal} (+{energy_gain} {meal_type})")
                
                # Record detailed meal data with meal type information
                meal_data = {
                    'type': 'eat',
                    'location': location.name,
                    'meal_type': meal_type,
                    'energy_gain': energy_gain,
                    'timestamp': {
                        'day': TimeManager.get_current_day(),
                        'hour': current_time
                    }
                }
                
                self.memory_mgr.add_memory(
                    agent_name=agent.name, 
                    memory_data=meal_data, 
                    memory_type=MemoryType.MEAL.value
                )
                agent.current_activity = 'eating'
                return True, meal_data

            # Scenario 2: Eating at home
            elif location.name == agent.residence:
                # Get grocery cost from the centralized system
                grocery_cost = agent.grocery_system.get_grocery_cost_for_meal()
                current_groceries = agent.grocery_system.get_level(agent.name)
                if current_groceries < grocery_cost:
                    fail_reason = f'Insufficient groceries for home meal. Needs {grocery_cost}, has {current_groceries}'
                    self.memory_mgr.add_memory(
                        agent_name=agent.name, 
                        memory_data={'reason': fail_reason}, 
                        memory_type=MemoryType.FAILED_ACTION.value
                    )
                    # Signal that a replan is needed
                    return False, {'error': fail_reason, 'replan_needed': True, 'reason_code': 'INSUFFICIENT_GROCERIES'}
                
                # Consume groceries and gain energy
                agent.grocery_system.update_grocery_level(agent.name, -grocery_cost)
                # Use the fixed energy gain for home meals
                energy_gain = ENERGY_GAIN_HOME_MEAL
                energy_before_meal = agent.energy_system.get_energy(agent.name)
                result = agent.energy_system.update_energy(agent.name, energy_gain)
                if not result.success:
                    print(f"Failed to update energy for {agent.name}: {result.error}")
                    return False, {'error': f'Failed to update energy: {result.error}'}

                # Determine home meal type based on current time
                from simulation_types import get_meal_period
                meal_type = get_meal_period(current_time)
                
                # Log home meal energy gain
                energy_after_meal = agent.energy_system.get_energy(agent.name)
                print(f"[ENERGY] {agent.name} home meal: {energy_before_meal} → {energy_after_meal} (+{energy_gain} home_{meal_type})")
                
                meal_data = {
                    'type': 'eat',
                    'location': location.name,
                    'meal_type': f'home_{meal_type}',  # Distinguish home meals
                    'energy_gain': energy_gain,
                    'grocery_cost': grocery_cost,
                    'timestamp': {
                        'day': TimeManager.get_current_day(),
                        'hour': current_time
                    }
                }
                
                self.memory_mgr.add_memory(
                    agent_name=agent.name, 
                    memory_data=meal_data, 
                    memory_type=MemoryType.MEAL.value
                )
                agent.current_activity = 'eating'
                return True, meal_data

            else:
                return False, {'error': f'Location {location.name} is not a valid place to eat.'}
            
        except Exception as e:
            print(f"Error handling eating for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False, {'error': str(e)}

    def _handle_rest(self, agent: 'Agent', location: 'Location') -> Tuple[bool, Dict[str, Any]]:
        """Handle resting at a location - only valid for naps (workplace 11-15) or sleep (home 23-6)."""
        try:
            current_time = TimeManager.get_current_hour()
            
            # Validate rest timing and location explicitly
            is_nap_time = 11 <= current_time <= 15
            is_sleep_time = 23 <= current_time or current_time <= 6
            is_at_workplace = location.name == agent.workplace
            is_at_home = location.name == agent.residence
            
            # Check if this is a valid rest scenario
            if is_nap_time and is_at_workplace:
                # Valid nap at workplace - use energy gain constant
                energy_gain = ENERGY_GAIN_NAP
                rest_type = 'nap'
                # Update energy for naps (add energy)
                result = agent.energy_system.update_energy(agent.name, energy_gain)
            elif is_sleep_time:
                # Sleep during 23-6 is handled by automatic sleep system, not manual rest
                return False, {'error': f'Sleep during hours 23-6 is handled automatically. Use "idle" for manual relaxation during sleep hours.'}
            else:
                # Invalid rest scenario - provide helpful error message
                if not is_nap_time:
                    return False, {'error': f'Invalid rest time: {current_time}:00. Use "rest" only for naps (11-15). Sleep (23-6) is automatic. Use "idle" for general relaxation.'}
                elif is_nap_time and not is_at_workplace:
                    return False, {'error': f'Cannot nap at {location.name} during {current_time}:00. Naps only allowed at workplace ({agent.workplace}) during 11-15.'}
                else:
                    return False, {'error': f'Invalid rest conditions: location={location.name}, time={current_time}, home={agent.residence}, workplace={agent.workplace}'}

            # Check if energy update was successful
            if not result.success:
                print(f"Failed to update energy for {agent.name}: {result.error}")
                return False, {"error": f"Failed to update energy: {result.error}"}
            
            # Record rest in memory
            rest_data = {
                'type': 'rest',
                'location': location.name,
                'rest_type': rest_type,
                'energy_gain': energy_gain,
                'timestamp': {
                    'day': TimeManager.get_current_day(),
                    'hour': current_time
                }
            }
            
            # Add memory using MemoryType.REST
            self.memory_mgr.add_memory(
                agent_name=agent.name, 
                memory_data=rest_data, 
                memory_type=MemoryType.REST.value
            )
            
            # Update agent's activity
            agent.current_activity = 'resting'
            
            return True, rest_data
            
        except Exception as e:
            print(f"Error handling rest for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False, {'error': str(e)}

    def _handle_idle(self, agent: 'Agent', location: 'Location') -> Tuple[bool, Dict[str, Any]]:
        """Handle idling at a location - no energy gain, just passing time."""
        try:
            current_time = TimeManager.get_current_hour()
            
            # Idle doesn't have location/time restrictions like rest does
            # Anyone can idle anywhere at any time - it's just relaxing/waiting
            
            # No energy gain for idle - it's just passing time
            energy_gain = 0
            
            # Record idle activity in memory
            idle_data = {
                'type': 'idle',
                'location': location.name,
                'description': 'Relaxing and passing time with no specific activity',
                'energy_gain': energy_gain,
                'timestamp': {
                    'day': TimeManager.get_current_day(),
                    'hour': current_time
                }
            }
            
            # Add memory using MemoryType.REST (since it's a relaxation activity)
            self.memory_mgr.add_memory(
                agent_name=agent.name, 
                memory_data=idle_data, 
                memory_type=MemoryType.REST.value
            )
            
            # Update agent's activity
            agent.current_activity = 'idling'
            
            return True, idle_data
            
        except Exception as e:
            print(f"Error handling idle for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False, {'error': str(e)}

    def _handle_purchase(self, agent: 'Agent', location: 'Location', action_details: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Simplified purchase handling using meal types (breakfast, lunch, dinner, snack).
        Uses current time to determine appropriate meal type instead of complex menu parsing.
        """
        try:
            current_time = TimeManager.get_current_hour()
            if not location.is_open(current_time):
                return False, {'error': 'Location is closed'}

            purchase_type = 'meal' if action_details.get('action') == 'eat' else 'goods'
            
            if purchase_type == 'meal':
                # Determine meal type based on current time
                from simulation_types import get_meal_period
                requested_meal_type = get_meal_period(current_time)

                # Use menu validator to validate and get the best available option
                is_valid, validation_result = self.menu_validator.validate_food_request(
                    agent.name, location.name, requested_meal_type, current_time
                )

                if is_valid:
                    # Validation successful, use the validated item
                    selected_meal_type = validation_result['meal_type']
                    base_cost = validation_result['price']
                    valid_item_name = validation_result['valid_item']

                    print(f"[MENU] {agent.name} ordering {valid_item_name} at {location.name} (${base_cost})")

                else:
                    # Validation failed, try emergency food options
                    print(f"[MENU] {agent.name} requested {requested_meal_type} at {location.name} failed: {validation_result['error']}")

                    # Try to get emergency food options within agent's budget
                    emergency_options = self.menu_validator.get_emergency_food_options(
                        current_time, agent.money
                    )

                    if emergency_options:
                        # Use the cheapest available option
                        emergency_food = emergency_options[0]
                        print(f"[EMERGENCY] {agent.name} switching to emergency food: {emergency_food['item']} at {emergency_food['location']}")

                        selected_meal_type = emergency_food['meal_type']
                        base_cost = emergency_food['price']
                        valid_item_name = emergency_food['item']

                        # If emergency food is at different location, this will fail gracefully
                        if emergency_food['location'] != location.name:
                            return False, {
                                'error': f"No valid food at {location.name}. Suggested: {emergency_food['item']} at {emergency_food['location']}",
                                'replan_needed': True,
                                'suggestion': emergency_food['location']
                            }
                    else:
                        # No emergency options available - agent might starve
                        self.menu_validator.log_invalid_request(agent.name, location.name, requested_meal_type, current_time, validation_result)
                        return False, {
                            'error': f"No affordable food available. {validation_result.get('suggestion', '')}",
                            'replan_needed': True,
                            'reason_code': 'NO_AFFORDABLE_FOOD'
                        }
                
                # Create simplified item record
                items_to_purchase = [{
                    'meal_type': selected_meal_type,
                    'base_price': base_cost
                }]
                
                # Apply discount if applicable
                final_cost = base_cost
                has_discount = False
                discount_config = location_config.get('discount', {})
                if discount_config:
                    discount_days = discount_config.get('days', [])
                    current_day = TimeManager.get_current_day()
                    
                    if current_day in discount_days:
                        discount_value = discount_config.get('value', 0)
                        discount_type = discount_config.get('type', 'percentage')
                        
                        if discount_type == 'percentage':
                            final_cost = base_cost * (1 - discount_value / 100)
                        else:  # flat discount
                            final_cost = max(0, base_cost - discount_value)
                        
                        has_discount = True
                        # Update item record with discount info
                        items_to_purchase[0]['final_price'] = final_cost
                        items_to_purchase[0]['discount_applied'] = discount_value
            
            else:  # purchase_type == 'goods' (grocery shopping)
                base_cost = location.prices.get('default', 15.0)
                final_cost = base_cost
                has_discount = False
                items_to_purchase = [{'meal_type': 'groceries', 'base_price': base_cost, 'final_price': final_cost}]

            # Check affordability
            if not agent.can_afford_purchase(final_cost):
                reason = f'Not enough money. Needs ${final_cost:.2f}, has ${agent.money:.2f}'
                return False, {'error': reason, 'replan_needed': True, 'reason_code': 'INSUFFICIENT_FUNDS'}

            # Execute the purchase
            purchase_record = agent.record_purchase(
                amount=final_cost,
                location=location.name,
                items=items_to_purchase,
                has_discount=has_discount
            )
            if not purchase_record:
                return False, {'error': 'Failed to record purchase for agent.'}

            # Record the sale for business metrics
            self.metrics_mgr.record_sale(
                location_name=location.name,
                agent_name=agent.name,
                amount=final_cost,
                items=items_to_purchase,
                has_discount=has_discount,
                simulation_day=TimeManager.get_current_day(),
                simulation_hour=current_time
            )

            # Handle grocery purchases (gain grocery levels)
            if location.location_type == 'grocery':
                grocery_gain = int(final_cost)  # Gain equivalent to cost
                agent.grocery_system.update_grocery_level(agent.name, grocery_gain)
            
            agent.current_activity = 'shopping'
            return True, purchase_record
            
        except Exception as e:
            print(f"Error handling purchase for {agent.name}: {str(e)}")
            traceback.print_exc()
            return False, {'error': str(e)}

class ConversationState:
    """Manages the state of a conversation."""
    def __init__(self):
        self.turns = []
        self.max_turns = MAX_CONVERSATION_TURNS
        self.current_turn = 0

    def add_turn(self, speaker: str, content: str):
        """Add a turn to the conversation."""
        self.turns.append({
            'speaker': speaker,
            'dialogue': content,  # Use 'dialogue' to match prompt expectations
            'turn_number': self.current_turn
        })
        self.current_turn += 1

    def should_continue(self) -> bool:
        """Check if conversation should continue."""
        return self.current_turn < self.max_turns

    def get_last_turns(self, num_turns: int = 3) -> List[str]:
        """Get the last N turns of conversation."""
        return self.turns[-num_turns:] if self.turns else []

class ConversationManager:
    """Manages conversations between agents in the simulation."""
    def __init__(self, memory_manager, locations=None):
        self.memory_manager = memory_manager
        self.locations = locations or {}  # Store locations for commitment validation
        self.active_conversations = {}
        # Tracks the time of the last conversation for each pair of agents
        self.last_conversation_time = {} 
        self.cooldown_hours = CONVERSATION_COOLDOWN_HOURS # Use constant from simulation_types
        self._lock = threading.Lock()

    def can_converse(self, agent1_name: str, agent2_name: str) -> bool:
        """Check if two agents can have a conversation based on a cooldown period."""
        # Sort names to ensure the key is always the same regardless of who initiates
        key = tuple(sorted((agent1_name, agent2_name)))
        last_convo_time = self.last_conversation_time.get(key)

        if last_convo_time is None:
            print(f"[DEBUG] Cooldown check: {agent1_name} and {agent2_name} have never talked before - ALLOWED")
            return True # They've never talked before

        last_day, last_hour = last_convo_time
        current_day = TimeManager.get_current_day()
        current_hour = TimeManager.get_current_hour()

        # Calculate total hours passed since the last conversation
        hours_since = (current_day - last_day) * 24 + (current_hour - last_hour)
        
        can_talk = hours_since >= self.cooldown_hours
        print(f"[DEBUG] Cooldown check: {agent1_name} and {agent2_name} last talked Day {last_day}, Hour {last_hour}. Hours since: {hours_since}/{self.cooldown_hours} - {'ALLOWED' if can_talk else 'BLOCKED'}")
        
        return can_talk

    def _determine_relationship_context(self, speaker: 'Agent', all_participants: List['Agent']) -> Dict[str, str]:
        """Determine relationship between speaker and each participant based on config data only."""
        relationship_context = {}
        
        for participant in all_participants:
            if participant.name == speaker.name:
                continue  # Skip self
                
            # Default to acquaintance
            relationship = 'acquaintance'
            
            # Check spouse relationship (highest priority)
            if hasattr(speaker, 'spouse') and speaker.spouse == participant.name:
                relationship = 'spouse'
            
            # Check household members for specific relationship types (roommates, spouses)
            elif hasattr(speaker, 'household_members') and participant.name in speaker.household_members:
                # Look through the full relationship config for detailed information
                if hasattr(speaker, 'relationships') and 'household_members' in speaker.relationships:
                    household_config = speaker.relationships['household_members']
                    for member_data in household_config:
                        if isinstance(member_data, dict) and member_data.get('name') == participant.name:
                            rel_type = member_data.get('relationship_type', 'household_member')
                            if rel_type == 'roommate':
                                relationship = 'roommate'
                            elif rel_type == 'spouse':
                                relationship = 'spouse'
                            else:
                                relationship = 'household_member'
                            break
                else:
                    # If participant name is in household_members but no detailed config found
                    relationship = 'household_member'
            
            # Check best friends (people who don't live together)
            elif hasattr(speaker, 'best_friends') and participant.name in speaker.best_friends:
                relationship = 'best_friend'
            
            # Check dating relationships (people who don't live together)
            elif hasattr(speaker, 'dating') and participant.name in speaker.dating:
                relationship = 'dating'
            
            # Check bidirectional relationships (in case config is one-way)
            elif hasattr(participant, 'spouse') and participant.spouse == speaker.name:
                relationship = 'spouse'
            elif hasattr(participant, 'household_members') and speaker.name in participant.household_members:
                # Check participant's household_members for relationship type
                if hasattr(participant, 'relationships') and 'household_members' in participant.relationships:
                    household_config = participant.relationships['household_members']
                    for member_data in household_config:
                        if isinstance(member_data, dict) and member_data.get('name') == speaker.name:
                            rel_type = member_data.get('relationship_type', 'household_member')
                            if rel_type == 'roommate':
                                relationship = 'roommate'
                            elif rel_type == 'spouse':
                                relationship = 'spouse'
                            else:
                                relationship = 'household_member'
                            break
                else:
                    relationship = 'household_member'
            elif hasattr(participant, 'best_friends') and speaker.name in participant.best_friends:
                relationship = 'best_friend'
            elif hasattr(participant, 'dating') and speaker.name in participant.dating:
                relationship = 'dating'
            
            relationship_context[participant.name] = relationship
        
        return relationship_context

    def handle_conversation(self, initiator: 'Agent', participants: List['Agent'], memory_mgr: MemoryManagerInterface, plan_executor: 'PlanExecutor', prompt_mgr: 'PromptManager', model_mgr: 'ModelManager', context: Dict) -> str:
        """
        Handle a conversation between agents synchronously.
        The initiator's thread controls the entire conversation, eliminating deadlocks.
        """
        try:
            all_participants = sorted([initiator] + participants, key=lambda x: x.name)
            participant_map = {p.name: p for p in all_participants}

            conv_state = ConversationState()
            
            # Determine conversation type - simplified, no location dependency
            conversation_type = "conversation_general"
            
            # The conversation happens as a series of sequential turns.
            while conv_state.should_continue():
                for current_speaker in all_participants:
                    if not conv_state.should_continue():
                        break # Max turns reached mid-round

                    # Listeners are all other participants in the conversation.
                    listeners = [p.name for p in all_participants if p.name != current_speaker.name]
                    
                    # Determine relationship for this specific speaker to other participants
                    speaker_relationships = self._determine_relationship_context(current_speaker, all_participants)
                    
                    # For the prompt, use the relationship to the primary listener (first other participant)
                    primary_listener = listeners[0] if listeners else 'unknown'
                    speaker_relationship = speaker_relationships.get(primary_listener, 'acquaintance')
                    
                    # Prepare context for prompt manager
                    prompt_context = {
                        'speaker': current_speaker.name,
                        'listener': listeners[0] if listeners else 'unknown',  # Pass single listener name
                        'location': context.get('location'),
                        'time': TimeManager.get_current_hour(),
                        'day': TimeManager.get_current_day(),
                        'relationship': speaker_relationship,  # Now config-based, not location-based
                        'previous_interaction': conv_state.get_last_turns(3),
                        'current_topic': context.get('current_topic', ''),
                        'location_type': context.get('location_type', ''),
                        'speaker_agent': current_speaker,
                        'conversation_type': conversation_type,
                        # Add valid locations to prevent hallucinated locations
                        'valid_locations': context.get('valid_locations', []),
                        # Add location hours information for conversation context
                        'location_hours_info': self._get_location_hours_info(TimeManager.get_current_hour()),
                        # Add dynamic context to make each prompt unique
                        'energy_level': current_speaker.energy_system.get_energy(current_speaker.name),
                        'money': current_speaker.money,
                        'grocery_level': current_speaker.grocery_system.get_level(current_speaker.name),
                        'current_activity': current_speaker.current_activity,
                        'household_members': current_speaker.household_members,
                        'best_friends': getattr(current_speaker, 'best_friends', []),
                        'dating': getattr(current_speaker, 'dating', []),
                        'nearby_agents': [p.name for p in all_participants if p.name != current_speaker.name],
                        # Add unique identifiers to break determinism
                        'conversation_turn': conv_state.current_turn,
                        'total_participants': len(all_participants),
                        'speaker_occupation': current_speaker.occupation,
                        'speaker_age': current_speaker.age,
                        'speaker_workplace': current_speaker.workplace,
                        'speaker_residence': current_speaker.residence,
                        # Add employment and schedule information to prevent hallucinated activities
                        'employment_type': current_speaker.income_info.get('type', 'hourly'),
                        'work_schedule': current_speaker.income_info.get('schedule', {}),
                        'is_student': False,  # Explicitly set based on occupation (can be enhanced)
                        'typical_work_hours': self._get_work_hours_description(current_speaker.income_info.get('schedule', {}))
                    }
                    
                    # Generate conversation turn using prompt manager
                    conversation_prompt = prompt_mgr.get_prompt("conversation", prompt_context)
                    
                    # Call LLM to generate the actual conversation response
                    conversation_result = model_mgr.generate(conversation_prompt, "conversation")
                    
                    # Parse conversation response - handle both string and structured responses
                    turn_data = {}
                    
                    # The model_mgr.generate() typically returns a string, so handle that case
                    if isinstance(conversation_result, str):
                        clean_result = conversation_result.strip()
                        
                        # Try to parse as JSON if it looks like structured data
                        if (clean_result.startswith('{') and clean_result.endswith('}')) or clean_result.startswith('```'):
                            try:
                                # Remove markdown code blocks if present
                                if clean_result.startswith('```json'):
                                    clean_result = clean_result[7:-3].strip()
                                elif clean_result.startswith('```'):
                                    clean_result = clean_result[3:-3].strip()
                                
                                turn_data = json.loads(clean_result)
                            except json.JSONDecodeError as e:
                                print(f"[WARNING] Failed to parse JSON conversation response: {e}")
                                # Fallback to treating entire response as dialogue
                                turn_data = {
                                    'dialogue': clean_result,
                                    'commitments': []
                                }
                        else:
                            # Plain text response - treat as dialogue and try to extract commitments
                            turn_data = {
                                'dialogue': clean_result,
                                'commitments': self._extract_commitments_from_dialogue(clean_result, current_speaker, all_participants, context)
                            }
                    else:
                        # If it's already a dict (shouldn't happen with current implementation)
                        turn_data = conversation_result if isinstance(conversation_result, dict) else {'dialogue': str(conversation_result), 'commitments': []}
                    
                    # Safely extract dialogue and commitments
                    dialogue = turn_data.get('dialogue', str(conversation_result))
                    commitments = turn_data.get('commitments', [])
                    
                    # Print conversation to terminal for debugging
                    print(f"[CONVERSATION] {current_speaker.name}: {dialogue}")
                    
                    # Add turn to conversation state
                    conv_state.add_turn(current_speaker.name, dialogue)
                    
                    # Process any commitments made in this turn
                    if commitments:
                        print(f"[INFO] Processing {len(commitments)} commitments made by {current_speaker.name}.")
                        for commitment in commitments:
                            self._process_commitment(commitment, current_speaker, participant_map)
                    
                    # Store conversation turn in each participant's memory
                    for p in all_participants:
                        memory_mgr.add_memory(
                            agent_name=p.name,
                            memory_type=MemoryType.CONVERSATION.value,
                            memory_data={
                                'dialogue': f"{current_speaker.name}: {dialogue}",
                                'participants': [p.name for p in all_participants],
                                'location': context.get('location'),
                                'turn_number': conv_state.current_turn - 1
                            }
                        )

                    # Note: Conversations no longer provide energy gain

            # Record the time of this conversation to enforce the cooldown
            # Use the same key format as can_converse() - two-agent tuple
            for i, participant1 in enumerate(all_participants):
                for participant2 in all_participants[i+1:]:
                    key = tuple(sorted([participant1.name, participant2.name]))
                    self.last_conversation_time[key] = (TimeManager.get_current_day(), TimeManager.get_current_hour())
                    print(f"[DEBUG] Recorded cooldown for {participant1.name} and {participant2.name} at Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
                
            # Return conversation summary
            return f"Conversation completed between {', '.join(p.name for p in all_participants)} over {conv_state.current_turn} turns."
                
        except Exception as e:
            print(f"Error handling conversation: {str(e)}")
            traceback.print_exc()
            return f"Error in conversation: {str(e)}"
            
    def _process_commitment(self, commitment: Dict[str, Any], speaker: 'Agent', participant_map: Dict[str, 'Agent']):
        """Processes a single commitment and updates the plans of involved agents."""
        try:
            print(f"[DEBUG] Processing commitment: {commitment}")
            
            schedule_update = commitment.get('schedule_update')
            if not schedule_update:
                print(f"[DEBUG] No schedule_update found in commitment: {commitment.keys()}")
                return

            print(f"[DEBUG] Found schedule_update: {schedule_update}")

            new_activities = []
            # Sort by time to ensure chronological processing
            for time_str, activity_details in sorted(schedule_update.items()):
                try:
                    # Handle both "18:00" and "18" formats
                    if ':' in time_str:
                        hour = int(time_str.split(':')[0])
                    else:
                        hour = int(time_str)
                    
                    # Validate target location exists in simulation
                    target_location = activity_details.get("target", "")
                    if target_location not in self.locations:
                        print(f"[ERROR] Invalid location in commitment: '{target_location}' is not a valid location")
                        print(f"[ERROR] Valid locations: {list(self.locations.keys())}")
                        print(f"[ERROR] Skipping commitment with invalid location")
                        return  # Skip entire commitment if any location is invalid
                    
                    activity = {
                        "time": hour,
                        "action": activity_details.get("action"),
                        "target": target_location,
                        "description": activity_details.get("reasoning", activity_details.get("description", "")),
                    }
                    new_activities.append(activity)
                    print(f"[DEBUG] Created activity: {activity}")
                except (ValueError, AttributeError) as e:
                    print(f"[WARNING] Could not parse activity from commitment: {time_str}: {activity_details} - Error: {e}")
                    continue
                    
            if not new_activities:
                print(f"[DEBUG] No valid activities created from commitment")
                return

            start_hour = new_activities[0]['time']
            print(f"[DEBUG] Commitment start hour: {start_hour}")
            
            # The speaker and anyone they made a commitment 'with' needs their plan updated
            agents_to_update_names = [speaker.name] + commitment.get('with', [])
            print(f"[DEBUG] Agents to update: {agents_to_update_names}")
            
            # Use a set to prevent updating the same agent twice
            for agent_name in set(agents_to_update_names):
                if agent_name in participant_map:
                    agent_to_update = participant_map[agent_name]
                    print(f"[INFO] Updating {agent_to_update.name}'s plan based on a commitment starting at {start_hour}:00.")
                    print(f"[DEBUG] New activities for {agent_to_update.name}: {new_activities}")
                    
                    # Store old plan for debugging
                    old_activities = agent_to_update.daily_plan.get('activities', []) if agent_to_update.daily_plan else []
                    old_activity_count = len(old_activities)
                    
                    agent_to_update.update_plan_slice(new_activities, start_hour)
                    
                    # Check if plan was actually updated
                    new_plan_activities = agent_to_update.daily_plan.get('activities', []) if agent_to_update.daily_plan else []
                    print(f"[DEBUG] {agent_to_update.name} plan updated: {old_activity_count} → {len(new_plan_activities)} activities")
                    
                    # Show what activities were added/changed
                    commitment_hours = {act['time'] for act in new_activities}
                    affected_activities = [act for act in new_plan_activities if act.get('time') in commitment_hours]
                    print(f"[DEBUG] {agent_to_update.name} affected activities: {affected_activities}")
                else:
                    print(f"[WARNING] Agent {agent_name} not found in participant_map")
                    
        except Exception as e:
            print(f"[ERROR] Error processing commitment: {str(e)}")
            import traceback
            traceback.print_exc()

    def _get_location_hours_info(self, current_hour: int) -> str:
        """Get location hours information for conversation context."""
        try:
            info_lines = []
            info_lines.append(f"Current time: {current_hour}:00")
            
            # Try to get actual location data if available through memory manager or other means
            # For now, provide general business hours guidance
            
            # General business hours guidance
            if 6 <= current_hour <= 21:
                info_lines.append("Most restaurants/shops open (typical hours: 7:00-21:00)")
            else:
                info_lines.append("Most restaurants/shops likely closed (outside typical 7:00-21:00 hours)")
            
            # Meal time specific guidance
            if 6 <= current_hour <= 9:
                info_lines.append("Breakfast time - breakfast places likely serving")
            elif 11 <= current_hour <= 14:
                info_lines.append("Lunch time - most dining locations serving")
            elif 17 <= current_hour <= 20:
                info_lines.append("Dinner time - most dining locations serving")
            elif 22 <= current_hour or current_hour <= 5:
                info_lines.append("Late night/early morning - most places closed except 24-hour locations")
            else:
                info_lines.append("Between meal times - snacks/beverages available at open locations")
            
            # Add a note about checking specific locations
            info_lines.append("Consider location-specific hours when making plans")
            
            return " | ".join(info_lines)
            
        except Exception as e:
            print(f"Error getting location hours info: {str(e)}")
            return f"Current time: {current_hour}:00 | Check individual location hours"

    def _get_work_hours_description(self, schedule: Dict[str, Any]) -> str:
        """Helper to describe work hours in a human-readable format."""
        work_type = schedule.get('type', 'full_time')
        if work_type == 'full_time':
            work_hours = schedule.get('work_hours', {})
            start_hour = work_hours.get('start', 9)
            end_hour = work_hours.get('end', 17)
            days = schedule.get('days', [])
            day_names = [f"Day {day}" for day in days]
            if len(day_names) == 5:
                return f"Full-time work from {start_hour}:00 to {end_hour}:00 on {', '.join(day_names)}"
            elif len(day_names) == 7:
                return f"Full-time work from {start_hour}:00 to {end_hour}:00 on {', '.join(day_names)}"
            else:
                return f"Full-time work from {start_hour}:00 to {end_hour}:00 on {', '.join(day_names)}"
        elif work_type == 'part_time':
            work_hours = schedule.get('work_hours', {})
            start_hour = work_hours.get('start', 9)
            end_hour = work_hours.get('end', 17)
            days = schedule.get('days', [])
            day_names = [f"Day {day}" for day in days]
            if len(day_names) == 5:
                return f"Part-time work from {start_hour}:00 to {end_hour}:00 on {', '.join(day_names)}"
            elif len(day_names) == 7:
                return f"Part-time work from {start_hour}:00 to {end_hour}:00 on {', '.join(day_names)}"
            else:
                return f"Part-time work from {start_hour}:00 to {end_hour}:00 on {', '.join(day_names)}"
        elif work_type == 'hourly':
            return "Hourly wage worker, typically working 8-10 hours per day."
        elif work_type == 'daily':
            return "Daily wage worker, typically working 8-10 hours per day."
        elif work_type == 'weekly':
            return "Weekly wage worker, typically working 40 hours per week."
        elif work_type == 'monthly':
            return "Monthly wage worker, typically working 160 hours per month."
        elif work_type == 'salary':
            return "Salaried worker, typically working 40 hours per week."
        else:
            return "Unknown employment type."

    def save_conversation_logs(self, filepath: Optional[str] = None) -> None:
        """Save conversation logs to file using the memory manager's _save_conversation method."""
        try:
            # Get all conversation memories from the memory manager
            conversation_count = 0
            
            # Iterate through all agents and their conversation memories
            for agent_name, memories in self.memory_manager.agent_memories.items():
                for memory in memories:
                    if memory.get('type') == MemoryType.CONVERSATION.value:
                        # Extract the conversation content from the memory structure
                        conversation_content = memory.get('content', {})
                        
                        # Create the conversation data structure expected by _save_conversation
                        conversation_data = {
                            'dialogue': conversation_content.get('dialogue', ''),
                            'participants': conversation_content.get('participants', []),
                            'location': conversation_content.get('location', ''),
                            'turn_number': conversation_content.get('turn_number', 0),
                            'simulation_day': memory.get('day', TimeManager.get_current_day()),
                            'simulation_hour': memory.get('hour', TimeManager.get_current_hour())
                        }
                        
                        # Use the memory manager's existing _save_conversation method
                        self.memory_manager._save_conversation(agent_name, conversation_data)
                        conversation_count += 1
            
            print(f"Saved {conversation_count} conversations to: {self.memory_manager.conversation_file}")
                
        except Exception as e:
            print(f"Error saving conversation logs: {str(e)}")
            traceback.print_exc()

    def _extract_commitments_from_dialogue(self, dialogue: str, speaker: 'Agent', all_participants: List['Agent'], context: Dict) -> List[Dict[str, Any]]:
        """Extract commitments from plain text dialogue using pattern matching."""
        try:
            commitments = []
            current_hour = TimeManager.get_current_hour()
            
            # Common patterns for making plans
            plan_patterns = [
                # "let's go to Location", "want to go to Location", "heading to Location"
                r"(?i)let[''']?s\s+(?:go\s+to|head\s+to|visit)\s+([A-Za-z\s]+?)(?:\s+(?:now|together|today))?[.\?!]",
                r"(?i)(?:want\s+to|should\s+we)\s+go\s+to\s+([A-Za-z\s]+?)(?:\s+(?:now|together|today))?[.\?!]",
                r"(?i)heading\s+(?:to|over\s+to)\s+([A-Za-z\s]+?)(?:\s+(?:now|together))?[.\?!]",
                
                # "meet at Location", "see you at Location"
                r"(?i)(?:meet|see\s+you)\s+at\s+([A-Za-z\s]+?)(?:\s+(?:now|later|today))?[.\?!]",
                
                # "grab food/dinner/lunch at Location"
                r"(?i)grab\s+(?:some\s+)?(?:food|dinner|lunch|breakfast)\s+at\s+([A-Za-z\s]+?)[.\?!]",
                r"(?i)(?:get|have)\s+(?:some\s+)?(?:food|dinner|lunch|breakfast)\s+at\s+([A-Za-z\s]+?)[.\?!]",
            ]
            
            dialogue_lower = dialogue.lower()
            
            # Check if this dialogue contains planning language
            for pattern in plan_patterns:
                import re
                matches = re.findall(pattern, dialogue)
                for match in matches:
                    location_name = match.strip()
                    
                    # Validate location exists in the simulation
                    valid_locations = context.get('valid_locations', [])
                    # Find best match for location (case-insensitive partial matching)
                    matched_location = None
                    for valid_loc in valid_locations:
                        if location_name.lower() in valid_loc.lower() or valid_loc.lower() in location_name.lower():
                            matched_location = valid_loc
                            break
                    
                    if matched_location:
                        # Create commitment for immediate action (current hour + 1)
                        next_hour = current_hour + 1
                        
                        # Determine action based on dialogue context
                        action = "go_to"
                        if any(food_word in dialogue_lower for food_word in ['food', 'eat', 'dinner', 'lunch', 'breakfast', 'meal']):
                            # If food-related, create a two-step plan: go_to then eat
                            commitment = {
                                "with": [p.name for p in all_participants if p.name != speaker.name],
                                "schedule_update": {
                                    f"{next_hour}:00": {
                                        "action": "go_to",
                                        "target": matched_location,
                                        "reasoning": f"Meeting {', '.join([p.name for p in all_participants if p.name != speaker.name])} as discussed"
                                    },
                                    f"{next_hour + 1}:00": {
                                        "action": "eat",
                                        "target": matched_location,
                                        "reasoning": f"Having meal together with {', '.join([p.name for p in all_participants if p.name != speaker.name])}"
                                    }
                                }
                            }
                        else:
                            # General meeting
                            commitment = {
                                "with": [p.name for p in all_participants if p.name != speaker.name],
                                "schedule_update": {
                                    f"{next_hour}:00": {
                                        "action": "go_to", 
                                        "target": matched_location,
                                        "reasoning": f"Meeting {', '.join([p.name for p in all_participants if p.name != speaker.name])} as discussed"
                                    }
                                }
                            }
                        
                        commitments.append(commitment)
                        print(f"[DEBUG] Extracted commitment from dialogue: {speaker.name} → {matched_location} at hour {next_hour}")
                        break  # Only create one commitment per dialogue turn
            
            return commitments
            
        except Exception as e:
            print(f"[ERROR] Error extracting commitments from dialogue: {str(e)}")
            import traceback
            traceback.print_exc()
            return []