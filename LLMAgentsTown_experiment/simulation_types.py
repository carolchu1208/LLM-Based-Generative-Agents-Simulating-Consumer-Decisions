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
    TYPE_CHECKING, TypeVar, Generic, Protocol
)
from enum import Enum
from dataclasses import dataclass

# Third-party imports
import requests

# Local imports
from thread_safe_base import (
    SimulationError, AgentError, LocationError,
    MemoryError, MetricsError, Result, ThreadSafeBase
)
from simulation_constants import (
    ENERGY_MAX, ENERGY_MIN, ENERGY_DECAY_PER_HOUR,
    ENERGY_COST_WORK_HOUR, ENERGY_COST_PER_STEP,
    ENERGY_GAIN_RESTAURANT_MEAL, ENERGY_GAIN_SNACK,
    ENERGY_GAIN_HOME_MEAL, ENERGY_GAIN_NAP,
    ENERGY_THRESHOLD_LOW, ENERGY_THRESHOLD_FOOD
)

# Type hints for circular imports
if TYPE_CHECKING:
    from main_simulation import Location

# Type definitions
class ActivityType(Enum):
    """Types of activities that agents can perform."""
    GO_TO = "go_to"
    WORK = "work"
    EAT = "eat"
    SHOP = "shop"
    REST = "rest"
    SOCIAL = "social"
    PLAN = "plan"
    IDLE = "idle"

class MemoryType(Enum):
    """Types of memories that can be recorded."""
    # State and Info Types
    STATE_UPDATE = "STATE_UPDATE"  # For tracking agent state changes
    AGENT_INFO = "AGENT_INFO"  # For storing agent's basic information
    RELATIONSHIP_INFO = "RELATIONSHIP_INFO"  # For storing relationship information

    # Activity Types
    ACTIVITY = "ACTIVITY"  # For tracking general activities
    TRAVEL = "TRAVEL"  # For tracking travel events
    WORK = "WORK"  # For tracking work events
    MEAL = "MEAL"  # For tracking meal-related events
    SHOPPING = "SHOPPING"  # For tracking shopping events
    REST = "REST"  # For tracking rest events
    SOCIAL = "SOCIAL"  # For tracking social events
    DINING = "DINING"  # For tracking dining events
    IDLE = "IDLE"  # For tracking idle events

    # Planning Types
    PLAN = "PLAN"  # For tracking plans
    PLAN_CREATION = "PLAN_CREATION"  # For tracking plan creation
    PLAN_UPDATE = "PLAN_UPDATE"  # For tracking plan updates
    PLANNING = "PLANNING"  # For tracking planning events

    # Interaction Types
    CONVERSATION = "CONVERSATION"  # For tracking conversations
    ENCOUNTER = "ENCOUNTER"  # For tracking encounters with other agents
    RELATIONSHIP = "RELATIONSHIP"  # For tracking relationship changes
    COMMITMENT = "COMMITMENT"  # For tracking commitments made

    # Transaction Types
    PURCHASE = "PURCHASE"  # For tracking purchases
    EARNING = "EARNING"  # For tracking earnings

    # Meta/Failure Types
    FAILED_ACTION = "FAILED_ACTION" # For when a planned action fails due to resource constraints

# Conversation limits
MAX_CONVERSATION_TURNS = 4
CONVERSATION_COOLDOWN_HOURS = 4

# Grocery Constants
GROCERY_MAX = 100
GROCERY_MIN = 0
GROCERY_THRESHOLD_LOW = 20
GROCERY_COST_HOME_MEAL = 10

# Financial Constants
INITIAL_MONEY_MULTIPLIER = 3  # Agent starts with 3 days worth of wages
MONEY_MIN = 0  # Minimum money (cannot go negative)
PRICE_GROCERY_PER_LEVEL = 1.0  # Cost per grocery level when shopping
PRICE_RESTAURANT_MEAL = 15.0  # Default restaurant meal price
PRICE_SNACK = 5.0  # Default snack price

# Travel Constants
MAX_STEPS_PER_HOUR = 20  # Maximum number of steps an agent can take in one hour
MINUTES_PER_STEP = 60 // MAX_STEPS_PER_HOUR  # 3 minutes per step
STEPS_PER_MINUTE = MAX_STEPS_PER_HOUR / 60  # 0.33 steps per minute

# Activity Types Configuration
ACTIVITY_TYPES = {
    ActivityType.GO_TO.value: {
        'subtypes': ['walking'],
        'energy_cost': ENERGY_COST_PER_HOUR_TRAVEL,
        'memory_type': 'TRAVEL'    
        },
    ActivityType.WORK.value: {
        'subtypes': ['office_work', 'manual_labor', 'customer_service'],
        'energy_cost': ENERGY_COST_WORK_HOUR,
        'min_energy': ENERGY_THRESHOLD_LOW,
        'total_hourly_cost': ENERGY_COST_WORK_HOUR + ENERGY_DECAY_PER_HOUR
    },
    ActivityType.REST.value: {
        'subtypes': ['sleeping', 'napping', 'relaxing'],
        'energy_gain': ENERGY_GAIN_NAP,
        'nap_hours': [11, 12, 13, 14, 15],
        'max_nap_duration': 1,
        'sleep_hours': [22, 23, 0, 1, 2, 3, 4, 5, 6]
    },
    ActivityType.EAT.value: {
        'subtypes': ['breakfast', 'lunch', 'dinner', 'snack'],
        'energy_gain': {
            'breakfast': ENERGY_GAIN_HOME_MEAL,
            'lunch': ENERGY_GAIN_RESTAURANT_MEAL,
            'dinner': ENERGY_GAIN_RESTAURANT_MEAL,
            'snack': ENERGY_GAIN_SNACK
        }
    },
    ActivityType.SHOP.value: {
        'subtypes': ['grocery', 'retail', 'window_shopping']
    },
    ActivityType.SOCIAL.value: {
        'subtypes': ['conversation', 'group_activity', 'meeting']
    },
    ActivityType.PLAN.value: {
        'subtypes': ['daily_planning', 'scheduling', 'decision_making']
    },
    ActivityType.IDLE.value: {
        'subtypes': ['waiting', 'thinking', 'observing'],
        'energy_cost': 0,
        'memory_type': 'ACTIVITY'
    }
}

# Memory Types Configuration
MEMORY_TYPES = {
    # State and Info Types
    MemoryType.STATE_UPDATE.value: {
        'type': 'agent_state',
        'activity_type': None,
        'description': 'Agent state update event'
    },
    MemoryType.AGENT_INFO.value: {
        'type': 'agent_info',
        'activity_type': None,
        'description': 'Agent basic information'
    },
    MemoryType.RELATIONSHIP_INFO.value: {
        'type': 'relationship_info',
        'activity_type': None,
        'description': 'Agent relationship information'
    },

    # Activity Types
    MemoryType.ACTIVITY.value: {
        'type': 'activity',
        'activity_type': None,
        'description': 'General activity event'
    },
    MemoryType.TRAVEL.value: {
        'type': MemoryType.TRAVEL.value,
        'activity_type': ActivityType.GO_TO.value,
        'description': 'Travel event'
    },
    MemoryType.WORK.value: {
        'type': MemoryType.WORK.value,
        'activity_type': ActivityType.WORK.value,
        'description': 'Work event'
    },
    MemoryType.MEAL.value: {
        'type': MemoryType.MEAL.value,
        'activity_type': ActivityType.EAT.value,
        'description': 'Meal event'
    },
    MemoryType.SHOPPING.value: {
        'type': MemoryType.SHOPPING.value,
        'activity_type': ActivityType.SHOP.value,
        'description': 'Shopping event'
    },
    MemoryType.REST.value: {
        'type': MemoryType.REST.value,
        'activity_type': ActivityType.REST.value,
        'description': 'Rest event'
    },
    MemoryType.SOCIAL.value: {
        'type': MemoryType.SOCIAL.value,
        'activity_type': ActivityType.SOCIAL.value,
        'description': 'Social event'
    },
    MemoryType.DINING.value: {
        'type': MemoryType.DINING.value,
        'activity_type': ActivityType.EAT.value,
        'description': 'Dining event'
    },
    MemoryType.IDLE.value: {
        'type': MemoryType.IDLE.value,
        'activity_type': ActivityType.IDLE.value,
        'description': 'Idle event'
    },

    # Planning Types
    MemoryType.PLAN.value: {
        'type': MemoryType.PLAN.value,
        'activity_type': ActivityType.PLAN.value,
        'description': 'Plan event'
    },
    MemoryType.PLAN_CREATION.value: {
        'type': MemoryType.PLAN_CREATION.value,
        'activity_type': ActivityType.PLAN.value,
        'description': 'Plan creation event'
    },
    MemoryType.PLAN_UPDATE.value: {
        'type': MemoryType.PLAN_UPDATE.value,
        'activity_type': ActivityType.PLAN.value,
        'description': 'Plan update event'
    },
    MemoryType.PLANNING.value: {
        'type': MemoryType.PLANNING.value,
        'activity_type': ActivityType.PLAN.value,
        'description': 'Planning event'
    },

    # Interaction Types
    MemoryType.CONVERSATION.value: {
        'type': MemoryType.CONVERSATION.value,
        'activity_type': ActivityType.SOCIAL.value,
        'description': 'Conversation event'
    },
    MemoryType.ENCOUNTER.value: {
        'type': MemoryType.ENCOUNTER.value,
        'activity_type': ActivityType.SOCIAL.value,
        'description': 'Encounter event'
    },
    MemoryType.RELATIONSHIP.value: {
        'type': MemoryType.RELATIONSHIP.value,
        'activity_type': None,
        'description': 'Relationship event'
    },
    MemoryType.COMMITMENT.value: {
        'type': MemoryType.COMMITMENT.value,
        'activity_type': None,
        'description': 'Commitment event'
    },

    # Transaction Types
    MemoryType.PURCHASE.value: {
        'type': MemoryType.PURCHASE.value,
        'activity_type': ActivityType.SHOP.value,
        'description': 'Purchase event'
    },
    MemoryType.EARNING.value: {
        'type': MemoryType.EARNING.value,
        'activity_type': ActivityType.WORK.value,
        'description': 'Earning event'
    },

    # Meta/Failure Types
    MemoryType.FAILED_ACTION.value: {
        'type': MemoryType.FAILED_ACTION.value,
        'activity_type': None,
        'description': 'Event for a failed planned action'
    }
}

# Memory Event class
class MemoryEvent:
    def __init__(self, memory_type: str, data: Dict[str, Any], simulation_time: int, version: int = 1):
        self.memory_type = memory_type
        self.data = data
        self.simulation_time = simulation_time
        self.version = version 

class EnergySystem(ThreadSafeBase):
    """Manages energy-related functionality for agents."""
    def __init__(self):
        super().__init__()
        self._energy_levels: Dict[str, int] = {}
        self._energy_history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def set_energy(self, agent_name: str, energy_level: int) -> Result[None]:
        """Set an agent's energy level and record the change in history."""
        try:
            with self._lock:
                # Ensure energy level is within bounds
                energy_level = max(ENERGY_MIN, min(ENERGY_MAX, energy_level))
                self._energy_levels[agent_name] = energy_level
                
                # Record the change in history
                if agent_name not in self._energy_history:
                    self._energy_history[agent_name] = []
                self._energy_history[agent_name].append({
                    'timestamp': time.time(),
                    'change': energy_level - self._energy_levels.get(agent_name, ENERGY_MAX),
                    'new_level': energy_level
                })
                
                return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error setting energy for {agent_name}: {str(e)}")

    def update_energy(self, agent_name: str, amount: int) -> Result[None]:
        """Update an agent's energy level and record the change in history."""
        try:
            with self._lock:
                current_energy = self._energy_levels.get(agent_name, ENERGY_MAX)
                new_energy = max(ENERGY_MIN, min(ENERGY_MAX, current_energy + amount))
                self._energy_levels[agent_name] = new_energy
                
                # Record the change in history
                if agent_name not in self._energy_history:
                    self._energy_history[agent_name] = []
                self._energy_history[agent_name].append({
                    'timestamp': time.time(),
                    'change': amount,
                    'new_level': new_energy
                })
                
                return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error updating energy for {agent_name}: {str(e)}")

    def get_energy_level(self, agent_name: str) -> Result[int]:
        """Get an agent's current energy level."""
        try:
            with self._lock:
                return Result.success(self._energy_levels.get(agent_name, ENERGY_MAX))
        except Exception as e:
            return Result.failure(f"Error getting energy level for {agent_name}: {str(e)}", fallback=ENERGY_MAX)

    def can_perform_action(self, agent_name: str, action_cost: int) -> Result[bool]:
        """Check if an agent has enough energy to perform an action."""
        try:
            with self._lock:
                current_energy = self._energy_levels.get(agent_name, ENERGY_MAX)
                return Result.success(current_energy >= action_cost)
        except Exception as e:
            return Result.failure(f"Error checking energy for action: {str(e)}", fallback=False)

    def calculate_energy_cost(self, action_type: str, steps: int = 1) -> Result[int]:
        """Calculate energy cost for various actions.
        
        Note: Natural decay (ENERGY_DECAY_PER_HOUR) is handled separately 
        in the main simulation loop every hour.
        
        Only work and travel cost energy:
        - Work: ENERGY_COST_WORK_HOUR per hour
        - Travel: ENERGY_COST_PER_STEP per step
        
        Other actions (conversation, shopping, idle, eat, rest) don't cost energy.
        """
        try:
            # Handle travel
            if action_type == 'travel':
                base_cost = ENERGY_COST_PER_STEP * steps
                return Result.success(base_cost)
            
            # Handle work
            elif action_type == 'work':
                base_cost = ENERGY_COST_WORK_HOUR * steps
                return Result.success(base_cost)
            
            # All other actions don't cost energy
            else:
                return Result.success(0)
            
        except Exception as e:
            return Result.failure(f"Error calculating energy cost: {str(e)}", fallback=0)

    def calculate_energy_gain(self, action_type: str, location: 'Location' = None, time: int = None, agent_residence: str = None, agent_workplace: str = None) -> Result[int]:
        """Calculate energy gain from various actions.
        
        Args:
            action_type: Type of action ('conversation', 'eat', 'rest')
            location: Location object where action occurs
            time: Current hour (0-23) for time-based restrictions
            agent_residence: The name of the agent's residence.
            agent_workplace: The name of the agent's workplace.
        """
        try:
            if action_type == 'eat':
                # Check for eating at agent's own residence
                if agent_residence and location and location.name == agent_residence:
                    return Result.success(ENERGY_GAIN_HOME_MEAL)
                
                # Check for eating at a restaurant
                if location and location.location_type in ['restaurant', 'local_shop']:
                    # Check if it's meal time for restaurants
                    if time is not None and get_meal_period(time) in ['breakfast', 'lunch', 'dinner']:
                        return Result.success(ENERGY_GAIN_RESTAURANT_MEAL)
                    else:  # Not meal time - treat as snack
                        return Result.success(ENERGY_GAIN_SNACK)
                
                # Default to snack if it's not a home meal or restaurant meal
                return Result.success(ENERGY_GAIN_SNACK)
                    
            elif action_type == 'rest':
                # Napping at the agent's specific workplace
                if agent_workplace and location and location.name == agent_workplace and time is not None and 11 <= time <= 15:
                    return Result.success(ENERGY_GAIN_NAP)
                # Sleeping at the agent's specific home
                elif agent_residence and location and location.name == agent_residence and time is not None and (23 <= time or time <= 6):
                    return Result.success(ENERGY_GAIN_SLEEP)
                else:
                    return Result.failure(f"Invalid rest conditions: location={location.name if location else None}, time={time}, home={agent_residence}, workplace={agent_workplace}")
                    
            return Result.success(0)
            
        except Exception as e:
            return Result.failure(f"Error calculating energy gain: {str(e)}", fallback=0)

    def get_energy_history(self, agent_name: str) -> Result[List[Dict[str, Any]]]:
        """Get energy history for an agent."""
        try:
            with self._lock:
                return Result.success(self._energy_history.get(agent_name, []))
        except Exception as e:
            return Result.failure(f"Error getting energy history: {str(e)}", fallback=[])

    def reset_energy(self, agent_name: str) -> Result[None]:
        """Reset an agent's energy to maximum."""
        try:
            with self._lock:
                self._energy_levels[agent_name] = ENERGY_MAX
                return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error resetting energy: {str(e)}")

    def clear_history(self, agent_name: Optional[str] = None) -> Result[None]:
        """Clear energy history for an agent or all agents."""
        try:
            with self._lock:
                if agent_name:
                    self._energy_history[agent_name] = []
                else:
                    self._energy_history.clear()
                return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error clearing energy history: {str(e)}")

    def get_energy(self, agent_name: str) -> int:
        """Get current energy level for an agent."""
        with self._lock:
            return self._energy_levels.get(agent_name, ENERGY_MAX)

class GrocerySystem(ThreadSafeBase):
    """Manages grocery-related functionality for agents."""
    def __init__(self):
        super().__init__()
        self._grocery_levels: Dict[str, int] = {}
        self._grocery_history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def set_grocery_level(self, agent_name: str, level: int) -> Result[None]:
        """Set an agent's grocery level and record the change in history."""
        try:
            with self._lock:
                # Ensure grocery level is within bounds
                level = max(GROCERY_MIN, min(GROCERY_MAX, level))
                self._grocery_levels[agent_name] = level
                
                # Record the change in history
                if agent_name not in self._grocery_history:
                    self._grocery_history[agent_name] = []
                self._grocery_history[agent_name].append({
                    'timestamp': time.time(),
                    'change': level - self._grocery_levels.get(agent_name, 0),
                    'new_level': level
                })
                
                return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error setting grocery level for {agent_name}: {str(e)}")

    def update_grocery_level(self, agent_name: str, amount: int) -> Result[None]:
        """Update an agent's grocery level and record the change in history."""
        try:
            with self._lock:
                current_level = self._grocery_levels.get(agent_name, GROCERY_MAX)
                new_level = max(GROCERY_MIN, min(GROCERY_MAX, current_level + amount))
                self._grocery_levels[agent_name] = new_level
                
                # Record the change in history
                if agent_name not in self._grocery_history:
                    self._grocery_history[agent_name] = []
                self._grocery_history[agent_name].append({
                    'timestamp': time.time(),
                    'change': amount,
                    'new_level': new_level
                })
                
                return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error updating grocery level for {agent_name}: {str(e)}")

    def get_grocery_level(self, agent_name: str) -> Result[int]:
        """Get an agent's current grocery level."""
        try:
            with self._lock:
                return Result.success(self._grocery_levels.get(agent_name, GROCERY_MAX))
        except Exception as e:
            return Result.failure(f"Error getting grocery level for {agent_name}: {str(e)}", fallback=GROCERY_MAX)

    def needs_groceries(self, agent_name: str, threshold: int = GROCERY_THRESHOLD_LOW) -> Result[bool]:
        """Check if an agent needs groceries based on a threshold."""
        try:
            with self._lock:
                current_level = self._grocery_levels.get(agent_name, GROCERY_MAX)
                return Result.success(current_level < threshold)
        except Exception as e:
            return Result.failure(f"Error checking grocery needs: {str(e)}", fallback=True)

    def get_grocery_history(self, agent_name: str) -> Result[List[Dict[str, Any]]]:
        """Get grocery history for an agent."""
        try:
            with self._lock:
                return Result.success(self._grocery_history.get(agent_name, []))
        except Exception as e:
            return Result.failure(f"Error getting grocery history: {str(e)}", fallback=[])

    def reset(self, agent_name: str) -> Result[None]:
        """Reset an agent's grocery level to default."""
        try:
            with self._lock:
                self._grocery_levels[agent_name] = GROCERY_MAX
                return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error resetting grocery level: {str(e)}")

    def clear_history(self, agent_name: Optional[str] = None) -> Result[None]:
        """Clear grocery history for an agent or all agents."""
        try:
            with self._lock:
                if agent_name:
                    self._grocery_history[agent_name] = []
                else:
                    self._grocery_history.clear()
                return Result.success(None)
        except Exception as e:
            return Result.failure(f"Error clearing grocery history: {str(e)}")

    def get_level(self, agent_name: str) -> int:
        """Get the grocery level for an agent."""
        with self._lock:
            return self._grocery_levels.get(agent_name, GROCERY_MAX)

    def get_grocery_cost_for_meal(self) -> int:
        """Get the standard grocery cost for a home meal."""
        return GROCERY_COST_HOME_MEAL

class PromptManagerInterface(Protocol):
    """Interface for prompt management functionality."""
    def get_prompt(self, prompt_type: str, context: dict) -> str:
        """Get a formatted prompt based on type and context."""
        ...

    def get_prompt_type(self, prompt: str) -> str:
        """Determine the type of a given prompt."""
        ...

    def get_discount_info(self, location_name: str, current_day: int) -> str:
        """Get discount information for a location."""
        ...

    def get_location_context(self, location_name: str, current_day: int) -> str:
        """Get context information for a location."""
        ...

    def validate_context(self, prompt_type: str, context: dict) -> bool:
        """Validate context for a prompt type."""
        ...

    def get_planning_prompt(self, agent_name: str, current_time: int, context: dict) -> str:
        """Get a planning prompt for an agent."""
        ...

    def get_plan_schema(self) -> dict:
        """Get the schema for planning prompts."""
        ...

    def get_conversation_prompt(self, speaker: str, listener: str, context: dict) -> str:
        """Get a conversation prompt between two agents."""
        ...

class MemoryManagerInterface(Protocol):
    """Interface for memory management functionality."""
    def record_memory(self, agent_name: str, memory_type: str, content: Dict[str, Any], timestamp: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> Result[None]:
        """Record a memory with a sequential ID and proper categorization."""
        ...

    def get_memories(self, agent_name: Optional[str] = None, memory_type: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get memories with optional filtering by agent, type, or category."""
        ...

    def get_recent_memories(self, agent_name: str, memory_type: str = None, limit: int = 10) -> List[Dict]:
        """Get recent memories for an agent."""
        ...

    def get_shared_memories(self, agent1: str, agent2: str, memory_type: str = None, limit: int = 10) -> List[Dict]:
        """Get memories shared between two agents."""
        ...

    def save_memories(self) -> Result[None]:
        """Save all memories to file."""
        ...

    def force_save_memories(self) -> Result[None]:
        """Force save memories regardless of buffer state."""
        ...

    def clear_memories(self) -> None:
        """Clear all memories."""
        ...

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of all memories."""
        ...

    def record_state_update(self, agent_name: str, state_data: Dict[str, Any]) -> None:
        """Record a state update for an agent."""
        ...

    def get_agent_state_history(self, agent_name: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict]:
        """Get state history for an agent."""
        ... 


# Agent defaults
DEFAULT_AGENT_ENERGY = ENERGY_MAX
DEFAULT_AGENT_MONEY_MULTIPLIER = 3  # 3 days of wages

# Memory management constants
MEMORY_SAVE_INTERVAL = 3600  # Save memories every hour (unified with other data types)
MEMORY_CLEANUP_INTERVAL = 3600  # Cleanup old memories every hour
DEFAULT_MEMORY_BUFFER_SIZE = 200  # Maximum number of memories to keep in buffer (reduced from 450)

# Metrics saving constants (unified with memories)
METRICS_SAVE_INTERVAL = 3600  # Save metrics every hour (unified with memories)

# Conversation saving constants (unified with memories)
CONVERSATION_BUFFER_SIZE = 50  # Save conversations when buffer reaches 50 items
CONVERSATION_SAVE_INTERVAL = 3600  # Save conversations every hour (unified with memories)

class TimeManager(ThreadSafeBase):
    """Manages time-related operations in the simulation."""
    _instance = None
    _current_day = 1  # Current day (1-7)
    _current_hour = 7  # Current hour (0-24)
    _total_hours = 161  # Total simulation hours (17h Day 1 + 24h×5 Days 2-6 + 24h Day 7)
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimeManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_current_time(cls) -> int:
        """Get current simulation time in hours."""
        with cls._lock:
            return (cls._current_day - 1) * 24 + cls._current_hour
    
    @classmethod
    def get_current_hour(cls) -> int:
        """Get current hour of the day."""
        with cls._lock:
            return cls._current_hour
    
    @classmethod
    def get_current_day(cls) -> int:
        """Get current day of the simulation."""
        with cls._lock:
            return cls._current_day
    
    @classmethod
    def get_total_hours_passed(cls) -> int:
        """Get total hours passed in the simulation."""
        with cls._lock:
            if cls._current_day == 1:
                return cls._current_hour - 7  # Day 1 starts at hour 7
            else:
                return (cls._current_day - 1) * 24 + (cls._current_hour - 7)
    
    @classmethod
    def is_simulation_end(cls) -> bool:
        """Check if simulation has reached its end."""
        with cls._lock:
            return cls.get_total_hours_passed() >= cls._total_hours
    
    @classmethod
    def advance_time(cls, hours: int = 1) -> None:
        """Advance simulation time by specified hours."""
        with cls._lock:
            print(f"[DEBUG] advance_time() called with hours={hours}")
            print(f"[DEBUG] Before advance: Day {cls._current_day}, Hour {cls._current_hour}")
            
            # Calculate new time without calling get_total_hours_passed() to avoid deadlock
            new_hour = cls._current_hour + hours
            new_day = cls._current_day
            
            # Handle hour overflow - allow natural day transitions
            while new_hour >= 24:
                new_hour -= 24
                new_day += 1
            
            # Update time
            cls._current_hour = new_hour
            cls._current_day = new_day
            
            # Allow the simulation to naturally reach Day 8, Hour 0 for end condition
            # No artificial bounds checking - let the simulation end condition handle it
                
            print(f"[DEBUG] After advance: Day {cls._current_day}, Hour {cls._current_hour}")
    
    @classmethod
    def set_time(cls, hour: int) -> None:
        """Set the current hour of the day."""
        with cls._lock:
            if 0 <= hour < 24:
                cls._current_hour = hour
    
    @classmethod
    def set_current_day(cls, day: int) -> None:
        """Set the current day of the simulation."""
        with cls._lock:
            if 1 <= day <= 8:  # Allow Day 8 for proper simulation end detection
                cls._current_day = day
    
    @classmethod
    def reset_time(cls) -> None:
        """Reset time to initial values."""
        with cls._lock:
            cls._current_day = 1
            cls._current_hour = 7
    
    @classmethod
    def is_sleep_time(cls) -> bool:
        """Check if current time is sleep time."""
        with cls._lock:
            return cls._current_hour in [22, 23, 0, 1, 2, 3, 4, 5, 6]
    
    @classmethod
    def is_nap_time(cls) -> bool:
        """Check if current time is nap time."""
        with cls._lock:
            return cls._current_hour in [11, 12, 13, 14, 15]
    
    @classmethod
    def is_meal_time(cls) -> Tuple[bool, str]:
        """Check if current time is meal time and return meal type."""
        with cls._lock:
            hour = cls._current_hour
            if 6 <= hour <= 9:
                return True, "breakfast"
            elif 11 <= hour <= 14:
                return True, "lunch"
            elif 17 <= hour <= 20:
                return True, "dinner"
            return False, "" 

def get_meal_period(current_hour: int) -> str:
    """Determine current meal period based on hour."""
    if 6 <= current_hour <= 9:
        return 'breakfast'
    elif 11 <= current_hour <= 14:
        return 'lunch'
    elif 17 <= current_hour <= 20:
        return 'dinner'
    else:
        return 'snack' 