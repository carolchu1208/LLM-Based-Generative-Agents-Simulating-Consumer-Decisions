# Standard library imports
import threading
import time as time_module  # Rename to avoid conflict with datetime.time
import json
import os
from datetime import datetime
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# Simulation settings
SIMULATION_SETTINGS: Dict[str, Any] = {
    'simulation': {
        'duration_days': 7,      # Duration of the simulation in days
        'time_step': 60,         # Time step in minutes
        'output_frequency': 24,  # How often to output summaries
        'parallel': True         # Run agents in parallel
    },
    'agent': {
        'max_memory_size': 1000, # Maximum number of memories
        'interaction_range': 5    # Range for agent interactions
    },
    'logging': {
        'log_level': 'INFO',     # Logging level
        'log_to_file': True      # Whether to log to file
    },
    'day_start_hour': 7, # Simulation day processing starts at 7 AM
    'day_end_hour': 24,  # Simulation day processing ends at midnight (hour 23 is last active hour)
    'planning_hour': 7   # Hour at which daily plans are made
}

# Activity types
ACTIVITY_TYPES: Dict[str, str] = {
    'TRAVEL': 'travel',
    'WORK': 'work',
    'RESTING': 'resting',
    'GROCERY_PURCHASE': 'grocery_purchase',
    'FOOD_PURCHASE': 'food_purchase',
    'DINING': 'dining',
    'EDUCATION': 'education',
    'RECREATION': 'recreation',
    'CONVERSATION': 'conversation',
    'QUICK_STOP': 'quick_stop'
}

# Memory types
MEMORY_TYPES: Dict[str, str] = {
    # Raw actions and system events
    'ACTION_RAW_OUTPUT': 'action_raw_output',  # Raw agent actions
    'SYSTEM_EVENT': 'system_event',            # System-level events
    
    # State updates (including resources)
    'AGENT_STATE_UPDATE_EVENT': 'agent_state_update_event',  # All state changes (energy, income, etc.)
    
    # Activities and travel
    'ACTIVITY_EVENT': 'activity_event',        # All activities (work, rest, quick stop, etc.)
    'TRAVEL_EVENT': 'travel_event',            # Travel-related events
    
    # Planning
    'PLANNING_EVENT': 'planning_event',        # Planning activities
    
    # Location
    'LOCATION_EVENT': 'location_event',        # All location-related events (changes, visits, etc.)
    
    # Social interactions
    'CONVERSATION_EVENT': 'conversation_event',  # All conversation-related events (including logs)
    'INTERACTION_EVENT': 'interaction_event',    # All social interactions (including household)
    
    # Purchases
    'PURCHASE_EVENT': 'purchase_event',         # All purchase events (food, groceries)
    
    # Emotional and satisfaction
    'EMOTIONAL_EVENT': 'emotional_event'        # All emotional states and satisfaction ratings
}

# Energy system constants
ENERGY_MAX: int = 100
ENERGY_MIN: int = 0
ENERGY_COST_PER_STEP: int = 1
ENERGY_DECAY_PER_HOUR: int = 10
ENERGY_COST_WORK_HOUR: int = 15
ENERGY_COST_PER_HOUR_TRAVEL: int = 5
ENERGY_COST_PER_HOUR_IDLE: int = 1
ENERGY_GAIN_RESTAURANT_MEAL: int = 40
ENERGY_GAIN_SNACK: int = 5
ENERGY_GAIN_HOME_MEAL: int = 20
ENERGY_GAIN_SLEEP: int = 50
ENERGY_GAIN_NAP: int = 10
ENERGY_GAIN_CONVERSATION: int = 5
ENERGY_THRESHOLD_LOW: int = 20  # Threshold below which agent should prioritize getting food

class MemoryEvent:
    """Base class for memory events with validation and serialization."""
    
    def __init__(self, memory_type: str, data: Dict[str, Any]) -> None:
        """Initialize a memory event.
        
        Args:
            memory_type: The type of memory event
            data: The data associated with the event
            
        Raises:
            ValueError: If memory_type is not in MEMORY_TYPES
        """
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Invalid memory type: {memory_type}")
        
        self.memory_type = memory_type
        self.data = data
        self.timestamp = time_module.time()  # Use renamed time module
        self.version = 1
        self.content_hash = self._generate_hash()
        self.affected_agents = self._determine_affected_agents()
    
    def _determine_affected_agents(self) -> List[str]:
        """Determine which agents should receive this memory event."""
        affected = set()
        
        # Always include the agent who created the memory
        if 'agent_name' in self.data:
            affected.add(self.data['agent_name'])
        
        # Add participants for conversation and interaction events
        if self.memory_type in ['CONVERSATION_LOG_EVENT', 'INTERACTION_EVENT']:
            participants = self.data.get('participants', [])
            affected.update(participants)
        
        # Add household members for household events
        elif self.memory_type == 'HOUSEHOLD_EVENT':
            household_members = self.data.get('household_members', [])
            affected.update(household_members)
        
        # Add relationship participants for relationship events
        elif self.memory_type == 'RELATIONSHIP_EVENT':
            participants = self.data.get('participants', [])
            affected.update(participants)
        
        # Add nearby agents for location events
        elif self.memory_type == 'LOCATION_EVENT':
            nearby_agents = self.data.get('nearby_agents', [])
            affected.update(nearby_agents)
        
        return list(affected)
    
    def _generate_hash(self) -> str:
        """Generate a hash of the memory content."""
        try:
            content_str = json.dumps({
                'type': self.memory_type,
                'time': self.data.get('time', 0),
                'location': self.data.get('location', ''),
                'participants': sorted(self.data.get('participants', [])),
                'content': self.data.get('content', '')
            }, sort_keys=True)
            return hashlib.md5(content_str.encode()).hexdigest()
        except Exception as e:
            print(f"Error generating memory hash: {str(e)}")
            return ''

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory event to dictionary format."""
        return {
            'type': self.memory_type,
            'data': self.data,
            'timestamp': self.timestamp,
            'version': self.version,
            'content_hash': self.content_hash,
            'affected_agents': self.affected_agents
        }

class TimeManager:
    """Utility class for handling time-related operations in the simulation.
    
    This class provides static methods for managing simulation time, including
    converting between simulation time and real-world time, checking meal times,
    and formatting time strings.
    """
    
    @staticmethod
    def get_hour_from_time(sim_time: int) -> int:
        """Convert simulation time to hour of day (0-23).
        
        Args:
            sim_time: The simulation time in hours
            
        Returns:
            The hour of day (0-23)
        """
        return sim_time % 24
    
    @staticmethod
    def get_day_from_time(sim_time: int) -> int:
        """Convert simulation time to day number (1-based)."""
        return (sim_time // 24) + 1
    
    @staticmethod
    def is_meal_time(sim_time: int) -> Tuple[bool, Optional[str]]:
        """Check if current time is a meal time.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_meal_time, meal_type)
        """
        hour = TimeManager.get_hour_from_time(sim_time)
        
        if 7 <= hour < 10:
            return True, "breakfast"
        elif 11 <= hour < 14:
            return True, "lunch"
        elif 17 <= hour < 20:
            return True, "dinner"
        elif 14 <= hour < 17 or 20 <= hour < 22:
            return True, "snack"
        return False, None
    
    @staticmethod
    def is_sleep_time(sim_time: int) -> bool:
        """Check if current time is sleep time (22:00-06:00)."""
        hour = TimeManager.get_hour_from_time(sim_time)
        return hour >= 22 or hour < 6
    
    @staticmethod
    def format_time(sim_time: int) -> str:
        """Format simulation time as HH:MM."""
        hour = TimeManager.get_hour_from_time(sim_time)
        return f"{hour:02d}:00"
    
    @staticmethod
    def format_datetime(sim_time: int) -> str:
        """Format simulation time as Day X, HH:MM."""
        day = TimeManager.get_day_from_time(sim_time)
        hour = TimeManager.get_hour_from_time(sim_time)
        return f"Day {day}, {hour:02d}:00"
    
    @staticmethod
    def get_time_until_next_meal(sim_time: int) -> Tuple[int, Optional[str]]:
        """Calculate hours until next meal time.
        
        Returns:
            Tuple[int, Optional[str]]: (hours_until_next_meal, next_meal_type)
        """
        hour = TimeManager.get_hour_from_time(sim_time)
        
        if hour < 7:
            return 7 - hour, "breakfast"
        elif hour < 11:
            return 11 - hour, "lunch"
        elif hour < 17:
            return 17 - hour, "dinner"
        elif hour < 22:
            return 22 - hour, "snack"
        else:
            return 31 - hour, "breakfast"  # Next day's breakfast
    
    @staticmethod
    def get_time_until_work(sim_time: int, work_schedule: Optional[dict] = None) -> int:
        """Calculate hours until next work time."""
        hour = TimeManager.get_hour_from_time(sim_time)
        work_start = work_schedule.get('start', 9) if work_schedule else 9
        
        if hour < work_start:
            return work_start - hour
        else:
            return 24 - hour + work_start  # Next day's work start 

class SimulationError(Exception):
    """Base class for simulation-specific exceptions."""
    pass

class AgentError(SimulationError):
    """Exception raised for agent-related errors."""
    pass

class LocationError(SimulationError):
    """Exception raised for location-related errors."""
    pass

class MemoryError(SimulationError):
    """Exception raised for memory-related errors."""
    pass

class MetricsError(SimulationError):
    """Exception raised for metrics-related errors."""
    pass

class ErrorHandler:
    """Utility class for standardized error handling."""
    
    @staticmethod
    def handle_error(error: Exception, context: str, raise_error: bool = True) -> None:
        """Handle an error with standardized logging and optional re-raising.
        
        Args:
            error: The exception to handle
            context: Description of where the error occurred
            raise_error: Whether to re-raise the error after handling
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        print(f"ERROR in {context}:")
        print(f"Type: {error_type}")
        print(f"Message: {error_msg}")
        
        import traceback
        traceback.print_exc()
        
        if raise_error:
            raise error
    
    @staticmethod
    def wrap_with_error_handling(func):
        """Decorator to wrap functions with standardized error handling."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(e, f"{func.__name__} in {func.__module__}")
                return None
        return wrapper
    
    @staticmethod
    def validate_condition(condition: bool, error_msg: str, error_type: type = SimulationError) -> None:
        """Validate a condition and raise an appropriate error if it fails.
        
        Args:
            condition: The condition to validate
            error_msg: Message to use if condition is False
            error_type: Type of exception to raise
        """
        if not condition:
            raise error_type(error_msg)
    
    @staticmethod
    def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get an attribute from an object.
        
        Args:
            obj: The object to get the attribute from
            attr: The attribute name
            default: Value to return if attribute doesn't exist
            
        Returns:
            The attribute value or default if not found
        """
        try:
            return getattr(obj, attr, default)
        except Exception as e:
            ErrorHandler.handle_error(e, f"Error getting attribute {attr}", raise_error=False)
            return default 

class ThreadSafeBase:
    """Base class for thread-safe operations."""
    
    def __init__(self) -> None:
        """Initialize thread-safe base with a lock."""
        self._lock = threading.Lock()
    
    def _thread_safe_operation(self, operation: callable, *args: Any, **kwargs: Any) -> Any:
        """Execute an operation with thread safety.
        
        Args:
            operation: The function to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            The result of the operation
        """
        with self._lock:
            return operation(*args, **kwargs)
    
    def _thread_safe_get(self, attr: str, default: Any = None) -> Any:
        """Thread-safe getter for an attribute.
        
        Args:
            attr: The attribute name to get
            default: Default value if attribute doesn't exist
            
        Returns:
            The attribute value or default
        """
        with self._lock:
            return getattr(self, attr, default)
    
    def _thread_safe_set(self, attr: str, value: Any) -> None:
        """Thread-safe setter for an attribute.
        
        Args:
            attr: The attribute name to set
            value: The value to set
        """
        with self._lock:
            setattr(self, attr, value)
    
    def _thread_safe_update(self, attr: str, update_func: callable) -> None:
        """Thread-safe update of an attribute using a function.
        
        Args:
            attr: The attribute name to update
            update_func: Function to update the attribute value
        """
        with self._lock:
            current_value = getattr(self, attr)
            new_value = update_func(current_value)
            setattr(self, attr, new_value)
    
    def _thread_safe_append(self, attr: str, value: Any) -> None:
        """Thread-safe append to a list attribute.
        
        Args:
            attr: The list attribute name
            value: Value to append
        """
        with self._lock:
            current_list = getattr(self, attr, [])
            current_list.append(value)
            setattr(self, attr, current_list)
    
    def _thread_safe_remove(self, attr: str, value: Any) -> None:
        """Thread-safe remove from a list attribute.
        
        Args:
            attr: The list attribute name
            value: Value to remove
        """
        with self._lock:
            current_list = getattr(self, attr, [])
            if value in current_list:
                current_list.remove(value)
                setattr(self, attr, current_list)
    
    def _thread_safe_clear(self, attr: str) -> None:
        """Thread-safe clear a list attribute.
        
        Args:
            attr: The list attribute name
        """
        with self._lock:
            setattr(self, attr, []) 

class SharedMemoryBuffer(ThreadSafeBase):
    """Centralized memory management system."""
    
    _instance: Optional['SharedMemoryBuffer'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'SharedMemoryBuffer':
        """Create or return the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SharedMemoryBuffer, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the shared memory buffer."""
        self.buffer = []
        self.memory_types = MEMORY_TYPES
        self._write_lock = threading.Lock()
        self._memory_handlers = {
            'CONVERSATION_LOG_EVENT': self._handle_conversation_event,
            'AGENT_STATE_UPDATE_EVENT': self._handle_state_event,
            'PLANNING_EVENT': self._handle_planning_event,
            'ACTIVITY_EVENT': self._handle_activity_event,
            'TRAVEL_EVENT': self._handle_travel_event,
            'INTERACTION_EVENT': self._handle_interaction_event,
            'HOUSEHOLD_EVENT': self._handle_household_event,
            'RELATIONSHIP_EVENT': self._handle_relationship_event,
            'LOCATION_EVENT': self._handle_location_event,
            'EMOTIONAL_STATE_EVENT': self._handle_emotional_event
        }
        
        # Initialize memory manager reference
        self.memory_manager = None

    def set_memory_manager(self, memory_manager):
        """Set the memory manager instance for file operations."""
        self.memory_manager = memory_manager

    def push(self, memory_event: Union[MemoryEvent, Dict[str, Any]]) -> None:
        """Add a memory event to the buffer."""
        with self._write_lock:
            # Convert dict to MemoryEvent if necessary
            if isinstance(memory_event, dict):
                memory_event = MemoryEvent(memory_event['type'], memory_event['data'])

            # Validate memory type
            if memory_event.memory_type not in self.memory_types:
                print(f"Warning: Invalid memory type: {memory_event.memory_type}")
                return

            # Check for duplicates
            if not self._is_duplicate_memory(memory_event):
                self.buffer.append(memory_event)

    def flush_to_agents(self, agents: List['Agent']) -> None:
        """Distribute buffered memories to relevant agents based on affected_agents."""
        with self._write_lock:
            # Group memories by affected agents for efficient distribution
            agent_memories = defaultdict(list)
            
            for event in self.buffer:
                # Get affected agents for this event
                affected_agents = event.affected_agents
                
                # If no specific affected agents, use the handler to determine distribution
                if not affected_agents:
                    handler = self._memory_handlers.get(event.memory_type)
                    if handler:
                        handler(event, agents)
                else:
                    # Group memory by affected agents
                    for agent_name in affected_agents:
                        agent_memories[agent_name].append(event)
            
            # Distribute memories to each affected agent
            for agent_name, memories in agent_memories.items():
                agent = next((a for a in agents if a.name == agent_name), None)
                if agent:
                    for memory in memories:
                        agent.record_memory(memory.memory_type, memory.data)
            
            # Clear the buffer after distribution
            self.buffer.clear()

    def flush_to_files(self) -> None:
        """Save buffered memories to files using the memory manager."""
        if not self.memory_manager:
            print("Warning: Memory manager not set, cannot save to files")
            return

        try:
            # Group memories by type for efficient saving
            conversation_logs = []
            metrics_data = {}
            daily_summary = {}

            for event in self.buffer:
                if event.memory_type == 'CONVERSATION_LOG_EVENT':
                    conversation_logs.append({
                        'timestamp': event.timestamp,
                        'agent_name': event.data.get('agent_name'),
                        'type': event.memory_type,
                        'content': event.data.get('content')
                    })
                elif event.memory_type in ['AGENT_STATE_UPDATE_EVENT', 'ACTIVITY_EVENT']:
                    # Update metrics data
                    agent_name = event.data.get('agent_name')
                    if agent_name not in metrics_data:
                        metrics_data[agent_name] = []
                    metrics_data[agent_name].append(event.data)
                elif event.memory_type in ['PLANNING_EVENT', 'HOUSEHOLD_EVENT']:
                    # Update daily summary
                    if 'daily_summary' not in daily_summary:
                        daily_summary['daily_summary'] = []
                    daily_summary['daily_summary'].append(event.data)

            # Save to files
            for log in conversation_logs:
                self.memory_manager.save_conversation_log(
                    agent_name=log['agent_name'],
                    content=log['content'],
                    log_type=log['type']
                )

            if metrics_data:
                self.memory_manager.save_metrics(metrics_data)

            if daily_summary:
                self.memory_manager.save_daily_summary(daily_summary)

        except Exception as e:
            print(f"Error saving memories to files: {str(e)}")
            traceback.print_exc()

    def flush_all(self, agents: List['Agent']) -> None:
        """Flush memories to both agents and files."""
        self.flush_to_agents(agents)
        self.flush_to_files()

    def _handle_conversation_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle conversation memory events."""
        participants = event.data.get('participants', [])
        for name in participants:
            agent = next((a for a in agents if a.name == name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_state_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle agent state update events."""
        agent_name = event.data.get('agent_name')
        if agent_name:
            agent = next((a for a in agents if a.name == agent_name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_planning_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle planning events."""
        agent_name = event.data.get('agent_name')
        if agent_name:
            agent = next((a for a in agents if a.name == agent_name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_activity_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle activity events."""
        agent_name = event.data.get('agent_name')
        if agent_name:
            agent = next((a for a in agents if a.name == agent_name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_travel_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle travel events."""
        agent_name = event.data.get('agent_name')
        if agent_name:
            agent = next((a for a in agents if a.name == agent_name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_interaction_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle interaction events."""
        participants = event.data.get('participants', [])
        for name in participants:
            agent = next((a for a in agents if a.name == name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_household_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle household events."""
        household_members = event.data.get('household_members', [])
        for name in household_members:
            agent = next((a for a in agents if a.name == name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_relationship_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle relationship events."""
        participants = event.data.get('participants', [])
        for name in participants:
            agent = next((a for a in agents if a.name == name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_location_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle location events."""
        agent_name = event.data.get('agent_name')
        if agent_name:
            agent = next((a for a in agents if a.name == agent_name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _handle_emotional_event(self, event: MemoryEvent, agents: List['Agent']):
        """Handle emotional state events."""
        agent_name = event.data.get('agent_name')
        if agent_name:
            agent = next((a for a in agents if a.name == agent_name), None)
            if agent:
                agent.record_memory(event.memory_type, event.data)

    def _is_duplicate_memory(self, memory_event: MemoryEvent) -> bool:
        """Check if this memory is a duplicate of an existing one."""
        try:
            for memory in self.buffer:
                if memory.content_hash == memory_event.content_hash:
                    return True
            return False
        except Exception as e:
            print(f"Error checking for duplicate memory: {str(e)}")
            return False

    def get_recent_memories(self, agent_name: str, memory_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories for an agent with optional filtering."""
        with self._write_lock:
            # Get all memories for the agent
            memories = [m for m in self.buffer if m.data.get('agent_name') == agent_name]
            
            # Apply memory type filter if specified
            if memory_type:
                memories = [m for m in memories if m.memory_type == memory_type]
            
            # Sort by time (most recent first)
            memories.sort(key=lambda m: m.data.get('time', 0), reverse=True)
            
            # Take the most recent ones up to the limit
            recent_memories = memories[:limit]
            
            # Convert to dictionary format
            return [m.data for m in recent_memories]

# Create singleton instance
shared_memory_buffer = SharedMemoryBuffer() 