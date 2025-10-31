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
import re

# Local imports
from simulation_types import (
    MemoryType, MemoryEvent, ActivityType,
    ACTIVITY_TYPES, MEMORY_TYPES,
    TimeManager, EnergySystem, GrocerySystem,
    PromptManagerInterface, MemoryManagerInterface,
    DEFAULT_MEMORY_BUFFER_SIZE, MEMORY_SAVE_INTERVAL, MEMORY_CLEANUP_INTERVAL,
    DEFAULT_AGENT_MONEY_MULTIPLIER
)
from simulation_constants import (
    Result, SimulationError, AgentError, LocationError,
    MemoryError, MetricsError, ThreadSafeBase
)
from shared_trackers import (
    SharedLocationTracker, SharedResourceManager,
    LocationLockManager
)
from metrics_manager import StabilityMetricsManager

if TYPE_CHECKING:
    from simulation_execution_classes import Agent

DISCOUNT_KEYWORDS = [
    # Specific to our 20% discount
    "20% off",
    "20 percent off",
    "twenty percent off",
    "20% discount",
    "twenty percent discount",
    
    # Day-specific (since it's Wednesday and Thursday)
    "wednesday special",
    "thursday special",
    "midweek special",
    "wednesday discount",
    "thursday discount",
    
    # General discount terms that might be used
    "discount",
    "sale",
    "special offer",
    "promotion",
    
    # Price-related terms
    "cheaper",
    "savings",
    "reduced price",
    "save money",
    
    # Time-related
    "today only",
    "limited time"
]

class MemoryManager(ThreadSafeBase, MemoryManagerInterface):
    """Manages agent memories and their persistence.
    
    Note: This class uses its own locking system (self._lock) which is independent of
    the locking system in SharedTracker. This is intentional as they protect different
    resources:
    - MemoryManager's lock protects memory operations (adding, saving, loading memories)
    - SharedTracker's locks protect location and resource operations
    
    These systems don't conflict because they operate on different data structures
    and are never acquired simultaneously.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, simulation_timestamp: str = None):
        """Initialize the memory manager."""
        # Check if already initialized to prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.simulation_timestamp = simulation_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.agent_memories = {}
        self._pending_memories = []
        self._last_save_time = time.time()
        self._save_interval = MEMORY_SAVE_INTERVAL  # Use constant: 30 minutes

        # Hierarchical memory: agent daily reflections (Plan-Execution-Evaluation)
        self.agent_reflections = {}  # {agent_name: {day: reflection_dict}}

        # No longer store time locally - always use TimeManager
        # self._simulation_day = TimeManager.get_current_day()
        # self._simulation_time = TimeManager.get_current_hour()
        
        self._memory_counts = {}
        self._memory_categories = {
            'state': ['STATE_UPDATE', 'AGENT_INFO', 'RELATIONSHIP_INFO', 'EARNING'],
            'activity': ['ACTIVITY', 'TRAVEL', 'WORK', 'MEAL', 'SHOPPING', 'REST', 'SOCIAL', 'DINING', 'IDLE'],
            'planning': ['PLAN', 'PLAN_CREATION', 'PLAN_UPDATE', 'PLANNING'],
            'interaction': ['CONVERSATION', 'ENCOUNTER', 'RELATIONSHIP', 'COMMITMENT'],
            'transaction': ['PURCHASE']
        }
        
        # Initialize state history
        self.state_history = {}
        
        # Initialize thread safety locks
        self._lock = threading.RLock()  # Use RLock for reentrant locking
        self._plan_lock = threading.RLock()  # Separate lock for plan operations
        
        # Initialize file locks
        self.file_locks = {
            'memories': threading.RLock(),
            'conversations': threading.RLock(),
            'state': threading.RLock()
        }
        
        # Set up file paths
        self._file_directories()
        
        self._migrate_state_update_types()  # Add migration call
        
        # Initialize memory limits to prevent RAM growth
        self.MAX_MEMORIES_PER_AGENT = 500  # Keep only last 500 memories per agent
        self.MAX_TOTAL_MEMORIES = 2000     # Global limit across all agents
        
        # Mark as initialized
        self._initialized = True

        # Manager references (set later via set_managers)
        self.prompt_mgr = None
        self.model_mgr = None

        print(f"[DEBUG] Memory Manager initialized with timestamp: {self.simulation_timestamp}")
        print(f"[DEBUG] Initial time: Day {TimeManager.get_current_day()}, Hour {TimeManager.get_current_hour()}")
        print("[DEBUG] Locks initialized successfully")

    def set_prompt_manager(self, prompt_mgr):
        """Set the PromptManager reference for reflection prompt generation."""
        self.prompt_mgr = prompt_mgr
        print("[DEBUG] PromptManager reference set in MemoryManager")

    def set_model_manager(self, model_mgr):
        """Set the ModelManager reference for LLM calls."""
        self.model_mgr = model_mgr
        print("[DEBUG] ModelManager reference set in MemoryManager")

    def _file_directories(self):
        """Set up paths for memory storage files."""
        try:
            # Set up main directory path
            main_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'LLMAgentsTown_memory_records')
            
            # Set up subdirectory paths
            agents_dir = os.path.join(main_dir, 'simulation_agents')
            self.conversation_dir = os.path.join(main_dir, 'simulation_conversations')
            plans_dir = os.path.join(main_dir, 'simulation_plans')

            # Create directories if they don't exist
            os.makedirs(agents_dir, exist_ok=True)
            os.makedirs(self.conversation_dir, exist_ok=True)
            os.makedirs(plans_dir, exist_ok=True)
            
            # Set up file paths - one file per simulation, using consistent naming
            self.memory_file = os.path.join(agents_dir, f"agents_memories_{self.simulation_timestamp}.jsonl")
            self.conversation_file = os.path.join(self.conversation_dir, f"conversations_{self.simulation_timestamp}.jsonl")

            # Set up saved plans file path with simulation timestamp
            self.saved_plans_file = os.path.join(plans_dir, f"saved_plans_{self.simulation_timestamp}.json")

            print(f"Set up memory file paths:")
            print(f"- Agents memories: {self.memory_file}")
            print(f"- Conversations: {self.conversation_file}")
            print(f"- Saved plans: {self.saved_plans_file}")
            
        except Exception as e:
            print(f"Error setting up file paths: {str(e)}")
            traceback.print_exc()

    def save_memories(self, force: bool = False) -> None:
        """Save pending memories to a .jsonl file by appending them."""
        try:
            # Check if we should save
            if not self._pending_memories:
                return

            if not force and len(self._pending_memories) < DEFAULT_MEMORY_BUFFER_SIZE:
                return
                
            # Use a timeout for lock acquisition
            if not self._lock.acquire(timeout=30):  # 30 second timeout
                print("[WARNING] Could not acquire lock for saving memories")
                return
                
            try:
                # Append pending memories to the .jsonl file
                with self.file_locks['memories']:
                    with open(self.memory_file, 'a') as f:
                        for agent_name, memory_entry in self._pending_memories:
                            log_entry = {
                                'agent_name': agent_name,
                                **memory_entry
                            }
                            f.write(json.dumps(log_entry) + '\n')
                
                # Clear buffer after successful save
                self._pending_memories.clear()
                self._last_save_successful = True
                
            finally:
                self._lock.release()
                
        except Exception as e:
            print(f"Error saving memories: {str(e)}")
            traceback.print_exc()
            self._last_save_successful = False

    def _save_conversation(self, agent_name: str, conversation_data: Dict[str, Any]) -> None:
        """Save conversation data to the conversation .jsonl file by appending."""
        try:
            with self.file_locks['conversations']:
                # Extract timing from conversation_data if available
                simulation_day = conversation_data.get('simulation_day', TimeManager.get_current_day())
                simulation_hour = conversation_data.get('simulation_hour', TimeManager.get_current_hour())
                
                # Add new conversation with metadata using preserved timing
                conversation_entry = {
                    'agent_name': agent_name,
                    'conversation_data': conversation_data,
                    'simulation_day': simulation_day,
                    'simulation_time': simulation_hour
                }
                
                # Append new conversation as a single line to the .jsonl file
                with open(self.conversation_file, 'a') as f:
                    f.write(json.dumps(conversation_entry) + '\n')
                
        except Exception as e:
            print(f"Error saving conversation for {agent_name}: {str(e)}")
            traceback.print_exc()

    def add_memory(self, agent_name: str, memory_data: Dict[str, Any], memory_type: str = 'event', timestamp: Optional[int] = None) -> None:
        """Adds a memory to the agent's history and pending buffer, making it available for saving."""
        
        # Prepare memory entry
        memory_entry = {
            'agent_name': agent_name,
            'type': memory_type,
            'day': TimeManager.get_current_day(),
            'hour': TimeManager.get_current_hour(),
            'content': memory_data
        }
        
        # Add to in-memory store and pending buffer
        if agent_name not in self.agent_memories:
            self.agent_memories[agent_name] = []
        self.agent_memories[agent_name].append(memory_entry)
        self._pending_memories.append((agent_name, memory_entry))
        
        # Optional: Log the memory addition
        # print(f"Added memory for {agent_name}: {memory_type} at Day {self.current_simulation_day}, Hour {self.current_simulation_hour}")

    def _get_category_for_memory_type(self, memory_type: str) -> str:
        """Get the category for a memory type."""
        return self._memory_categories.get(memory_type, 'other')

    def save_pending_memories_and_cleanup(self, day: int, current_time: int, agent_states: List[Dict[str, Any]], metrics_manager: 'StabilityMetricsManager') -> None:
        """Save pending memories and perform cleanup operations. Called by main simulation."""
        try:
            with self._lock:
                # Record state updates for each agent
                for state in agent_states:
                    agent_name = state.get('agent_name')
                    if agent_name:
                        self._record_state_update_internal(agent_name, state)
                
                # Save memories if needed
                if len(self._pending_memories) >= DEFAULT_MEMORY_BUFFER_SIZE:
                    self.save_memories(force=True)
                
                # Cleanup old memories if needed
                if time.time() - self.last_cleanup_time >= MEMORY_CLEANUP_INTERVAL:
                    self.cleanup_old_memories()
                    self.last_cleanup_time = time.time()
                
        except Exception as e:
            print(f"Error saving pending memories and cleanup: {str(e)}")
            traceback.print_exc()

    def _process_end_of_day(self):
        """Process end of day events."""
        try:
            with self._lock:
                # Save any pending memories
                if self._pending_memories:
                    self.save_memories(force=True)
                
                # Cleanup old memories
                self.cleanup_old_memories()
                
                # Reset cleanup timer
                self.last_cleanup_time = time.time()
                
        except Exception as e:
            print(f"Error processing end of day: {str(e)}")
            traceback.print_exc()

    def get_pending_memories(self) -> List[Dict]:
        """Get pending memories from buffer for agents to process."""
        try:
            with self._lock:
                memories = self._pending_memories.copy()
                return memories
        except Exception as e:
            print(f"Error getting pending memories: {str(e)}")
            return []

    def get_recent_memories(self, agent_name: str, memory_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get the most recent memories for an agent, sorted by time."""
        try:
            with self._lock:
                if agent_name not in self.agent_memories:
                    return []

                # Get all memories for the agent
                agent_mems = self.agent_memories.get(agent_name, [])

                # Filter by memory type if provided
                if memory_type:
                    filtered_mems = [m for m in agent_mems if m.get('type') == memory_type]
                else:
                    filtered_mems = agent_mems

                # Sort memories by day and hour to get the most recent
                # The sort key uses a tuple (day, hour) for chronological sorting
                sorted_mems = sorted(
                    filtered_mems,
                    key=lambda m: (m.get('day', 0), m.get('hour', 0)),
                    reverse=True
                )

                # Return the last 'limit' memories
                return sorted_mems[:limit]
        except Exception as e:
            print(f"Error getting recent memories for {agent_name}: {str(e)}")
            traceback.print_exc()
            return []

    def clear_memories(self) -> None:
        """Clear all memories."""
        try:
            with self._lock:
                self.agent_memories = {
                    'activities': {},
                    'conversations': {},
                    'plans': {},
                    'states': {},
                    'metrics': {}
                }
                self._pending_memories = []
                self._initialize_memory_structures()
        except Exception as e:
            print(f"Error clearing memories: {str(e)}")
            traceback.print_exc()


    def _get_agent_memory_counts(self) -> Dict[str, Dict[str, int]]:
        """Get memory counts per agent and category."""
        agent_counts = {}
        for memory_id, memory in self.agent_memories.items():
            agent_name = memory['agent_name']
            if agent_name not in agent_counts:
                agent_counts[agent_name] = {'total': 0}
            
            agent_counts[agent_name]['total'] += 1
            
            category = self._get_category_for_memory_type(memory['type'])
            if category not in agent_counts[agent_name]:
                agent_counts[agent_name][category] = 0
            agent_counts[agent_name][category] += 1
            
        return agent_counts

    def get_agent_state_history(self, agent_name: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict]:
        """Get state history for an agent within a time range."""
        try:
            with self._lock:
                if agent_name not in self.state_history:
                    return []
                    
                history = self.state_history[agent_name]
                
                # Filter by time range if specified
                if start_time is not None or end_time is not None:
                    filtered_history = []
                    for entry in history:
                        entry_time = entry.get('simulation_time', 0)
                        if (start_time is None or entry_time >= start_time) and \
                           (end_time is None or entry_time <= end_time):
                            filtered_history.append(entry)
                    return filtered_history
                    
                return history
                
        except Exception as e:
            print(f"Error getting agent state history: {str(e)}")
            return []

    def extract_and_clean_plan(self, raw_response: Union[str, Dict[str, Any]], plan_type: str = "daily_plan") -> Dict[str, Any]:
        """Extract and clean plan from raw response into standardized JSON format.
        
        Args:
            raw_response: The raw response from the LLM
            plan_type: Type of plan - "daily_plan" (17 hours) or "emergency_replan" (2 hours)
        
        Expected output format:
        {
            'activities': [
                {
                    'time': int,  # Hour (0-23)
                    'action': str,  # Action type (e.g., 'work', 'eat', 'sleep')
                    'target': str,  # Target location
                    'description': str  # Description of the activity
                }
            ],
            'status': str,  # 'success', 'failed_to_parse', or 'error'
            'day': int,  # Current day
            'timestamp': str  # ISO format timestamp
        }
        """
        try:
            # Enhanced debugging for plan extraction
            print(f"[PLAN_DEBUG] extract_and_clean_plan called with plan_type: {plan_type}")
            print(f"[PLAN_DEBUG] Raw response type: {type(raw_response)}")
            if isinstance(raw_response, str):
                print(f"[PLAN_DEBUG] Raw response length: {len(raw_response)}")
                print(f"[PLAN_DEBUG] Raw response preview (first 300 chars): {raw_response[:300]}")
                print(f"[PLAN_DEBUG] Raw response preview (last 300 chars): {raw_response[-300:]}")
                print(f"[PLAN_DEBUG] Response contains 'activities': {'activities' in raw_response.lower()}")
                print(f"[PLAN_DEBUG] Response contains JSON markers: {{ = {'{' in raw_response}, }} = {'}' in raw_response}")
                print(f"[PLAN_DEBUG] Response contains code blocks: ``` = {'```' in raw_response}")
            else:
                print(f"[PLAN_DEBUG] Raw response content: {raw_response}")
            print(f"[PLAN_DEBUG] Starting plan extraction...")
            # Initialize standardized output structure
            cleaned_plan = {
                'activities': [],
                'status': 'success',
                'day': TimeManager.get_current_day(),
            }
            
            # If raw_response is already a dictionary
            if isinstance(raw_response, dict):
                if 'activities' in raw_response:
                    # Convert time values to integers and clean activities
                    activities = []
                    for activity in raw_response['activities']:
                        try:
                            # Convert time to integer if it's a string
                            if isinstance(activity['time'], str):
                                activity['time'] = int(activity['time'].split(':')[0])
                            activities.append(activity)
                        except (ValueError, KeyError) as e:
                            print(f"[WARNING] Skipping invalid activity: {activity}. Error: {e}")
                            continue
                    
                    # Validate that we have activities for all required hours (based on plan type)
                    if not self._validate_hourly_activities(activities, plan_type):
                        cleaned_plan['status'] = 'failed_to_parse'
                        print(f"Error: Plan validation failed for {plan_type}")
                        return cleaned_plan
                    cleaned_plan['activities'] = activities
                return cleaned_plan
                
            # If raw_response is a string, try to parse it as JSON
            if isinstance(raw_response, str):
                # Clean the response string first
                cleaned_response = raw_response.strip()
                
                # Remove code block markers if present (look anywhere in response, not just start)
                if '```' in cleaned_response:
                    # Find the first and last ```
                    first_marker = cleaned_response.find('```')
                    last_marker = cleaned_response.rfind('```')
                    if first_marker != -1 and last_marker != -1 and first_marker != last_marker:
                        # Extract content between markers (excluding the markers themselves)
                        content = cleaned_response[first_marker+3:last_marker]
                        # Remove any language specifier (like ```json)
                        if content.startswith('json'):
                            content = content[4:].strip()  # Remove 'json'
                        cleaned_response = content.strip()
                
                # Handle double curly braces (template format) - convert to single braces
                if '{{' in cleaned_response and '}}' in cleaned_response:
                    print(f"[DEBUG] Found double braces in response, converting to single braces")
                    cleaned_response = cleaned_response.replace('{{', '{').replace('}}', '}')
                
                try:
                    parsed = json.loads(cleaned_response)
                    if isinstance(parsed, dict) and 'activities' in parsed:
                        # Convert time values to integers and clean activities
                        activities = []
                        for activity in parsed['activities']:
                            try:
                                # Convert time to integer if it's a string
                                if isinstance(activity['time'], str):
                                    activity['time'] = int(activity['time'].split(':')[0])
                                activities.append(activity)
                            except (ValueError, KeyError) as e:
                                print(f"[WARNING] Skipping invalid activity: {activity}. Error: {e}")
                                continue
                        
                        # Validate that we have activities for all required hours
                        print(f"[PLAN_DEBUG] Parsed {len(activities)} activities from JSON")
                        if not self._validate_hourly_activities(activities, plan_type):
                            cleaned_plan['status'] = 'failed_to_parse'
                            print("Error: Plan must include activities for all hours from 07:00 to 23:00")
                            return cleaned_plan
                        print(f"[PLAN_DEBUG] JSON parsing successful! {len(activities)} activities validated")
                        cleaned_plan['activities'] = activities
                        return cleaned_plan
                except json.JSONDecodeError as e:
                    print(f"[WARNING] JSON parsing failed: {e}")
                    print(f"[DEBUG] Cleaned response that failed to parse: {cleaned_response[:200]}...")
                
                # If JSON parsing fails, try to extract plan using regex for text format
                # Pattern to match time, action, target, and reasoning
                pattern = r'(\d{2}):00\s+Action:\s+(\w+)\s+Target:\s+([^\n]+)\s+Reasoning:\s+([^\n]+)'
                matches = re.finditer(pattern, cleaned_response)
                
                for match in matches:
                    time, action, target, reasoning = match.groups()
                    try:
                        cleaned_plan['activities'].append({
                            'time': int(time),  # Convert time to integer
                            'action': action.strip(),
                            'target': target.strip(),
                            'description': reasoning.strip()
                        })
                    except ValueError as e:
                        print(f"[WARNING] Skipping invalid regex match: {match.groups()}. Error: {e}")
                        continue
                
                # Validate that we have activities for all required hours
                if not self._validate_hourly_activities(cleaned_plan['activities'], plan_type):
                    cleaned_plan['status'] = 'failed_to_parse'
                    print("Error: Plan must include activities for all hours from 07:00 to 23:00")
                    return cleaned_plan
                
                if cleaned_plan['activities']:
                    return cleaned_plan
            
            # If all parsing attempts fail
            cleaned_plan['status'] = 'failed_to_parse'
            print(f"[PLAN_DEBUG] All parsing attempts failed!")
            print(f"[ERROR] Failed to extract plan from response. Type: {type(raw_response)}")
            if isinstance(raw_response, str):
                print(f"[PLAN_DEBUG] Response preview: {raw_response[:200]}...")
                print(f"[PLAN_DEBUG] Contains {{: {'{' in raw_response}")
                print(f"[PLAN_DEBUG] Contains }}: {'}' in raw_response}")
                print(f"[PLAN_DEBUG] Contains 'activities': {'activities' in raw_response.lower()}")
            print(f"[PLAN_DEBUG] Returning failed plan with {len(cleaned_plan['activities'])} activities")
            return cleaned_plan
            
        except Exception as e:
            print(f"Error extracting and cleaning plan: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'activities': [],
                'status': 'error',
                'day': TimeManager.get_current_day(),
            }

    def _validate_hourly_activities(self, activities: List[Dict[str, Any]], plan_type: str = "daily_plan") -> bool:
        """Validate that activities cover all required hours."""
        try:
            # Determine expected parameters based on plan type
            if plan_type == "daily_plan":
                expected_count = 17
                expected_hours = list(range(7, 24))  # 07:00 to 23:00
            elif plan_type == "emergency_replan":
                expected_count = 2
                # For emergency replan, we need to determine the expected hours dynamically
                # since they depend on the current time
                current_hour = TimeManager.get_current_hour()
                expected_hours = [current_hour, current_hour + 1]
            else:
                print(f"Error: Unknown plan type: {plan_type}")
                return False
            
            # Check activity count
            if len(activities) != expected_count:
                print(f"Error: Expected {expected_count} hours of activities, got {len(activities)}")
                return False
                
            # Check that all expected hours are covered
            activity_hours = [activity.get('time') for activity in activities]
            missing_hours = set(expected_hours) - set(activity_hours)
            if missing_hours:
                print(f"Error: Missing activities for hours: {sorted(missing_hours)}")
                return False
                
            # Check for duplicate hours
            if len(activity_hours) != len(set(activity_hours)):
                print(f"Error: Duplicate hours found in activities")
                return False
            
            # Validate each activity
            for activity in activities:
                # Check required fields
                required_fields = ['time', 'action', 'target', 'description']
                for field in required_fields:
                    if field not in activity or activity[field] is None:
                        print(f"Error: Missing or null field '{field}' in activity: {activity}")
                        return False
                
                # Validate time is in expected range
                if activity['time'] not in expected_hours:
                    print(f"Error: Activity time {activity['time']} not in expected hours {expected_hours}")
                    return False
                
                # Validate action type
                if activity['action'] not in ['go_to', 'shop', 'work', 'rest', 'eat', 'sleep', 'idle']:
                    print(f"Error: Invalid action type: {activity['action']}")
                    return False
                
                # Validate target location exists (get valid locations from shared resource)
                valid_locations = self._get_valid_locations()
                if valid_locations and activity['target'] not in valid_locations:
                    print(f"Error: Invalid target location: '{activity['target']}'")
                    print(f"Valid locations: {list(valid_locations)}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error in activity validation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def record_plan_creation(self, agent_name: str, raw_response: str, cleaned_plan: Dict[str, Any], current_time: int, current_day: int) -> None:
        """Record plan creation in memory.
        
        Args:
            agent_name: Name of the agent
            raw_response: Original plan response (not stored)
            cleaned_plan: Standardized JSON structure of the plan
            current_time: Current simulation hour
            current_day: Current simulation day
        """
        try:
            # Try to acquire plan lock with timeout
            if not self._plan_lock.acquire(timeout=10):  # Increased timeout to 10 seconds
                print(f"[ERROR] Could not acquire plan lock for {agent_name} after 10 seconds")
                return
                
            try:
                print(f"[DEBUG] Acquired plan lock for {agent_name}")
                
                # Initialize agent's memory structure if needed
                if agent_name not in self.agent_memories:
                    self.agent_memories[agent_name] = []
                
                # Create plan memory entry
                plan_memory = {
                    'type': 'PLAN_CREATION',
                    'content': cleaned_plan,
                    'time': current_time,
                    'day': current_day,
                }
                
                # Add to memory buffer
                self._pending_memories.append((agent_name, plan_memory))
                
                # Add to agent's memories
                self.agent_memories[agent_name].append(plan_memory)
                
                print(f"[DEBUG] Stored plan for {agent_name} with {len(cleaned_plan.get('activities', []))} activities")
                
                # Force save if buffer is full
                if len(self._pending_memories) >= DEFAULT_MEMORY_BUFFER_SIZE:
                    self.force_save_memories()
                    
            finally:
                self._plan_lock.release()
                print(f"[DEBUG] Released plan lock for {agent_name}")
                    
        except KeyboardInterrupt:
            print(f"\nInterrupted while recording plan creation for {agent_name}")
            # Don't try to release lock here since we don't know if we acquired it
            raise
        except Exception as e:
            print(f"[ERROR] Error recording plan creation for {agent_name}: {str(e)}")
            # Don't try to release lock here since we don't know if we acquired it
            traceback.print_exc()

    def _migrate_state_update_types(self) -> None:
        """Migrate historical state update types to use the canonical type."""
        try:
            with self._lock:
                # Update state history
                for agent_name, history in self.state_history.items():
                    for entry in history:
                        if entry['type'] in ['STATE_UPDATE_EVENT', 'agent_state_update_event', 'state_update']:
                            entry['type'] = MemoryType.STATE_UPDATE.value

                # Update pending memories
                for i, (agent_name, memory) in enumerate(self._pending_memories):
                    if memory['type'] in ['STATE_UPDATE_EVENT', 'agent_state_update_event', 'state_update']:
                        memory['type'] = MemoryType.STATE_UPDATE.value

        except Exception as e:
            print(f"Error migrating state update types: {str(e)}")
            traceback.print_exc()

    def record_initial_state(self, agent: 'Agent') -> Result:
        """Records a comprehensive initial state for a newly created agent."""
        try:
            agent_name = agent.name
            
            # Consolidate all initial data into a single, comprehensive memory entry
            initial_state_content = {
                'name': agent.name,
                'age': agent.age,
                'occupation': agent.occupation,
                'residence': agent.residence,
                'workplace': agent.workplace,
                'income_info': agent.income_info,
                'relationship_status': agent.relationship_status,
                'spouse': agent.spouse,
                'household_members': agent.household_members,
                'energy_level': agent.energy_system.get_energy(agent_name),
                'grocery_level': agent.grocery_system.get_level(agent_name),
                'money': agent.money,
                'current_activity': 'idle',
                'current_location': agent.residence
            }

            memory_entry = {
                'type': MemoryType.AGENT_INFO.value, # A more descriptive type for this event
                'day': TimeManager.get_current_day(),
                'hour': TimeManager.get_current_hour(),
                'content': initial_state_content
            }
            
            # Add to memory buffer for saving
            self._pending_memories.append((agent_name, memory_entry))
            
            # Add to agent's in-memory list
            if agent_name not in self.agent_memories:
                self.agent_memories[agent_name] = []
            self.agent_memories[agent_name].append(memory_entry)
            
            print(f"[DEBUG] Recorded initial state for agent {agent_name}.")
            return Result.success(f"Recorded initial state for {agent_name}")
            
        except Exception as e:
            return Result.failure(f"Error recording initial state for {agent_name}: {str(e)}")

    def save_daily_summaries(self, metrics_manager: 'StabilityMetricsManager', current_day: int = None, agents_dict: Dict = None) -> None:
        """
        Generate and save daily narrative reflections for all agents.

        This replaces the old numeric summaries with Plan-Execution-Evaluation reflections.

        Args:
            metrics_manager: Metrics manager (kept for backward compatibility, not used)
            current_day: The day number to generate reflections for
            agents_dict: Dictionary mapping agent names to agent objects
        """
        try:
            # Use passed current_day parameter instead of calling TimeManager (prevents deadlocks)
            if current_day is None:
                # Fallback: try to get from preserved timing in memories
                if self._pending_memories:
                    latest_memory = max(self._pending_memories,
                                      key=lambda x: (x[1].get('simulation_day', 0), x[1].get('simulation_hour', 0)))
                    current_day = latest_memory[1].get('simulation_day', 1)
                else:
                    current_day = 1  # Default fallback

            if agents_dict is None:
                print("[WARNING] No agents_dict provided, skipping reflection generation")
                return

            print(f"\n[REFLECTION] Generating narrative reflections for all agents at end of Day {current_day}")

            day_reflections = {}

            # Generate reflection for each agent
            for agent_name, agent_obj in agents_dict.items():
                reflection = self.generate_agent_daily_reflection(agent_name, current_day, agent_obj)
                if reflection:
                    # Store in memory
                    if agent_name not in self.agent_reflections:
                        self.agent_reflections[agent_name] = {}
                    self.agent_reflections[agent_name][current_day] = reflection
                    day_reflections[agent_name] = reflection

            # Save all reflections to daily_summaries directory
            self._store_daily_reflections(current_day, day_reflections)

            print(f"[REFLECTION] Completed reflections for {len(day_reflections)} agents on Day {current_day}")

        except Exception as e:
            print(f"Error saving daily summaries: {str(e)}")
            traceback.print_exc()

    def _store_daily_reflections(self, day: int, reflections_dict: Dict[str, Dict]) -> None:
        """
        Save daily reflections to the daily_summaries directory.

        This stores narrative reflections (not numeric data) for agent decision-making.

        Args:
            day: The day number
            reflections_dict: Dictionary mapping agent names to reflection dictionaries
        """
        try:
            # Get the daily summaries directory path
            main_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'LLMAgentsTown_memory_records')
            summaries_dir = os.path.join(main_dir, 'simulation_daily_summaries')
            os.makedirs(summaries_dir, exist_ok=True)

            # Create filename with simulation timestamp
            summary_file = os.path.join(summaries_dir, f"daily_summary_{self.simulation_timestamp}.json")

            # Load existing summaries if file exists
            all_summaries = {}
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        all_summaries = json.load(f)
                except:
                    pass

            # Add current day's reflections
            all_summaries[f"day_{day}"] = reflections_dict

            # Save back to file
            with open(summary_file, 'w') as f:
                json.dump(all_summaries, f, indent=2)

            print(f"[REFLECTION] Saved narrative reflections to {summary_file}")

        except Exception as e:
            print(f"[ERROR] Failed to save daily reflections: {str(e)}")
            traceback.print_exc()

    def get_plan_for_hour(self, agent_name: str, current_hour: int, current_day: int) -> Optional[Dict[str, Any]]:
        """Get the activity planned for the current hour from memory."""
        try:
            # Try to acquire plan lock with timeout
            if not self._plan_lock.acquire(timeout=10):  # Increased timeout to 10 seconds
                print(f"[ERROR] Could not acquire plan lock for {agent_name} after 10 seconds")
                return None
                
            try:
                print(f"[DEBUG] Acquired plan lock for {agent_name}")
                
                # Get the most recent plan for this agent and day
                plan_creation = None
                
                # First check pending memories
                for memory in reversed(self._pending_memories):
                    if (memory[0] == agent_name and 
                        memory[1].get('type') == 'PLAN_CREATION' and
                        memory[1].get('day') == current_day):
                        plan_creation = memory[1]
                        break
                
                # If not found in pending memories, check agent's memories
                if not plan_creation and agent_name in self.agent_memories:
                    for memory in reversed(self.agent_memories[agent_name]):
                        if (memory.get('type') == 'PLAN_CREATION' and
                            memory.get('day') == current_day):
                            plan_creation = memory
                            break
                
                if not plan_creation:
                    print(f"[DEBUG] No plan found for {agent_name} on day {current_day}")
                    return None
                    
                # Find the activity for the current hour
                plan = plan_creation.get('content', {})
                for activity in plan.get('activities', []):
                    if activity.get('time') == current_hour:
                        print(f"[DEBUG] Found activity for {agent_name} at hour {current_hour}: {activity}")
                        return activity
                        
                print(f"[DEBUG] No activity found for {agent_name} at hour {current_hour}")
                return None
                
            finally:
                self._plan_lock.release()
                print(f"[DEBUG] Released plan lock for {agent_name}")
                    
        except Exception as e:
            print(f"[ERROR] Error getting plan for hour {current_hour}: {str(e)}")
            traceback.print_exc()
            return None

    # ======= PLAN SAVING/LOADING FOR DEBUGGING =======
    
    def save_current_plans(self, agents: Dict[str, 'Agent']) -> None:
        """Save all agents' current daily plans for debugging reuse with simulation timestamp."""
        try:
            current_day = TimeManager.get_current_day()
            current_hour = TimeManager.get_current_hour()
            
            # Collect current plans
            current_plans = {}
            for agent_name, agent in agents.items():
                if hasattr(agent, 'daily_plan') and agent.daily_plan:
                    current_plans[agent_name] = agent.daily_plan
            
            if not current_plans:
                print("[SAVE] No plans to save")
                return
            
            # Load existing saved plans or create new structure
            all_saved_plans = {}
            if os.path.exists(self.saved_plans_file):
                try:
                    with open(self.saved_plans_file, 'r') as f:
                        all_saved_plans = json.load(f)
                except Exception as e:
                    print(f"[SAVE] Error reading existing plans file: {e}")
                    all_saved_plans = {}
            
            # Add current plans with timestamp
            save_entry = {
                'day': current_day,
                'hour': current_hour,
                'timestamp': datetime.now().isoformat(),
                'simulation_timestamp': self.simulation_timestamp,
                'plans': current_plans
            }
            
            # Use a unique key for this save
            save_key = f"day_{current_day}_hour_{current_hour}"
            all_saved_plans[save_key] = save_entry
            
            # Keep only the 5 most recent saves to avoid file bloat
            if len(all_saved_plans) > 5:
                # Sort by day and hour, keep the latest 5
                sorted_keys = sorted(all_saved_plans.keys(), 
                                   key=lambda k: (all_saved_plans[k]['day'], all_saved_plans[k]['hour']))
                for old_key in sorted_keys[:-5]:
                    del all_saved_plans[old_key]
            
            # Save to simulation-specific file
            with open(self.saved_plans_file, 'w') as f:
                json.dump(all_saved_plans, f, indent=2)
            
            print(f"[SAVE] Daily plans saved to: {self.saved_plans_file}")
            print(f"[SAVE] Saved plans for {len(current_plans)} agents (Day {current_day}, Hour {current_hour})")
            print(f"[SAVE] Total saved sessions: {len(all_saved_plans)}")
            
        except Exception as e:
            print(f"Error saving daily plans: {str(e)}")
            traceback.print_exc()

    def get_latest_saved_plans(self) -> Optional[tuple]:
        """Find and load the most recent saved plans from current simulation or look for other simulation files."""
        try:
            # First try to load from current simulation file
            if os.path.exists(self.saved_plans_file):
                result = self._load_plans_from_file(self.saved_plans_file)
                if result:
                    return result
            
            # If no plans in current simulation file, look for other simulation files
            plans_dir = os.path.dirname(self.saved_plans_file)
            if os.path.exists(plans_dir):
                plan_files = [f for f in os.listdir(plans_dir) if f.startswith('saved_plans_') and f.endswith('.json')]
                plan_files.sort(reverse=True)  # Sort by filename (newest timestamp first)
                
                for plan_file in plan_files:
                    if plan_file != os.path.basename(self.saved_plans_file):  # Skip current file (already checked)
                        file_path = os.path.join(plans_dir, plan_file)
                        result = self._load_plans_from_file(file_path)
                        if result:
                            sim_timestamp = plan_file.replace('saved_plans_', '').replace('.json', '')
                            print(f"[LOAD] Using plans from different simulation: {sim_timestamp}")
                            return result
            
            print(f"[LOAD] No saved plans found in any simulation files")
            return None
            
        except Exception as e:
            print(f"Error finding latest saved plans: {str(e)}")
            traceback.print_exc()
            return None

    def _load_plans_from_file(self, file_path: str) -> Optional[tuple]:
        """Helper method to load plans from a specific file."""
        try:
            with open(file_path, 'r') as f:
                all_saved_plans = json.load(f)
            
            if not all_saved_plans:
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
                return None
            
            latest_save = all_saved_plans[latest_key]
            plans_data = latest_save.get('plans', {})
            
            print(f"[LOAD] Found latest saved plans from Day {latest_day}, Hour {latest_hour}")
            print(f"[LOAD] Loaded plans for {len(plans_data)} agents from {file_path}")
            print(f"[LOAD] Available saves in file: {len(all_saved_plans)}")
            
            return latest_day, plans_data
            
        except Exception as e:
            print(f"Error loading plans from {file_path}: {str(e)}")
            return None

    def show_available_saved_plans(self) -> None:
        """Show what saved plans are available for debugging across all simulation files."""
        try:
            plans_dir = os.path.dirname(self.saved_plans_file)
            if not os.path.exists(plans_dir):
                print(f"[DEBUG] No plans directory found at {plans_dir}")
                return
            
            # Find all plan files
            plan_files = [f for f in os.listdir(plans_dir) if f.startswith('saved_plans_') and f.endswith('.json')]
            
            if not plan_files:
                print(f"[DEBUG] No saved plan files found in {plans_dir}")
                return
            
            print(f"[DEBUG] Found {len(plan_files)} saved plan files:")
            
            # Sort files by timestamp (newest first)
            plan_files.sort(reverse=True)
            
            for plan_file in plan_files:
                file_path = os.path.join(plans_dir, plan_file)
                sim_timestamp = plan_file.replace('saved_plans_', '').replace('.json', '')
                
                try:
                    # Get file size
                    stat = os.stat(file_path)
                    size_kb = stat.st_size / 1024
                    
                    # Load and show content
                    with open(file_path, 'r') as f:
                        all_saved_plans = json.load(f)
                    
                    if all_saved_plans:
                        print(f"[DEBUG] \n📁 Simulation {sim_timestamp} ({size_kb:.1f} KB):")
                        
                        # Sort by day and hour
                        sorted_saves = sorted(all_saved_plans.items(), 
                                            key=lambda x: (x[1]['day'], x[1]['hour']))
                        
                        for save_key, save_data in sorted_saves:
                            day = save_data.get('day', 0)
                            hour = save_data.get('hour', 0)
                            agent_count = len(save_data.get('plans', {}))
                            timestamp = save_data.get('timestamp', 'unknown')
                            
                            # Mark current simulation
                            current_marker = " ← CURRENT" if sim_timestamp == self.simulation_timestamp else ""
                            print(f"[DEBUG]   Day {day}, Hour {hour}: {agent_count} agents ({timestamp}){current_marker}")
                    else:
                        print(f"[DEBUG] \n📁 Simulation {sim_timestamp}: Empty file")
                        
                except Exception as e:
                    print(f"[DEBUG] \n📁 Simulation {sim_timestamp}: Error reading file - {str(e)}")
                
        except Exception as e:
            print(f"[DEBUG] Error checking saved plans: {str(e)}")

    def force_save_memories(self) -> None:
        """Force save all pending memories immediately."""
        try:
            self.save_memories(force=True)
            print("[SAVE] Force saved all pending memories")
        except Exception as e:
            print(f"Error force saving memories: {str(e)}")
            traceback.print_exc()

    def _get_valid_locations(self) -> Set[str]:
        """Get valid locations from the shared resource manager or return empty set if not available."""
        try:
            # Try to get valid locations from shared resource manager
            from shared_trackers import SharedResourceManager
            resource_mgr = SharedResourceManager()
            return resource_mgr.get_valid_locations()
        except Exception:
            # If we can't get locations from shared resource, return empty set
            # This allows validation to pass when location info isn't available
            return set()

    def _get_memory_base_path(self) -> str:
        """Get the base path for memory storage."""
        return self.memory_base_path

    # ============================================================================
    # HIERARCHICAL MEMORY: AGENT DAILY REFLECTIONS (PLAN-EXECUTION-EVALUATION)
    # ============================================================================

    def generate_agent_daily_reflection(self, agent_name: str, day: int, agent_obj) -> Dict[str, Any]:
        """
        Generate a comprehensive daily reflection for an agent using Plan-Execution-Evaluation cycle.

        This function:
        1. Retrieves the original plan from PLAN_CREATION
        2. Retrieves plan updates from PLAN_UPDATE
        3. Retrieves execution memories (WORK, MEAL, CONVERSATION, TRAVEL, REST)
        4. Gets current metrics from agent object (money, energy, grocery)
        5. Sends all data to LLM for evaluation with nuanced language
        6. Returns structured reflection with evaluation keywords

        Args:
            agent_name: Name of the agent
            day: The day number to generate reflection for
            agent_obj: The agent object to get current state

        Returns:
            Dict containing reflection with sections: plan, execution, evaluation, metrics
        """
        try:
            print(f"[REFLECTION] Generating daily reflection for {agent_name} on Day {day}")

            # 1. Retrieve original plan (PLAN_CREATION)
            original_plan = None
            plan_memories = [m for m in self.agent_memories.get(agent_name, [])
                           if m.get('type') == 'PLAN_CREATION' and m.get('day') == day]
            if plan_memories:
                original_plan = plan_memories[0].get('content', {})

            # 2. Retrieve plan updates (PLAN_UPDATE)
            plan_updates = [m for m in self.agent_memories.get(agent_name, [])
                          if m.get('type') == 'PLAN_UPDATE' and m.get('day') == day]

            # 3. Retrieve execution memories
            execution_types = ['WORK', 'MEAL', 'CONVERSATION', 'TRAVEL', 'REST', 'STATE_UPDATE']
            execution_memories = [m for m in self.agent_memories.get(agent_name, [])
                                if m.get('type') in execution_types and m.get('day') == day]

            # Sort by hour
            execution_memories.sort(key=lambda m: m.get('hour', 0))

            # 4. Get current metrics from agent object
            current_metrics = {
                'money': getattr(agent_obj, 'money', 0),
                'energy': getattr(agent_obj.energy_system, 'get_energy', lambda x: 0)(agent_name) if hasattr(agent_obj, 'energy_system') else 0,
                'grocery': getattr(agent_obj.grocery_system, 'get_level', lambda x: 0)(agent_name) if hasattr(agent_obj, 'grocery_system') else 0,
                'daily_income': getattr(agent_obj, 'daily_income', 0),
                'daily_expenses': getattr(agent_obj, 'daily_expenses', 0)
            }

            # 5. Prepare LLM prompt for evaluation
            prompt = self._build_reflection_prompt(
                agent_name, day, original_plan, plan_updates,
                execution_memories, current_metrics
            )

            # 6. Call LLM to generate reflection
            if not self.model_mgr:
                raise RuntimeError("ModelManager not set. Call set_model_manager() first.")

            llm_response = self.model_mgr.generate(prompt, 'action')

            # 7. Structure the reflection
            reflection = {
                'agent_name': agent_name,
                'day': day,
                'timestamp': datetime.now().isoformat(),
                'original_plan': original_plan,
                'plan_updates_count': len(plan_updates),
                'execution_summary': llm_response,  # LLM-generated narrative with evaluation
                'metrics': current_metrics,
                'memory_counts': {
                    'work': len([m for m in execution_memories if m.get('type') == 'WORK']),
                    'meals': len([m for m in execution_memories if m.get('type') == 'MEAL']),
                    'conversations': len([m for m in execution_memories if m.get('type') == 'CONVERSATION']),
                    'travel': len([m for m in execution_memories if m.get('type') == 'TRAVEL']),
                }
            }

            print(f"[REFLECTION] Successfully generated reflection for {agent_name} on Day {day}")
            return reflection

        except Exception as e:
            print(f"[ERROR] Failed to generate reflection for {agent_name} on Day {day}: {str(e)}")
            traceback.print_exc()
            return None

    def _build_reflection_prompt(self, agent_name: str, day: int,
                                original_plan: Dict, plan_updates: List[Dict],
                                execution_memories: List[Dict], metrics: Dict) -> str:
        """
        Build the LLM prompt for generating agent daily reflection.

        This prompt instructs the LLM to:
        - Compare original plan vs actual execution
        - Include plan changes and reasons
        - Evaluate outcomes using numerical metrics
        - Use nuanced evaluation language organically
        """

        prompt = f"""You are generating a daily reflection summary for {agent_name} at the end of Day {day}.

This reflection should follow a Plan-Execution-Evaluation structure:

## ORIGINAL PLAN (Morning)
"""

        # Add original plan
        if original_plan and 'activities' in original_plan:
            prompt += "The agent planned to:\n"
            for activity in original_plan['activities']:
                time = activity.get('time', '?')
                action = activity.get('action', '?')
                target = activity.get('target', '?')
                desc = activity.get('description', '')
                prompt += f"- Hour {time}: {action} at {target}"
                if desc:
                    prompt += f" ({desc})"
                prompt += "\n"
        else:
            prompt += "No formal plan was recorded.\n"

        # Add plan updates
        prompt += f"\n## PLAN MODIFICATIONS (During Day)\n"
        if plan_updates:
            prompt += f"The plan was modified {len(plan_updates)} times:\n"
            for i, update in enumerate(plan_updates[:5], 1):  # Show first 5 updates
                hour = update.get('hour', '?')
                reason = update.get('content', {}).get('reason', 'Unknown reason')
                prompt += f"{i}. Hour {hour}: {reason}\n"
        else:
            prompt += "No modifications were made to the original plan.\n"

        # Add execution summary
        prompt += f"\n## ACTUAL EXECUTION (What Happened)\n"

        # Group memories by type
        work_mems = [m for m in execution_memories if m.get('type') == 'WORK']
        meal_mems = [m for m in execution_memories if m.get('type') == 'MEAL']
        conv_mems = [m for m in execution_memories if m.get('type') == 'CONVERSATION']

        if work_mems:
            prompt += f"\nWork ({len(work_mems)} sessions):\n"
            for mem in work_mems[:3]:  # Show first 3
                hour = mem.get('hour', '?')
                content = mem.get('content', {})
                location = content.get('location', '?')
                wage = content.get('wage', 0)
                prompt += f"- Hour {hour}: Worked at {location}, earned ${wage}\n"

        if meal_mems:
            prompt += f"\nMeals ({len(meal_mems)} times):\n"
            for mem in meal_mems[:5]:  # Show first 5
                hour = mem.get('hour', '?')
                content = mem.get('content', {})
                food = content.get('food_name', '?')
                location = content.get('location', '?')
                cost = content.get('price', 0)
                prompt += f"- Hour {hour}: Ate {food} at {location} (${cost})\n"

        if conv_mems:
            prompt += f"\nConversations ({len(conv_mems)} interactions):\n"
            for mem in conv_mems[:3]:  # Show first 3
                hour = mem.get('hour', '?')
                content = mem.get('content', {})
                participants = content.get('participants', [])
                location = content.get('location', '?')
                prompt += f"- Hour {hour}: Talked with {', '.join(participants)} at {location}\n"

        # Add metrics
        prompt += f"\n## OUTCOMES (Metrics)\n"
        prompt += f"Financial: Started with some money, earned ${metrics['daily_income']}, spent ${metrics['daily_expenses']}, ended with ${metrics['money']}\n"
        prompt += f"Energy: Current level is {metrics['energy']}\n"
        prompt += f"Grocery: Stock level is {metrics['grocery']}\n"

        # Instructions for LLM
        prompt += f"""

## YOUR TASK
Write a concise 2-3 paragraph reflection that:
1. Compares what was planned vs what actually happened
2. Notes any significant changes or unexpected events
3. Evaluates the day's outcome based on metrics (money earned/spent, energy management)
4. Uses natural evaluative language when appropriate (e.g., "successful", "productive", "delayed", "incomplete", "valuable", "careless")

The evaluation should flow naturally from the numerical outcomes - if the agent earned good money and maintained energy, describe it positively; if plans were disrupted or resources depleted, describe challenges.

Keep the tone third-person and objective, like a behavioral researcher's notes.

REFLECTION:"""

        return prompt

    def get_agent_reflection_summary(self, agent_name: str, days: int = 3) -> str:
        """
        Retrieve recent reflection summaries for an agent to use in planning prompts.

        Args:
            agent_name: Name of the agent
            days: Number of recent days to retrieve (default 3)

        Returns:
            Formatted string of recent reflections, or empty string if none available
        """
        try:
            if agent_name not in self.agent_reflections:
                return ""

            agent_refs = self.agent_reflections[agent_name]
            if not agent_refs:
                return ""

            # Get most recent days
            sorted_days = sorted(agent_refs.keys(), reverse=True)
            recent_days = sorted_days[:days]

            if not recent_days:
                return ""

            summary = ""
            for day in sorted(recent_days):  # Sort chronologically for display
                reflection = agent_refs[day]
                execution_summary = reflection.get('execution_summary', '')
                metrics = reflection.get('metrics', {})

                summary += f"\n**Day {day} Summary:**\n"
                summary += f"{execution_summary}\n"
                summary += f"(Ended with: ${metrics.get('money', 0)}, Energy: {metrics.get('energy', 0)})\n"

            return summary

        except Exception as e:
            print(f"[ERROR] Failed to retrieve reflection summary for {agent_name}: {str(e)}")
            return ""
