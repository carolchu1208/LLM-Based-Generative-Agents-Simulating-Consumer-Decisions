import json
import os
from datetime import datetime
import time
from typing import List, Dict, Optional, Any
import traceback
import threading
from typing import Optional, List
from collections import defaultdict

# Import the refined constants
from simulation_constants import MEMORY_TYPES, ACTIVITY_TYPES, MemoryEvent

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

class MemoryManager:
    def __init__(self, memory_dir: str):
        """Initialize the memory manager with a directory for storing memories."""
        self.memory_dir = memory_dir
        self.agent_memory_dir = os.path.join(memory_dir, 'simulation_agents')
        self.simulation_memory_dir = os.path.join(memory_dir, 'simulation_memory')
        os.makedirs(self.agent_memory_dir, exist_ok=True)
        os.makedirs(self.simulation_memory_dir, exist_ok=True)
        
        # Initialize memory storage
        self.memories = {}  # agent_name -> list of memories
        self.simulation_memories = []  # List of simulation-wide memories
        
        # Create memory files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.agent_memory_file = os.path.join(self.agent_memory_dir, f'consolidated_memories_{timestamp}.json')
        self.simulation_memory_file = os.path.join(self.simulation_memory_dir, f'simulation_memories_{timestamp}.jsonl')
        
        # Initialize files
        self._initialize_memory_files()
        
    def _initialize_memory_files(self):
        """Initialize memory files with proper structure."""
        try:
            # Initialize agent memory file
            with open(self.agent_memory_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'agents': {}
                }, f, indent=2)
            
            # Initialize simulation memory file
            with open(self.simulation_memory_file, 'w') as f:
                f.write('')  # Empty file for JSONL format
                
        except Exception as e:
            print(f"Error initializing memory files: {str(e)}")
            traceback.print_exc()
            
    def add_memory(self, memory_event: MemoryEvent) -> None:
        """Add a memory event and persist it to disk."""
        try:
            # Validate memory type
            if memory_event.memory_type not in MEMORY_TYPES:
                raise ValueError(f"Invalid memory type: {memory_event.memory_type}")
            
            # Add to in-memory storage
            if memory_event.data.get('agent_name') not in self.memories:
                self.memories[memory_event.data.get('agent_name')] = []
            self.memories[memory_event.data.get('agent_name')].append(memory_event)
            
            # Persist to disk
            self._persist_memory(memory_event)
            
        except Exception as e:
            print(f"Error adding memory: {str(e)}")
            traceback.print_exc()
            
    def _persist_memory(self, memory_event: MemoryEvent) -> None:
        """Persist a memory event to the appropriate file."""
        try:
            # Prepare memory data
            memory_data = {
                'agent_name': memory_event.data.get('agent_name'),
                'memory_type': memory_event.memory_type,
                'memory_data': memory_event.data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to simulation memory file (JSONL format)
            with open(self.simulation_memory_file, 'a') as f:
                f.write(json.dumps(memory_data) + '\n')
            
            # Update consolidated agent memories
            self._update_consolidated_memories(memory_event)
            
        except Exception as e:
            print(f"Error persisting memory: {str(e)}")
            traceback.print_exc()
            
    def _update_consolidated_memories(self, memory_event: MemoryEvent) -> None:
        """Update the consolidated memories file for an agent."""
        try:
            # Load current consolidated memories
            with open(self.agent_memory_file, 'r') as f:
                consolidated = json.load(f)
            
            # Initialize agent if not exists
            agent_name = memory_event.data.get('agent_name')
            if agent_name not in consolidated['agents']:
                consolidated['agents'][agent_name] = {
                    'agent_name': agent_name,
                    'memories': {
                        'conversations': [],
                        'purchases': [],
                        'travels': [],
                        'activities': [],
                        'planning': [],
                        'state_updates': [],
                        'satisfaction_ratings': [],
                        'system_events': [],
                        'other': []
                    },
                    'total_memories': 0
                }
            
            # Add memory to appropriate category
            agent_data = consolidated['agents'][agent_name]
            memory_category = self._get_memory_category(memory_event.memory_type)
            
            if memory_category in agent_data['memories']:
                agent_data['memories'][memory_category].append(memory_event.data)
                agent_data['total_memories'] += 1
            
            # Save updated consolidated memories
            with open(self.agent_memory_file, 'w') as f:
                json.dump(consolidated, f, indent=2)
            
        except Exception as e:
            print(f"Error updating consolidated memories: {str(e)}")
            traceback.print_exc()
            
    def _get_memory_category(self, memory_type: str) -> str:
        """Get the category for a memory type."""
        categories = {
            'CONVERSATION_EVENT': 'conversations',
            'PURCHASE_EVENT': 'purchases',
            'TRAVEL_EVENT': 'travels',
            'ACTIVITY_EVENT': 'activities',
            'PLANNING_EVENT': 'planning',
            'AGENT_STATE_UPDATE_EVENT': 'state_updates',
            'SATISFACTION_EVENT': 'satisfaction_ratings',
            'SYSTEM_EVENT': 'system_events'
        }
        return categories.get(memory_type, 'other')
            
    def get_memories(self, agent_name: str, memory_type: Optional[str] = None, 
                    start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[MemoryEvent]:
        """Get memories for an agent with optional filtering."""
        try:
            memories = self.memories.get(agent_name, [])
            
            # Apply filters
            if memory_type:
                memories = [m for m in memories if m.memory_type == memory_type]
            if start_time is not None:
                memories = [m for m in memories if m.memory_data.get('time', 0) >= start_time]
            if end_time is not None:
                memories = [m for m in memories if m.memory_data.get('time', 0) <= end_time]
                
            return memories
            
        except Exception as e:
            print(f"Error getting memories: {str(e)}")
            traceback.print_exc()
            return []
            
    def get_memory_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get a summary of an agent's memories."""
        try:
            memories = self.memories.get(agent_name, [])
            
            # Count memories by type
            memory_counts = {}
            for memory in memories:
                memory_counts[memory.memory_type] = memory_counts.get(memory.memory_type, 0) + 1
                
            # Get latest state
            latest_state = None
            for memory in reversed(memories):
                if memory.memory_type == 'AGENT_STATE_UPDATE_EVENT':
                    latest_state = memory.memory_data
                    break
                    
            return {
                'total_memories': len(memories),
                'memory_counts': memory_counts,
                'latest_state': latest_state
            }
            
        except Exception as e:
            print(f"Error getting memory summary: {str(e)}")
            traceback.print_exc()
            return {}
            
    def save_conversation_log(self, agent_name: str, content: str, log_type: str = 'CONVERSATION_EVENT') -> None:
        """Save a conversation log."""
        try:
            memory_data = {
                'content': content,
                'time': datetime.now().isoformat(),
                'type': log_type
            }
            
            memory_event = MemoryEvent(
                agent_name=agent_name,
                memory_type=log_type,
                memory_data=memory_data
            )
            
            self.add_memory(memory_event)
            
        except Exception as e:
            print(f"Error saving conversation log: {str(e)}")
            traceback.print_exc()
            
    def load_memories(self) -> None:
        """Load memories from disk."""
        try:
            # Load simulation memories
            if os.path.exists(self.simulation_memory_file):
                with open(self.simulation_memory_file, 'r') as f:
                    for line in f:
                        memory_data = json.loads(line)
                        memory_event = MemoryEvent(
                            agent_name=memory_data['agent_name'],
                            memory_type=memory_data['memory_type'],
                            memory_data=memory_data['memory_data']
                        )
                        
                        if memory_event.agent_name not in self.memories:
                            self.memories[memory_event.agent_name] = []
                        self.memories[memory_event.agent_name].append(memory_event)
            
            # Load consolidated agent memories
            if os.path.exists(self.agent_memory_file):
                with open(self.agent_memory_file, 'r') as f:
                    consolidated = json.load(f)
                    for agent_name, agent_data in consolidated['agents'].items():
                        for category, memories in agent_data['memories'].items():
                            for memory_data in memories:
                                memory_event = MemoryEvent(
                                    agent_name=agent_name,
                                    memory_type=self._get_memory_type_from_category(category),
                                    memory_data=memory_data
                                )
                                
                                if agent_name not in self.memories:
                                    self.memories[agent_name] = []
                                self.memories[agent_name].append(memory_event)
                    
        except Exception as e:
            print(f"Error loading memories: {str(e)}")
            traceback.print_exc()
            
    def _get_memory_type_from_category(self, category: str) -> str:
        """Get the memory type from a category."""
        type_map = {
            'conversations': 'CONVERSATION_EVENT',
            'purchases': 'PURCHASE_EVENT',
            'travels': 'TRAVEL_EVENT',
            'activities': 'ACTIVITY_EVENT',
            'planning': 'PLANNING_EVENT',
            'state_updates': 'AGENT_STATE_UPDATE_EVENT',
            'satisfaction_ratings': 'SATISFACTION_EVENT',
            'system_events': 'SYSTEM_EVENT'
        }
        return type_map.get(category, 'OTHER_EVENT')
            
    def clear_memories(self) -> None:
        """Clear all memories."""
        try:
            self.memories.clear()
            self.simulation_memories.clear()
            
            # Clear files
            with open(self.simulation_memory_file, 'w') as f:
                f.write('')
            with open(self.agent_memory_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'agents': {}
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error clearing memories: {str(e)}")
            traceback.print_exc()

    def get_recent_memories(self, agent_name: str, memory_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories for an agent with optional filtering."""
        try:
            # Get all memories for the agent
            memories = self.memories.get(agent_name, [])
            
            # Apply memory type filter if specified
            if memory_type:
                memories = [m for m in memories if m.memory_type == memory_type]
            
            # Sort by time (most recent first)
            memories.sort(key=lambda m: m.memory_data.get('time', 0), reverse=True)
            
            # Take the most recent ones up to the limit
            recent_memories = memories[:limit]
            
            # Convert to dictionary format
            return [m.memory_data for m in recent_memories]
            
        except Exception as e:
            print(f"Error getting recent memories: {str(e)}")
            traceback.print_exc()
            return []