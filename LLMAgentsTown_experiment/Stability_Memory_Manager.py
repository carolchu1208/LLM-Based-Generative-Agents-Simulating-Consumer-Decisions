import json
import os
from datetime import datetime
import time
from typing import List, Dict, Optional
import traceback
import threading

# Import the refined constants
from simulation_constants import MEMORY_TYPES, ACTIVITY_TYPES # Added ACTIVITY_TYPES for validator

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
    def __init__(self, memory_limit=1000, time_unit=0.5):  # 1 unit = 0.5 seconds
        self.memory_limit = memory_limit
        self.memories: Dict[str, List[Dict]] = {}
        self.compressed_memories = {}
        self.memory_ratings = {}
        self.start_time = datetime.now()
        self.time_unit = time_unit
        self.consolidation_threshold = 0.3  # Threshold for memory consolidation
        self.memory_id_counter = 0  # Counter for unique memory IDs
        self._lock = threading.Lock()
        
        # Updated validator dictionary using new MEMORY_TYPES constants
        self.memory_type_validators = {
            MEMORY_TYPES['LOCATION_CHANGE_EVENT']: self._validate_location_change,
            MEMORY_TYPES['CONVERSATION_LOG_EVENT']: self._validate_conversation_log, # Renamed for clarity
            MEMORY_TYPES['FOOD_PURCHASE_EVENT']: self._validate_food_purchase,
            MEMORY_TYPES['GROCERY_PURCHASE_EVENT']: self._validate_grocery_purchase,
            MEMORY_TYPES['ACTIVITY_EVENT']: self._validate_activity_event,
            MEMORY_TYPES['PLANNING_EVENT']: self._validate_planning_event,
            MEMORY_TYPES['AGENT_STATE_UPDATE_EVENT']: self._validate_agent_state_update,
            # ACTION_RAW_OUTPUT, SYSTEM_EVENT, GENERIC_EVENT often just need 'content' and 'time',
            # so a generic check in add_memory might suffice, or we can add simple validators if needed.
            MEMORY_TYPES['ACTION_RAW_OUTPUT']: self._validate_generic_content_event,
            MEMORY_TYPES['GENERIC_EVENT']: self._validate_generic_content_event,
            MEMORY_TYPES['SYSTEM_EVENT']: self._validate_generic_content_event,
        }
        print(f"Simulation started at: {self.start_time.strftime('%H:%M:%S')}")

    # --- Validator Methods (ensure these align with your data structures) ---
    def _validate_generic_content_event(self, data):
        """Validates events that primarily need 'content' and 'time'."""
        required = ['content', 'time']
        return all(key in data for key in required)

    def _validate_location_change(self, data):
        """Validate location change memory data"""
        # 'from' and 'to' could be location names (strings) or coordinates (tuples)
        required = ['time'] # 'from' and 'to' are good to have, but path might start/end outside named locs
        if not all(key in data for key in required):
            return False
        # content detailing the change is also good
        if 'content' not in data and ('from_coord' not in data or 'to_coord' not in data):
            return False # Must have some detail about the change
        return True

    def _validate_conversation_log(self, data):
        """Validate conversation log memory data"""
        required = ['content', 'time', 'location'] # 'participants' is highly recommended
        if not all(key in data for key in required):
            return False
        if 'participants' not in data or not isinstance(data['participants'], list):
            # print("Warning: 'participants' missing or not a list in CONVERSATION_LOG_EVENT")
            pass # Participants are good but maybe not strictly required for all logs
        return True

    def _validate_food_purchase(self, data):
        """Validate food purchase memory data"""
        required = ['item_description', 'amount', 'location', 'time'] # 'used_discount' is good to have
        return all(key in data for key in required)
    
    def _validate_grocery_purchase(self, data):
        """Validate grocery purchase memory data"""
        required = ['items_list', 'amount', 'location', 'time'] # 'used_discount' is good to have
        return all(key in data for key in required)

    def _validate_activity_event(self, data):
        """Validate activity event memory data"""
        required = ['activity_type_tag', 'description', 'time']
        if not all(key in data for key in required):
            return False
        if data['activity_type_tag'] not in ACTIVITY_TYPES.values():
            # print(f"Warning: Invalid 'activity_type_tag': {data['activity_type_tag']} in ACTIVITY_EVENT")
            return False
        return True

    def _validate_planning_event(self, data):
        """Validate planning event memory data."""
        required = ['plan_content', 'time'] # or similar e.g. 'event_type' like 'daily_plan_generated'
        return all(key in data for key in required)

    def _validate_agent_state_update(self, data):
        """Validate agent state update memory data."""
        required = ['energy_level', 'money', 'grocery_level', 'time']
        return all(key in data for key in required)

    def add_memory(self, agent_name: str, memory_type_key: str, data: Dict):
        """Add a memory record for an agent. This method is thread-safe.
           memory_type_key should be one of the keys from MEMORY_TYPES dictionary.
        """
        if memory_type_key not in MEMORY_TYPES:
            print(f"Warning: Attempted to add memory for {agent_name} with unrecognized type key: {memory_type_key}. Memory not added.")
            return

        actual_memory_type_value = MEMORY_TYPES[memory_type_key]

        if not isinstance(data, dict) or 'time' not in data:
            print(f"Warning: Invalid data for memory type {actual_memory_type_value} for agent {agent_name}. Missing 'time' or not a dict. Data: {data}. Memory not added.")
            return
        
        validator = self.memory_type_validators.get(actual_memory_type_value)
        if validator and not validator(data):
            print(f"Warning: Data validation failed for memory type '{actual_memory_type_value}' for agent {agent_name}. Data: {data}. Memory not added.")
            return

        with self._lock:
            if agent_name not in self.memories:
                self.memories[agent_name] = []
            
            current_sim_time = data.get('time') # Should always be present due to check above
            # Ensure simulation_day and simulation_hour are consistently added
            day = (int(current_sim_time) // 24) + 1
            hour = int(current_sim_time) % 24
            data['simulation_day'] = day
            data['simulation_hour'] = hour
            
            memory_id = data.get('id', f"mem_{agent_name}_{self.memory_id_counter}") # Include agent name in ID
            if 'id' not in data: # Only increment if we generated it
                self.memory_id_counter += 1

            memory_record = {
                'id': memory_id,
                'type': actual_memory_type_value,
                'data': data, # data now includes sim_day and sim_hour
                'simulation_time': current_sim_time, # Redundant with data.time but good for quick access
                'recorded_at_real_time': datetime.now().isoformat()
            }
            
            self.memories[agent_name].append(memory_record)

            # Memory pruning logic (FIFO)
            if len(self.memories[agent_name]) > self.memory_limit:
                self.memories[agent_name].pop(0) # Remove the oldest memory
        
    def get_memories_for_day(self, agent_name: str, current_sim_time: int, memory_type_key: Optional[str] = None) -> List[Dict]:
        """Get all memories for a specific agent on a specific simulation day (thread-safe for read).
           memory_type_key should be one of the keys from MEMORY_TYPES dictionary if specified.
        """
        target_day = (current_sim_time // 24) + 1
        actual_memory_type_value = MEMORY_TYPES.get(memory_type_key) if memory_type_key else None

        agent_day_memories = []
        with self._lock: 
            if agent_name not in self.memories:
                return []
            for memory in self.memories[agent_name]:
                # memory['data'] should reliably have 'simulation_day' due to add_memory logic
                if memory.get('data', {}).get('simulation_day') == target_day:
                    if actual_memory_type_value is None or memory['type'] == actual_memory_type_value:
                        agent_day_memories.append(memory)
        return agent_day_memories
        
    def retrieve_memories(self, agent_name: str, current_time: int, 
                         memory_type_key: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve recent memories for an agent (thread-safe for read).
           memory_type_key should be one of the keys from MEMORY_TYPES dictionary if specified.
        """
        actual_memory_type_value = MEMORY_TYPES.get(memory_type_key) if memory_type_key else None

        with self._lock:
            if agent_name not in self.memories:
                return []
            
            relevant_memories = []
            for memory in self.memories.get(agent_name, []):
                # Check type first
                if actual_memory_type_value and memory.get('type') != actual_memory_type_value:
                    continue
                # Then check time
                memory_sim_time = memory.get('simulation_time') # Direct access is fine
                if isinstance(memory_sim_time, (int, float)) and memory_sim_time <= current_time:
                    relevant_memories.append(memory)
        
        # Sort by simulation_time descending (most recent first)
        sorted_memories = sorted(relevant_memories, 
                               key=lambda x: x.get('simulation_time', 0),
                               reverse=True)
        
        if limit:
            return sorted_memories[:limit]
        return sorted_memories
        
    def save_to_file(self, filename: str):
        """Save all memories to a JSONL file (should be called when sim is paused or at end)."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        records_dir = os.path.join(base_dir, 'LLMAgentsTown_memory_records', 'simulation_memory')
        os.makedirs(records_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(records_dir, f'memories_{timestamp}.jsonl')
        
        with self._lock: # Ensure thread safety while reading self.memories
            with open(filepath, 'w') as f:
                for agent_name, memories_list in self.memories.items():
                    for memory in memories_list:
                        record = {
                            'agent': agent_name,
                            'type': memory['type'], # This is already the string value
                            'simulation_time': memory['simulation_time'],
                            'simulation_day': (memory['simulation_time'] // 24) + 1,
                            'simulation_hour': memory['simulation_time'] % 24,
                            'data': memory['data']
                        }
                        f.write(json.dumps(record) + '\n')
        
        print(f"\nMemories saved to: {filepath}")
        return filepath

    def timestep_to_realtime(self, timestep):
        """Convert simulation timestep to datetime"""
        # Assuming timestep is in hours for the simulation
        return self.start_time + datetime.timedelta(hours=timestep)
                
    def get_important_memories(self, agent_name, importance_threshold=0.7):
        """Get memories above a certain importance threshold (assumes 'importance' in data)."""
        # This method might need adjustment based on how 'importance' is defined and stored
        # With generic events, 'importance' might not always be present.
        important_memories = []
        with self._lock:
            if agent_name not in self.memories: return []
            for memory in self.memories[agent_name]:
                if memory.get('data', {}).get('importance', 0.0) >= importance_threshold:
                    important_memories.append(memory)
        return important_memories

    def get_food_related_memories(self, agent_name, recency_weight=0.7, relevance_weight=0.3):
        """Get memories related to food with weighted scoring (operates on memory['data']['content'])."""
        food_keywords = ["food", "eat", "meal", "restaurant", "chicken", "hungry", "lunch", "dinner", "grocery", "cook", "purchase", "buy"]
        
        scored_memories = []
        current_real_time = datetime.now()
        
        with self._lock:
            if agent_name not in self.memories: return []

            for memory in self.memories[agent_name]:
                memory_content = ""
                # Try to get content from common places
                if 'content' in memory.get('data', {}):
                    memory_content = str(memory['data']['content'])
                elif 'description' in memory.get('data', {}): # For ACTIVITY_EVENT
                    memory_content = str(memory['data']['description'])
                elif memory['type'] == MEMORY_TYPES['FOOD_PURCHASE_EVENT']:
                    memory_content = str(memory['data'].get('item_description', '')) + " food purchase"
                elif memory['type'] == MEMORY_TYPES['GROCERY_PURCHASE_EVENT']:
                    memory_content = str(memory['data'].get('items_list', '')) + " grocery purchase"

                if not memory_content: continue # Skip if no relevant text found
                
                is_food_related = any(keyword in memory_content.lower() for keyword in food_keywords)
                
                if is_food_related:
                    time_diff_seconds = (current_real_time - datetime.fromisoformat(memory['recorded_at_real_time'])).total_seconds()
                    recency_score = max(0, 1 - (time_diff_seconds / (7 * 24 * 60 * 60))) # Within last week
                    
                    keyword_count = sum(1 for keyword in food_keywords if keyword in memory_content.lower())
                    relevance_score = min(1.0, keyword_count / 3) # Cap at 1.0 for up to 3 keywords
                    
                    total_score = (recency_weight * recency_score) + (relevance_weight * relevance_score)
                    scored_memories.append((memory, total_score))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in scored_memories]

    def mark_memory_as_planned(self, agent_name, memory_id):
        """Mark a memory as having generated a plan"""
        with self._lock:
            if agent_name not in self.memories: return
            for memory in self.memories[agent_name]:
                if memory.get('id') == memory_id:
                    memory.setdefault('data', {})['plan_generated_from_this_memory'] = True
                    break

    def calculate_memory_score(self, memory_record: Dict, current_sim_time: int, context_text: Optional[str] = None):
        """Calculate unified memory score for a given memory record."""
        score = memory_record.get('data', {}).get('importance', 0.5) # Base importance
        
        memory_sim_time = memory_record.get('simulation_time', current_sim_time)
        time_diff_hours = max(0, current_sim_time - memory_sim_time) # Ensure non-negative diff
        recency_score = 1.0 / (1 + (time_diff_hours / 24.0)) 
        score += recency_score * 0.3 
        
        # Semantic relevance if context is provided
        # Check if memory has textual content for comparison
        memory_content_for_relevance = None
        if 'content' in memory_record.get('data', {}):
            memory_content_for_relevance = str(memory_record['data']['content'])
        elif 'description' in memory_record.get('data', {}): # for ACTIVITY_EVENT
            memory_content_for_relevance = str(memory_record['data']['description'])
        
        if context_text and memory_content_for_relevance:
            relevance = self.calculate_semantic_similarity(
                memory_content_for_relevance,
                context_text
            )
            score += relevance * 0.2
        
        # Bonus for certain types - now using the actual string value from MEMORY_TYPES
        # No STORE_VISIT_EVENT or RECOMMENDATION_EVENT in new simplified list
        # This bonus might be removed or rethought for new types if applicable.
        # memory_type_value = memory_record.get('type')
        # if memory_type_value in [MEMORY_TYPES.get('SOME_NEW_IMPORTANT_TYPE')]: 
        #     score += 0.1
        
        return min(1.0, max(0.0, score)) # Ensure score is between 0 and 1

    def get_today_memories(self, agent_name: str, current_sim_time: int) -> List[Dict]:
        """Get all memories from the current simulation day for a specific agent."""
        current_sim_day = (current_sim_time // 24) + 1
        memories_today = []
        with self._lock:
            if agent_name not in self.memories: return []
            for memory in self.memories[agent_name]:
                if memory.get('data',{}).get('simulation_day') == current_sim_day:
                    memories_today.append(memory)
        return memories_today

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (simple word overlap)."""
        if not text1 or not text2: return 0.0 # Handle empty strings
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        if not words1 or not words2: return 0.0
        intersection_len = len(words1.intersection(words2))
        union_len = len(words1.union(words2))
        return intersection_len / union_len if union_len > 0 else 0.0

    def save_memories(self, filepath: Optional[str] = None) -> None:
        """Save all agent memories to a single JSON file. This should be thread-safe for reading memories."""
        memories_to_save = {}
        
        with self._lock:
            # Create a deep copy to avoid issues if memories are modified during iteration by another thread (less likely here but good practice)
            # For this specific case, self.memories[agent_name] is a list of dicts. dicts are mutable.
            # A simple list(agent_memory_list) creates a shallow copy of the list.
            # We are mostly concerned about the list structure itself changing during iteration if not locked.
            # The lock already prevents modification of self.memories structure.
            for agent_name, agent_memory_list in self.memories.items():
                memories_to_save[agent_name] = {
                    'memories': [dict(mem) for mem in agent_memory_list], # Create shallow copies of each memory dict
                    'total_memories': len(agent_memory_list)
                }

        try:
            actual_filepath = filepath
            if not actual_filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir) 
                memories_dir = os.path.join(project_root, "LLMAgentsTown_memory_records", "simulation_agents")
                os.makedirs(memories_dir, exist_ok=True)
                actual_filepath = os.path.join(memories_dir, f"consolidated_memories_{timestamp}.json")

            output_data = {
                'saved_at_timestamp': datetime.now().isoformat(),
                'agents_memories': memories_to_save
            }

            with open(actual_filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nConsolidated agent memories saved to: {actual_filepath}")

        except Exception as e:
            print(f"Error saving consolidated memories: {str(e)}")
            traceback.print_exc()