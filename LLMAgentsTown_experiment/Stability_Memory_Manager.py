import json
import os
from datetime import datetime
import time

class MemoryManager:
    def __init__(self, memory_limit=1000, time_unit=0.5):  # 1 unit = 0.5 seconds
        self.memory_limit = memory_limit
        self.memories = {}
        self.compressed_memories = {}
        self.memory_ratings = {}
        self.start_time = datetime.now()
        self.time_unit = time_unit
        self.consolidation_threshold = 0.3  # Threshold for memory consolidation
        self.memory_id_counter = 0  # Counter for unique memory IDs
        print(f"Simulation started at: {self.start_time.strftime('%H:%M:%S')}")

    def add_memory(self, agent_name, event_type, details):
        """Record agent memories of interactions and purchases"""
        if agent_name not in self.memories:
            self.memories[agent_name] = []
            
        current_time = datetime.now()
        
        # Family planning and routine memories
        if event_type == "family_planning":
            memory = {
                'id': self.memory_id_counter,
                'type': 'family_planning',
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'schedule': details.get('schedule', {}),
                'participants': details.get('participants', []),
                'conversation': details.get('conversation', ''),
                'decisions': {
                    'school_transport': details.get('school_transport', {}),
                    'dinner_plans': details.get('dinner_plans', {}),
                    'evening_activities': details.get('evening_activities', {})
                },
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
            
        elif event_type == "family_meal":
            memory = {
                'id': self.memory_id_counter,
                'type': 'family_meal',
                'meal_type': details.get('meal_type', 'breakfast'),  # breakfast, lunch, dinner
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'location': details.get('location', 'residence'),
                'participants': details.get('participants', []),
                'conversation': details.get('conversation', ''),
                'coordinated_schedule': details.get('coordinated_schedule', {}),
                'discussion_points': details.get('discussion_points', {}),
                'grocery_used': details.get('grocery_used', 0),
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
            
        elif event_type == "daily_planning":
            memory = {
                'id': self.memory_id_counter,
                'type': 'daily_planning',
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'plan': details.get('plan', {}),
                'is_first_day': details.get('is_first_day', False),
                'coordinated_with_family': details.get('coordinated_with_family', False),
                'morning_routine': details.get('morning_routine', {}),
                'school_schedule': details.get('school_schedule', {}),
                'work_schedule': details.get('work_schedule', {}),
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
            
        elif event_type == "morning_routine":
            memory = {
                'id': self.memory_id_counter,
                'type': 'morning_routine',
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'activity': details.get('activity', ''),
                'location': details.get('location', ''),
                'with_parent': details.get('with_parent', False),
                'supervised': details.get('supervised', False),
                'breakfast_status': details.get('breakfast_status', {}),
                'school_prep': details.get('school_prep', {}),
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
            
        elif event_type == "schedule_coordination":
            memory = {
                'id': self.memory_id_counter,
                'type': 'schedule_coordination',
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'coordinated_with': details.get('coordinated_with', []),
                'updated_schedule': details.get('updated_schedule', {}),
                'transport_arrangements': details.get('transport_arrangements', {}),
                'meal_plans': details.get('meal_plans', {}),
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
        
        # Handle existing memory types
        elif event_type == "store_visit":
            # Enhanced satisfaction tracking
            satisfaction_data = {
                'overall_rating': details.get('satisfaction', None),  # 1-5 scale
                'food_quality': details.get('food_quality', None),    # 1-5 scale
                'price_satisfaction': details.get('price_satisfaction', None),  # 1-5 scale
                'service': details.get('service_rating', None),       # 1-5 scale
                'wait_time': details.get('wait_time', None),         # in minutes
                'specific_feedback': details.get('feedback', ''),     # text feedback
                'would_recommend': details.get('would_recommend', None),  # boolean
                'return_intention': details.get('return_intention', None)  # 1-5 scale
            }

            memory = {
                'id': self.memory_id_counter,
                'type': 'store_visit',
                'location': 'Fried Chicken Shop',
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'visit_number': self.get_visit_count(agent_name) + 1,
                'experience': details.get('experience', ''),
                'who_recommended': details.get('recommended_by', None),
                'shared_with': [],  # Will be updated when agent shares experience
                'price_paid': details.get('price_paid', None),
                'was_discount': details.get('used_discount', False),
                'satisfaction': satisfaction_data,
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
            
            self.memories[agent_name].append(memory)
            self.memory_id_counter += 1

        elif event_type == "received_recommendation":
            memory = {
                'id': self.memory_id_counter,
                'type': 'received_info',
                'source': details['from_agent'],
                'content': details['message'],
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'acted_upon': False,  # Will be updated if they visit
                'feedback_given': False,  # Will be updated if they report back
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
            
            self.memories[agent_name].append(memory)
            self.memory_id_counter += 1
            
        elif event_type == "shared_experience":
            memory = {
                'id': self.memory_id_counter,
                'type': 'shared_info',
                'told_to': details['to_agent'],
                'content': details['message'],
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'about_visit': details.get('visit_number'),
                'response': details.get('response', None),
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
            
            # Update the original visit memory's shared_with list
            for m in reversed(self.memories[agent_name]):
                if m['type'] == 'store_visit' and m['visit_number'] == details.get('visit_number'):
                    m['shared_with'].append(details['to_agent'])
                    break
                    
            self.memories[agent_name].append(memory)
            self.memory_id_counter += 1
            
        elif event_type == "word_of_mouth":
            # Need to ensure details contains 'sentiment' and 'listener'
            if not details or 'sentiment' not in details or 'listener' not in details:
                details = {
                    'sentiment': 'neutral',
                    'listener': 'unknown',
                    'content': str(details)  # Convert old format to new
                }
            
        elif event_type == "social":
            # Add specific handling for food-related social interactions
            memory = {
                'id': self.memory_id_counter,
                'type': 'social',
                'subtype': details.get('topic', 'general'),  # 'food', 'general', etc.
                'location': details['location'],
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'content': details['content'],
                'participants': details.get('participants', []),
                'influence': details.get('influence', False),
                'food_related': 'food' in details.get('topic', '').lower(),
                'plan_generated': False,  # Track if this memory led to a plan
                'importance': self.determine_memory_importance(agent_name, event_type, details)
            }
            
            self.memories[agent_name].append(memory)
            self.memory_id_counter += 1

        # Add grocery-specific memory handling
        elif event_type == "grocery_update":
            memory = {
                'id': self.memory_id_counter,
                'type': 'grocery_update',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'content': details['content'],
                'grocery_level': details.get('grocery_level', 0),
                'location': details.get('location'),
                'importance': 0.6 if details.get('grocery_level', 100) <= 30 else 0.3  # Higher importance when low
            }
            
            self.memories[agent_name].append(memory)
            self.memory_id_counter += 1

        # Add the memory to the agent's memory list
        self.memories[agent_name].append(memory)
        self.memory_id_counter += 1
        
        # Consolidate memories if over limit
        if len(self.memories[agent_name]) > self.memory_limit:
            self.consolidate_memories(agent_name)

    def add_store_visit_memory(self, agent_name, details):
        """Handle store visit memories"""
        # Get visit history
        previous_visits = self.get_visit_history(agent_name)
        
        visit_details = {
            **details,  # Include basic details
            'type': 'store_visit',
            'visit_number': len(previous_visits) + 1,
            'is_return_customer': len(previous_visits) > 0,
            'last_visit_time': previous_visits[-1]['time'] if previous_visits else None,
            'satisfaction': self.initialize_satisfaction_tracking()
        }
        
        importance = self.determine_memory_importance(agent_name, "store_visit", visit_details)
        visit_details['importance'] = importance
        
        self.add_memory(agent_name, "store_visit", visit_details)

    def add_social_memory(self, agent_name, details):
        """Handle social interaction memories"""
        # Extract participants
        content = details['content']
        participants = []
        if "with" in content:
            participants = [name.strip() for name in content.split("with")[1].split("and")]
            
        social_details = {
            **details,
            'type': 'social',
            'participants': participants,
            'food_related': 'food' in content.lower() or 'restaurant' in content.lower(),
            'about_location': 'Fried Chicken Shop' if 'Fried Chicken Shop' in content else None
        }
        
        importance = self.determine_memory_importance(agent_name, "social", social_details)
        social_details['importance'] = importance
        
        self.add_memory(agent_name, "social", social_details) 
        
    def get_visit_count(self, agent_name):
        """Get the number of times an agent has visited the shop"""
        return len([m for m in self.memories.get(agent_name, []) 
                   if m['type'] == 'store_visit' and 
                   m['location'] == 'Fried Chicken Shop'])

    def get_information_chain(self, agent_name):
        """Get who told this agent about the shop and who they told"""
        received_from = None
        told_to = []
        
        for memory in self.memories.get(agent_name, []):
            if memory['type'] == 'received_info':
                received_from = memory['source']
            elif memory['type'] == 'shared_info':
                told_to.append(memory['told_to'])
                
        return {
            'received_from': received_from,
            'shared_with': told_to
        }

    def update_visit_feedback(self, agent_name, visit_number, feedback_details):
        """Update detailed feedback for a specific visit"""
        for memory in reversed(self.memories.get(agent_name, [])):
            if memory['type'] == 'store_visit' and memory['visit_number'] == visit_number:
                # Update satisfaction data
                memory['satisfaction'].update({
                    'overall_rating': feedback_details.get('overall_rating', memory['satisfaction']['overall_rating']),
                    'food_quality': feedback_details.get('food_quality', memory['satisfaction']['food_quality']),
                    'price_satisfaction': feedback_details.get('price_satisfaction', memory['satisfaction']['price_satisfaction']),
                    'service': feedback_details.get('service_rating', memory['satisfaction']['service']),
                    'wait_time': feedback_details.get('wait_time', memory['satisfaction']['wait_time']),
                    'specific_feedback': feedback_details.get('feedback', memory['satisfaction']['specific_feedback']),
                    'would_recommend': feedback_details.get('would_recommend', memory['satisfaction']['would_recommend']),
                    'return_intention': feedback_details.get('return_intention', memory['satisfaction']['return_intention'])
                })
                
                # Update memory importance based on new feedback
                memory['importance'] = self.determine_memory_importance(agent_name, event_type, feedback_details)
                break

    def get_satisfaction_history(self, agent_name):
        """Get satisfaction trends for an agent's visits"""
        satisfaction_history = []
        for memory in self.memories.get(agent_name, []):
            if memory['type'] == 'store_visit':
                satisfaction_history.append({
                    'visit_number': memory['visit_number'],
                    'timestamp': memory['timestamp'],
                    'was_discount': memory['was_discount'],
                    'satisfaction': memory['satisfaction'],
                    'price_paid': memory['price_paid']
                })
        return satisfaction_history

    def get_satisfaction_summary(self, agent_name):
        """Get a summary of an agent's satisfaction with the shop"""
        visits = [m for m in self.memories.get(agent_name, []) 
                 if m['type'] == 'store_visit']
        
        if not visits:
            return None

        total_ratings = {
            'overall_rating': [],
            'food_quality': [],
            'price_satisfaction': [],
            'service': [],
            'wait_time': [],
            'would_recommend': 0,
            'total_visits': len(visits)
        }

        for visit in visits:
            sat = visit['satisfaction']
            if sat['overall_rating']:
                total_ratings['overall_rating'].append(sat['overall_rating'])
            if sat['food_quality']:
                total_ratings['food_quality'].append(sat['food_quality'])
            if sat['price_satisfaction']:
                total_ratings['price_satisfaction'].append(sat['price_satisfaction'])
            if sat['service']:
                total_ratings['service'].append(sat['service'])
            if sat['wait_time']:
                total_ratings['wait_time'].append(sat['wait_time'])
            if sat['would_recommend']:
                total_ratings['would_recommend'] += 1

        # Calculate averages
        summary = {
            'average_overall': sum(total_ratings['overall_rating']) / len(total_ratings['overall_rating']) if total_ratings['overall_rating'] else None,
            'average_food_quality': sum(total_ratings['food_quality']) / len(total_ratings['food_quality']) if total_ratings['food_quality'] else None,
            'average_price_satisfaction': sum(total_ratings['price_satisfaction']) / len(total_ratings['price_satisfaction']) if total_ratings['price_satisfaction'] else None,
            'average_service': sum(total_ratings['service']) / len(total_ratings['service']) if total_ratings['service'] else None,
            'average_wait_time': sum(total_ratings['wait_time']) / len(total_ratings['wait_time']) if total_ratings['wait_time'] else None,
            'recommendation_rate': total_ratings['would_recommend'] / total_ratings['total_visits'] if total_ratings['total_visits'] > 0 else 0,
            'total_visits': total_ratings['total_visits']
        }
        
        return summary

    def mark_recommendation_acted_upon(self, agent_name, from_agent):
        """Mark a received recommendation as acted upon when the agent visits"""
        for memory in self.memories.get(agent_name, []):
            if memory['type'] == 'received_info' and memory['source'] == from_agent:
                memory['acted_upon'] = True
                break

    def mark_feedback_given(self, agent_name, to_agent):
        """Mark that feedback was given to someone who made a recommendation"""
        for memory in self.memories.get(agent_name, []):
            if memory['type'] == 'received_info' and memory['source'] == to_agent:
                memory['feedback_given'] = True
                break

    def get_relevant_memories(self, agent_name, context=None, limit=5):
        """Get relevant memories based on context"""
        if agent_name not in self.memories:
            return []
            
        memories = self.memories[agent_name]
        scored_memories = []
        
        for memory in memories:
            # Calculate relevance score
            relevance = 1.0
            if context:
                relevance = self.calculate_semantic_similarity(
                    str(memory.get('content', '') or memory.get('experience', '')),
                    context
                )
            
            # Calculate recency score (newer = higher score)
            try:
                memory_time = datetime.strptime(memory['timestamp'], '%Y-%m-%d %H:%M')
                time_diff = (datetime.now() - memory_time).total_seconds() / 3600  # hours
                recency = 1.0 / (1 + time_diff)
            except:
                recency = 0.5
            
            # Combine scores
            total_score = (relevance + recency + memory.get('importance', 0.5)) / 3
            scored_memories.append((memory, total_score))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored_memories[:limit]]

    def determine_memory_importance(self, agent_name, event_type, details):
        """Calculate memory importance with added family and routine factors"""
        base_importance = 0.5
        importance_score = base_importance
        
        # Family-related importance
        if event_type in ['family_planning', 'family_meal', 'schedule_coordination']:
            importance_score += 0.2  # Family interactions are important
            
            # Additional importance for key decisions
            if details.get('decisions', {}).get('school_transport'):
                importance_score += 0.1
            if details.get('coordinated_schedule'):
                importance_score += 0.1
            
        # Morning routine importance
        if event_type == 'morning_routine':
            if details.get('supervised'):  # Supervised activities for young children
                importance_score += 0.15
            if details.get('breakfast_status'):  # Meal-related memories
                importance_score += 0.1
            if details.get('school_prep'):  # School preparation
                importance_score += 0.1
            
        # Daily planning importance
        if event_type == 'daily_planning':
            if details.get('is_first_day'):  # First day is more memorable
                importance_score += 0.2
            if details.get('coordinated_with_family'):
                importance_score += 0.15
            
        # Cap importance between 0.1 and 1.0
        return min(max(importance_score, 0.1), 1.0)

    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        # Simple word overlap for now
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    def consolidate_memories(self, agent_name):
        """Consolidate memories when over limit"""
        if agent_name not in self.memories:
            return
            
        memories = self.memories[agent_name]
        
        # First, remove low-importance memories
        important_memories = [m for m in memories if m.get('importance', 0.5) >= self.consolidation_threshold]
        low_importance_memories = [m for m in memories if m.get('importance', 0.5) < self.consolidation_threshold]
        
        # If we're still over limit after removing low-importance memories
        if len(important_memories) > self.memory_limit:
            # Sort by importance and recency
            important_memories.sort(key=lambda x: (x.get('importance', 0.5), x['timestamp']), reverse=True)
            important_memories = important_memories[:self.memory_limit]
        
        # Update the agent's memories
        self.memories[agent_name] = important_memories
        
        # Optionally, create a consolidated summary of removed memories
        if low_importance_memories:
            summary = self.create_memory_summary(low_importance_memories)
            if summary:
                self.memories[agent_name].append(summary)

    def create_memory_summary(self, memories):
        """Create a summary memory from multiple low-importance memories"""
        if not memories:
            return None
        
        # Group memories by type
        grouped = {}
        for memory in memories:
            mem_type = memory.get('type', 'general')
            if mem_type not in grouped:
                grouped[mem_type] = []
            grouped[mem_type].append(memory)
        
        # Create summary content
        summary_content = []
        for mem_type, type_memories in grouped.items():
            if mem_type == 'store_visit':
                count = len(type_memories)
                summary_content.append(f"Made {count} unmemorable visits to the store")
            elif mem_type == 'social':
                count = len(type_memories)
                summary_content.append(f"Had {count} routine social interactions")
        
        if not summary_content:
            return None
        
        return {
            'id': self.memory_id_counter,
            'type': 'consolidated_summary',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'content': '. '.join(summary_content),
            'original_count': len(memories),
            'importance': self.consolidation_threshold,  # Set at threshold level
            'is_summary': True
        }

    def save_to_file(self):
        """Save memories in JSONL format with updated path structure"""
        # Get parent directory (LLMAgentsTown_Stability)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Path to records directory
        records_dir = os.path.join(base_dir, 'LLMAgentsTown_memory_records', 'simulation_agents')
        os.makedirs(records_dir, exist_ok=True)

        filename = f"agent_memories_{self.start_time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        filepath = os.path.join(records_dir, filename)

        with open(filepath, 'w') as f:
            # Write simulation metadata as first line
            f.write(json.dumps({
                'type': 'metadata',
                'simulation_start': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'simulation_end': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }) + '\n')
            
            # Write each agent's memories as separate lines
            for agent_name, memories in self.memories.items():
                # Group memories by type for better organization
                grouped_memories = {
                    'type': 'agent_memories',
                    'agent': agent_name,
                    'store_visits': [m for m in memories if m['type'] == 'store_visit'],
                    'received_info': [m for m in memories if m['type'] == 'received_info'],
                    'shared_info': [m for m in memories if m['type'] == 'shared_info']
                }
                f.write(json.dumps(grouped_memories) + '\n')

        print(f"\nAgent memories saved to: {filepath}")
        return filepath

    def timestep_to_realtime(self, timestep):
        """Convert a timestep to real datetime"""
        return self.start_time + datetime.timedelta(seconds=timestep * self.time_unit)

    def get_recent_memories(self, agent_name, current_time=None, time_window=24, limit=None):
        """Get recent memories within a time window, sorted by importance"""
        if agent_name not in self.memories:
            return []
        
        recent_memories = [
            memory for memory in self.memories[agent_name]
            if not current_time or current_time - memory['timestamp'] <= time_window
        ]
        
        # Sort by importance and timestamp
        sorted_memories = sorted(recent_memories, 
                               key=lambda x: (x['importance'], x['timestamp']),
                               reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            return sorted_memories[:limit]
        return sorted_memories
                    
    def update_memory_importance(self, agent_name, memory_id, new_importance):
        """Update the importance of a specific memory"""
        if agent_name not in self.memories:
            return
            
        for memory in self.memories[agent_name]:
            if memory['id'] == memory_id:
                memory['importance'] = min(1.0, new_importance)  # Cap at 1.0
                break
                
    def get_important_memories(self, agent_name, importance_threshold=0.7):
        """Get memories above a certain importance threshold"""
        if agent_name not in self.memories:
            return []
            
        return [
            memory for memory in self.memories[agent_name]
            if memory['importance'] >= importance_threshold
        ]

    def get_food_related_memories(self, agent_name, recency_weight=0.7, relevance_weight=0.3):
        """Get memories related to food with weighted scoring"""
        if agent_name not in self.memories:
            return []
        
        food_keywords = ["food", "eat", "meal", "restaurant", "chicken", "hungry", "lunch", "dinner"]
        
        scored_memories = []
        current_time = time.time()  # Use current time as reference
        
        for memory in self.memories[agent_name]:
            # Check if memory is related to food
            is_food_related = any(keyword in memory['content'].lower() for keyword in food_keywords)
            
            if is_food_related:
                # Calculate recency score (normalized 0-1)
                time_diff = current_time - memory['timestamp']
                recency_score = max(0, 1 - (time_diff / (7 * 24 * 60 * 60)))  # Within last week
                
                # Calculate relevance score based on how many food keywords appear
                keyword_count = sum(1 for keyword in food_keywords if keyword in memory['content'].lower())
                relevance_score = min(1.0, keyword_count / 3)  # Cap at 1.0
                
                # Calculate combined score
                total_score = (recency_weight * recency_score) + (relevance_weight * relevance_score)
                
                scored_memories.append((memory, total_score))
        
        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the memories, not the scores
        return [memory for memory, _ in scored_memories]

    def mark_memory_as_planned(self, agent_name, memory_id):
        """Mark a memory as having generated a plan"""
        if agent_name not in self.memories:
            return
            
        for memory in self.memories[agent_name]:
            if memory['id'] == memory_id:
                memory['plan_generated'] = True
                break

    def retrieve_memories(self, agent_name, current_time, memory_type=None, context=None, limit=5):
        """Get recent memories with optional type and context filtering"""
        if agent_name not in self.memories:
            return []
        
        memories = self.memories[agent_name]
        scored_memories = []
        
        for memory in memories:
            # Apply type filter if specified
            if memory_type and memory.get('type') != memory_type:
                continue
            
            # Calculate comprehensive score
            score = self.calculate_memory_score(memory, current_time, context)
            scored_memories.append((memory, score))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored_memories[:limit]]

    def calculate_memory_score(self, memory, current_time, context=None):
        """Calculate unified memory score"""
        # Base importance
        score = memory.get('importance', 0.5)
        
        # Time decay
        try:
            memory_time = datetime.strptime(memory['timestamp'], '%Y-%m-%d %H:%M')
            time_diff = (datetime.now() - memory_time).total_seconds() / 3600  # Convert to hours
            recency = 1.0 / (1 + (time_diff / 24))  # Decay over days
        except:
            recency = 0.5  # Default if timestamp parsing fails
        
        score += recency * 0.3
        
        # Context relevance if provided
        if context:
            relevance = self.calculate_semantic_similarity(
                memory.get('content', ''),
                context
            )
            score += relevance * 0.2
        
        # Memory type bonus
        if memory.get('type') in ['store_visit', 'received_recommendation']:
            score += 0.1
        
        # Cap final score at 1.0
        return min(1.0, score)

    def get_today_memories(self, agent_name):
        """Get all memories from the current day"""
        if agent_name not in self.memories:
            return []
        
        # Get current day's start time
        current_time = datetime.now()
        day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Filter memories from today
        today_memories = [
            memory for memory in self.memories[agent_name]
            if datetime.strptime(memory['timestamp'], '%Y-%m-%d %H:%M') >= day_start
        ]
        
        return today_memories