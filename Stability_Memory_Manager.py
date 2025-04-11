import json
import os
from datetime import datetime

class MemoryManager:
    def __init__(self, memory_limit=200, time_unit=0.5):  # 1 unit = 0.5 seconds
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
        """Add a new memory for an agent"""
        if agent_name not in self.memories:
            self.memories[agent_name] = []
            
        current_time = datetime.now()
        
        if event_type == "store_visit":
            # Expects specific format for store visits
            satisfaction_data = {
                'overall_rating': details.get('satisfaction', None),
                'food_quality': details.get('food_quality', None),
                'price_satisfaction': details.get('price_satisfaction', None),
                'service': details.get('service_rating', None),
                'wait_time': details.get('wait_time', None),
                'specific_feedback': details.get('feedback', ''),
                'would_recommend': details.get('would_recommend', None),
                'return_intention': details.get('return_intention', None)
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
                'importance': self.determine_memory_importance(agent_name, details.get('experience', ''))
            }
            
            self.memories[agent_name].append(memory)
            self.memory_id_counter += 1

        elif event_type == "received_recommendation":
            # Expects specific format for recommendations
            memory = {
                'id': self.memory_id_counter,
                'type': 'received_info',
                'source': details['from_agent'],
                'content': details['message'],
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M'),
                'acted_upon': False,
                'feedback_given': False,
                'importance': self.determine_memory_importance(agent_name, details['message'])
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
                'importance': self.determine_memory_importance(agent_name, details['message'])
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
            
        # Consolidate memories if over limit
        if len(self.memories[agent_name]) > self.memory_limit:
            self.consolidate_memories(agent_name)

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
                memory['importance'] = self.determine_memory_importance(agent_name, 
                    f"{memory['experience']} {memory['satisfaction']['specific_feedback']}")
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

    def determine_memory_importance(self, agent_name, content):
        """Determine memory importance"""
        # For now, use a simple heuristic
        importance = 0.5  # default importance
        
        # Increase importance for certain keywords
        keywords = ['discount', 'delicious', 'terrible', 'amazing', 'recommend']
        content_lower = content.lower()
        for keyword in keywords:
            if keyword in content_lower:
                importance += 0.1
        
        return min(max(importance, 0.1), 1.0)

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
        
        # Sort by importance and recency
        memories.sort(key=lambda x: (x.get('importance', 0.5), x['timestamp']), reverse=True)
        
        # Keep most important memories
        self.memories[agent_name] = memories[:self.memory_limit]

    def save_to_file(self):
        """Save memories in JSONL format"""
        try:
            # Current path
            base_dir = "/Users/carolchu/Desktop/AI_Village/LLMAgentsTown_Stability"
            records_dir = os.path.join(base_dir, 'memory_records')
            os.makedirs(records_dir, exist_ok=True)

            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
            
            # Saves to: /memory_records/simulation_agents/agent_memories_{timestamp}.jsonl
            agents_file = os.path.join(records_dir, 'simulation_agents', f'agent_memories_{timestamp}.jsonl')
            os.makedirs(os.path.dirname(agents_file), exist_ok=True)
            
            with open(agents_file, 'w') as f:
                # Write metadata
                f.write(json.dumps({
                    'type': 'metadata',
                    'simulation_start': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'simulation_end': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }) + '\n')
                
                # Write memories for each agent
                for agent_name, memories in self.memories.items():
                    if memories:  # Only write if there are memories
                        grouped_memories = {
                            'type': 'agent_memories',
                            'agent': agent_name,
                            'store_visits': [m for m in memories if m['type'] == 'store_visit'],
                            'received_info': [m for m in memories if m['type'] == 'received_info'],
                            'shared_info': [m for m in memories if m['type'] == 'shared_info']
                        }
                        f.write(json.dumps(grouped_memories) + '\n')

            print(f"\nAgent memories saved to: {agents_file}")
            return agents_file
            
        except Exception as e:
            print(f"Error saving memories: {str(e)}")
            return None

    def timestep_to_realtime(self, timestep):
        """Convert a timestep to real datetime"""
        return self.start_time + datetime.timedelta(seconds=timestep * self.time_unit)

    def get_recent_memories(self, agent_name, current_time, time_window=24):
        """Get recent memories within a time window, sorted by importance"""
        if agent_name not in self.memories:
            return []
            
        recent_memories = [
            memory for memory in self.memories[agent_name]
            if current_time - memory['timestamp'] <= time_window
        ]
        
        # Sort by importance and timestamp
        return sorted(recent_memories, 
                    key=lambda x: (x['importance'], x['timestamp']),
                    reverse=True)
                    
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

    def get_family_schedule(self, agent_name, current_time):
        """Get family members' schedules from recent memories"""
        day = (current_time // 24) + 1
        family_members = self.get_family_members(agent_name)
        family_schedule = {}
        
        # Look through recent memories (last 24 hours) for schedule information
        for member in family_members:
            member_memories = [m for m in self.memories[member] 
                             if current_time - m['time'] <= 24]
            schedule_info = {}
            for memory in member_memories:
                if "going to work" in memory['content']:
                    schedule_info['work'] = (9, 17)  # Example work hours
                elif "at school" in memory['content']:
                    schedule_info['school'] = (8, 15)  # Example school hours
            family_schedule[member] = schedule_info
        
        return family_schedule