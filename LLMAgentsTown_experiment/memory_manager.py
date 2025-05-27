class MemoryManager:
    def __init__(self):
        self.memories = []
        self.memory_types = {
            'ACTIVITY_EVENT': {'weight': 1.0, 'decay_rate': 0.1},
            'PLANNING_EVENT': {'weight': 1.2, 'decay_rate': 0.05},
            'CONVERSATION': {'weight': 1.5, 'decay_rate': 0.08},
            'PURCHASE': {'weight': 1.3, 'decay_rate': 0.07},
            'GENERIC_EVENT': {'weight': 0.8, 'decay_rate': 0.12},
            'SYSTEM_EVENT': {'weight': 0.5, 'decay_rate': 0.15}
        }

    def add_memory(self, agent_name: str, memory_type: str, memory_data: Dict):
        if memory_type not in self.memory_types:
            memory_type = 'GENERIC_EVENT'
            
        memory = {
            'agent_name': agent_name,
            'memory_type': memory_type,
            'timestamp': memory_data.get('time', 0),
            'data': memory_data,
            'importance': self.memory_types[memory_type]['weight']
        }
        self.memories.append(memory)

    def get_recent_memories(self, agent_name: str, current_time: int, limit: int = 5, memory_type: str = None) -> List[Dict]:
        agent_memories = [m for m in self.memories if m['agent_name'] == agent_name]
        
        if memory_type:
            agent_memories = [m for m in agent_memories if m['memory_type'] == memory_type]
            
        # Sort by recency and importance
        agent_memories.sort(key=lambda x: (current_time - x['timestamp'], -x['importance']))
        
        return agent_memories[:limit]

    def get_memories_for_day(self, agent_name: str, current_sim_time: int, memory_type_key: str = None) -> List[Dict]:
        day_start = (current_sim_time // 24) * 24
        day_end = day_start + 24
        
        day_memories = [
            m for m in self.memories 
            if m['agent_name'] == agent_name 
            and day_start <= m['timestamp'] < day_end
            and (not memory_type_key or m['memory_type'] == memory_type_key)
        ]
        
        return sorted(day_memories, key=lambda x: x['timestamp'])

    def get_memories_by_location(self, agent_name: str, location_name: str, current_time: int, limit: int = 5) -> List[Dict]:
        location_memories = [
            m for m in self.memories 
            if m['agent_name'] == agent_name 
            and m.get('data', {}).get('location') == location_name
        ]
        
        location_memories.sort(key=lambda x: (current_time - x['timestamp'], -x['importance']))
        return location_memories[:limit]

    def get_interaction_memories(self, agent_name: str, other_agent_name: str, current_time: int, limit: int = 5) -> List[Dict]:
        interaction_memories = []
        
        for memory in self.memories:
            if memory['agent_name'] != agent_name:
                continue
                
            data = memory.get('data', {})
            if (
                memory['memory_type'] == 'CONVERSATION' 
                and other_agent_name in data.get('participants', [])
            ) or (
                'involved_agents' in data 
                and other_agent_name in data['involved_agents']
            ):
                interaction_memories.append(memory)
        
        interaction_memories.sort(key=lambda x: (current_time - x['timestamp'], -x['importance']))
        return interaction_memories[:limit]

    def clear_old_memories(self, current_time: int, max_age: int = 72):
        self.memories = [
            m for m in self.memories 
            if current_time - m['timestamp'] <= max_age
        ] 