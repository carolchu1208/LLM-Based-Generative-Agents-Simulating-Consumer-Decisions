# Saving Strategy Recommendations

## **🎯 Recommended Saving Frequencies**

### **1. Agent Memories**
```python
# RECOMMENDED: Coordinated saving strategy
MEMORY_SAVE_FREQUENCY = {
    'buffer_threshold': 200,        # Reduced from 450 (more frequent saves)
    'time_interval': 1800,          # 30 minutes (reduced from 5 minutes)
    'hourly_save': True,            # Save every hour regardless
    'end_of_day': True,             # Force save at end of day
    'emergency_save': True          # Save on interrupt/error
}
```

**Content Generated:**
- Activities (work, travel, rest, shopping)
- State updates (energy, location, money)
- Plans and planning decisions
- Purchases and financial transactions
- Encounters and social interactions
- Agent relationship changes

**Saving Triggers:**
- ✅ **Every hour** (consistent)
- ✅ **Buffer reaches 200 items** (reduced threshold)
- ✅ **End of day** (force save)
- ✅ **Simulation interrupt** (emergency save)

### **2. Metrics**
```python
# RECOMMENDED: Less frequent, more efficient
METRICS_SAVE_FREQUENCY = {
    'hourly_record': True,          # Record every hour (in memory)
    'save_interval': 7200,          # Save to file every 2 hours (reduced from 4h)
    'end_of_day': True,             # Force save at end of day
    'emergency_save': True          # Save on interrupt/error
}
```

**Content Generated:**
- Energy levels per agent per hour
- Grocery levels per agent per hour
- Financial states (money, income, expenses)
- Location visit records
- Purchase records and transaction details
- Shop metrics (revenue, customer count)
- Customer metrics (spending patterns)

**Saving Triggers:**
- ✅ **Record every hour** (in memory)
- ✅ **Save to file every 2 hours** (reduced frequency)
- ✅ **End of day** (force save)
- ✅ **Simulation interrupt** (emergency save)

### **3. Daily Summaries**
```python
# RECOMMENDED: Keep current frequency (appropriate)
DAILY_SUMMARY_FREQUENCY = {
    'generate_hourly': True,        # Generate summary every hour (in memory)
    'save_end_of_day': True,        # Save to file only at end of day
    'emergency_save': True          # Save on interrupt/error
}
```

**Content Generated:**
- Agent memory summaries by type
- Daily totals and trends
- Relationship changes and patterns
- Activity pattern analysis
- Financial summaries
- Energy and grocery level trends

**Saving Triggers:**
- ✅ **Generate every hour** (in memory)
- ✅ **Save to file at end of day only** (appropriate frequency)
- ✅ **Simulation interrupt** (emergency save)

### **4. Conversations**
```python
# RECOMMENDED: Batch saving strategy
CONVERSATION_SAVE_FREQUENCY = {
    'buffer_threshold': 50,         # Save when 50 conversations accumulated
    'time_interval': 3600,          # Save every hour regardless
    'end_of_day': True,             # Force save at end of day
    'emergency_save': True          # Save on interrupt/error
}
```

**Content Generated:**
- Agent-to-agent conversation logs
- Encounter details and outcomes
- Social interaction patterns
- Relationship building events
- Topic analysis and sentiment

**Saving Triggers:**
- ✅ **Buffer reaches 50 conversations** (batch saving)
- ✅ **Every hour** (time-based backup)
- ✅ **End of day** (force save)
- ✅ **Simulation interrupt** (emergency save)

## **🔄 Coordinated Saving Strategy**

### **Hour-End Processing (Every Hour)**
```python
def _process_hour_end(self, current_day: int, current_hour: int):
    # 1. Record metrics (in memory only)
    self.metrics_mgr.record_hour_metrics(current_day, current_hour, agents)
    
    # 2. Update memory manager time
    self.memory_mgr.update_simulation_time(current_day, current_hour)
    
    # 3. Check if we should save based on coordinated schedule
    self._coordinated_save_check(current_day, current_hour)
```

### **Coordinated Save Check**
```python
def _coordinated_save_check(self, current_day: int, current_hour: int):
    """Coordinated saving strategy to prevent I/O bottlenecks."""
    
    # Always save memories if buffer is full
    if len(self.memory_mgr._pending_memories) >= 200:
        self.memory_mgr.save_memories(force=True)
    
    # Save conversations if buffer is full
    if len(self.conversation_mgr._pending_conversations) >= 50:
        self.conversation_mgr.save_conversation_logs()
    
    # Save metrics every 2 hours (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
    if current_hour % 2 == 0:
        self.metrics_mgr.save_metrics()
    
    # Save memories every hour as backup
    self.memory_mgr.save_memories(force=True)
```

### **End-of-Day Processing**
```python
def _process_end_of_day(self):
    current_day = TimeManager.get_current_day()
    
    # 1. Save all pending data
    self.memory_mgr.save_memories(force=True)
    self.metrics_mgr.save_metrics(force=True)
    self.conversation_mgr.save_conversation_logs()
    
    # 2. Generate and save daily summaries
    self.memory_mgr.save_daily_summaries(self.metrics_mgr, current_day)
    
    # 3. Clear buffers for next day
    self.memory_mgr._pending_memories.clear()
    self.metrics_mgr.clear_daily_metrics()
```

## **📈 Performance Benefits**

### **Before (Current Issues)**
- ❌ Memory saves: Inconsistent (450 buffer + 4h + events)
- ❌ Metrics saves: Too frequent (every 4h)
- ❌ Conversations: Per-conversation saves
- ❌ No coordination: I/O bottlenecks

### **After (Recommended)**
- ✅ Memory saves: Every hour + 200 buffer threshold
- ✅ Metrics saves: Every 2 hours (appropriate frequency)
- ✅ Conversations: Batched (50 conversations or hourly)
- ✅ Coordinated saves: Prevents I/O bottlenecks
- ✅ Consistent timing: Predictable save patterns

## **💾 File Organization**

### **Recommended File Structure**
```
LLMAgentsTown_memory_records/
├── simulation_agents/
│   └── agents_memories_YYYYMMDD_HHMMSS.json      # Hourly saves
├── simulation_metrics/
│   └── metrics_YYYYMMDD_HHMMSS.json              # Every 2 hours
├── simulation_daily_summaries/
│   └── daily_summary_YYYYMMDD_HHMMSS.json        # End of day
└── simulation_conversations/
    └── conversations_YYYYMMDD_HHMMSS.json        # Hourly/batched
```

## **⚙️ Implementation Priority**

1. **HIGH**: Update memory buffer threshold (450 → 200)
2. **HIGH**: Reduce metrics save frequency (4h → 2h)
3. **MEDIUM**: Implement conversation batching
4. **MEDIUM**: Add coordinated save check
5. **LOW**: Optimize file naming conventions

This strategy provides consistent, efficient, and coordinated data saving while maintaining data integrity and preventing I/O bottlenecks. 