# Simulation settings define global parameters for the simulation
SIMULATION_SETTINGS = {
    'simulation': {
        'duration_days': 7,      # Duration of the simulation in days
        'time_step': 60,         # Time step in minutes (e.g., 60 for hourly updates)
        'output_frequency': 24   # How often to output summaries (e.g., every 24 hours)
    },
    'agent': {
        'max_memory_size': 1000, # Maximum number of memories an agent can store
        'interaction_range': 5  # Range within which agents can interact (if using spatial grid)
    },
    'logging': {
        'log_level': 'INFO',     # Logging level (DEBUG, INFO, WARNING, ERROR)
        'log_to_file': True      # Whether to log to a file
    },
    'day_start_hour': 7, # Simulation day processing starts at 7 AM
    'day_end_hour': 24,  # Simulation day processing ends at midnight (hour 23 is last active hour)
    'planning_hour': 7   # Hour at which daily plans are made
}

# Simplified Activity Types Constants as per user request
ACTIVITY_TYPES = {
    'TRAVEL': 'travel',                     # Moving between locations
    'WORK': 'work',                         # Performing job-related tasks (can be at home or a workplace)
    'RESTING': 'resting',                   # Sleeping, relaxing, passive time at home
    'GROCERY_PURCHASE': 'grocery_purchase', # The specific act of buying groceries
    'FOOD_PURCHASE': 'food_purchase',       # The specific act of buying prepared food
    'EDUCATION': 'education',               # Learning, attending classes
    'RECREATION': 'recreation',             # Leisure, sports, entertainment
    'CONVERSATION': 'conversation'          # Engaging in any form of dialogue
}

# Simplified Memory and Record Types as per user request and discussion
MEMORY_TYPES = {
    # Essential LLM & Simulation Mechanics
    'ACTION_RAW_OUTPUT': 'action_raw_output',         # Raw LLM text output for agent's decided action/thought
    'PLANNING_EVENT': 'planning_event',               # Agent generated/updated its daily plan
    'AGENT_STATE_UPDATE_EVENT': 'agent_state_update_event', # Periodic log of agent's core status (energy, etc.)
    'SYSTEM_EVENT': 'system_event',                   # Simulation internal messages, errors, major phase changes

    # Core Agent Actions/Events
    'LOCATION_CHANGE_EVENT': 'location_change_event', # Agent arrived at a new Location object (not just grid step)
    'CONVERSATION_LOG_EVENT': 'conversation_log_event', # Log of dialogue content. Data field includes participants, location, etc.
    'FOOD_PURCHASE_EVENT': 'food_purchase_event',     # Event of purchasing prepared food
    'GROCERY_PURCHASE_EVENT': 'grocery_purchase_event',# Event of purchasing groceries
    
    # Broad Activity Logging
    'ACTIVITY_EVENT': 'activity_event',               # Generic log for an agent engaging in a primary activity from ACTIVITY_TYPES (e.g., Work, Education, Recreation, Resting).
                                                      # The 'data' field would include:
                                                      #   - 'activity_type_tag': (e.g., ACTIVITY_TYPES['WORK'])
                                                      #   - 'description': (e.g., "Worked on project report", "Watched a movie", "Slept")
                                                      #   - 'duration_hours': (optional, if applicable)
    
    'GENERIC_EVENT': 'generic_event'                  # For anything else not fitting above (travel steps, mid-travel decisions, non-purchase store visits etc.)
                                                      # The 'data' field includes 'content' detailing the event.
} 