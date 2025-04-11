# Imports first
import ollama
import time
import json
import networkx as nx
import os
from datetime import datetime
import random
from Stability_Metrics import FriedChickenMetrics
from Stability_Memory_Manager import MemoryManager

# Load experiment settings
with open('experiment_settings.json', 'r') as f:
    experiment_settings = json.load(f)

# Initialize global variables
global_time = 0  # Current time
SIMULATION_DURATION_DAYS = 7  # Core simulation parameter
TIME_STEP = experiment_settings['simulation']['time_step']  # minutes

# Add output control flags at the top
PRINT_CONFIG = {
    'daily_summary': True,      # Keep daily metrics summary
    'key_events': True,         # Important events like discount days
    'errors': True,             # Always show errors
    'debug': False,             # Detailed operation logs
    'conversations': False      # Individual conversation logs
}

# Then class definitions
class Location:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)

class Agent:
    def __init__(self, name, description, starting_location, residence, metrics, memory_manager):
        self.name = name
        self.description = description
        self.location = starting_location
        self.residence = residence
        self.metrics = metrics
        self.memory_manager = memory_manager
        self.conversation_history = []
        self.has_children = any(member.startswith(('child_', 'kids', 'children')) 
                              for member in description['family']['household_members'])
        self.last_reflection_time = 0
        self.memory_count_since_reflection = 0
        self.plans = []  # Add this line to store plans
        
    def should_reflect(self, current_time, memory_importance):
        """Determine if agent should reflect based on original paper's criteria"""
        # Check if enough time has passed since last reflection
        time_since_reflection = current_time - self.last_reflection_time
        
        # Check if we're at a natural transition point (returning home)
        is_returning_home = self.location == self.residence and time_since_reflection > 4
        
        # Check if we have enough new memories
        has_enough_memories = self.memory_count_since_reflection >= 5
        
        # Check if this memory is particularly important
        is_important_memory = memory_importance > 0.8
        
        return is_returning_home or has_enough_memories or is_important_memory
        
    def reflect(self, current_time):
        """Reflect on recent experiences and update memory importance"""
        # Get recent memories
        recent_memories = self.memory_manager.get_recent_memories(self.name, current_time)
        
        # Construct reflection prompt
        prompt = f"""You are {self.name}. Reflect on your recent experiences:
        {recent_memories}
        
        Consider:
        1. What patterns do you notice in your behavior?
        2. How have your interactions with others affected you?
        3. What have you learned about the Fried Chicken Shop?
        4. How might this influence your future decisions?
        
        Provide a brief reflection."""
        
        reflection = generate(prompt_meta.format(prompt))
        
        # Update memory importance based on reflection
        for memory in recent_memories:
            # Increase importance of memories that align with reflection insights
            if any(keyword in reflection.lower() for keyword in memory['content'].lower().split()):
                self.memory_manager.update_memory_importance(
                    self.name,
                    memory['id'],
                    memory['importance'] * 1.2  # Increase importance
                )
        
        # Reset counters
        self.last_reflection_time = current_time
        self.memory_count_since_reflection = 0
        
        return reflection
        
    def plan(self, current_time):
        """Plan agent's next action based on time of day and location"""
        hour = current_time % 24
        
        # Create new plan
        new_plan = self.create_daily_plan(hour)
        # Store the plan
        self.plans.append(new_plan)
        return new_plan

    def create_daily_plan(self, hour):
        """Create a plan based on time of day"""
        if hour < 9:  # Early morning
            return f"Stay at {self.residence} and prepare for the day"
        elif hour == 9:  # Start of day
            return f"Go to {self.description['workplace']}"
        elif 9 < hour < 17:  # Work hours
            return f"Work at {self.description['workplace']}"
        elif hour == 17:  # End of work day
            return f"Consider going home or visiting local shops"
        elif 17 < hour < 22:  # Evening
            if self.location == self.residence:
                return "Spend time at home"
            else:
                return "Visit local establishments or head home"
        else:  # Late night
            return f"Return to {self.residence} for rest"

    def execute_action(self, all_agents, current_location, global_time):
        """Execute an action based on the current plan"""
        # Get list of other agents in same location
        others_here = [agent.name for agent in all_agents if agent.location == self.location and agent != self]
        
        # Get the latest plan
        current_plan = self.plans[-1] if self.plans else "No plan yet"
        
        # Add context about residence
        residence_context = ""
        if self.location == self.residence:
            residence_context = " This is your residence where you can rest."
        elif global_time >= experiment_settings['simulation']['daily_hours']['end'] or \
             global_time <= experiment_settings['simulation']['daily_hours']['start']:
            residence_context = f" It's {global_time}:00. Consider returning to your residence at {self.residence} for rest if you haven't recently."
        
        # Add context about educational facilities
        education_context = ""
        is_educational = any(edu_term in self.location.lower() for edu_term in ["school", "academy", "preschool", "daycare", "learning center"])
        
        if is_educational:
            if self.has_children:
                education_context = f" This is an educational facility where your child(ren) might attend."
            else:
                education_context = f" This is an educational facility for children."
        
        prompt = f"""You are {self.name}. Your plan is: {current_plan}. 
        You are in {self.location}{residence_context}{education_context}
        Current time: {global_time}:00
        Other agents here: {', '.join(others_here)}
        
        Consider your daily routine, preferences, and current context to decide your next action.
        What do you do?"""
        
        action = generate(prompt_meta.format(prompt))
        return action if action else f"Stayed at {self.location}"

    def update_memory(self, action, time):
        """Update agent's own memory with their action"""
        if action:
            details = {
                'content': action,
                'time': time,
                'location': self.location
            }
            
            if "Fried Chicken Shop" in action:
                sentiment_prompt = f"Analyze the sentiment towards the Fried Chicken Shop in this text. Only respond with one word: 'positive', 'negative', or 'neutral'. Text: {action}"
                sentiment = generate(sentiment_prompt).strip().lower()
                
                details['sentiment'] = sentiment
                details['listener'] = 'self'
                
                self.memory_manager.add_memory(self.name, "word_of_mouth", details)
                self.metrics.record_interaction(
                    self.name,
                    self.location,
                    "word_of_mouth",
                    details
                )
            else:
                self.memory_manager.add_memory(self.name, "general", details)

# Ollama generation function
def generate(prompt, model='llama3.2'):
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt
        )
        return response['response'].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Set default value for prompt_meta
prompt_meta = '### Instruction:\n{}\n### Response:'

# Logging flags
log_locations = False
log_actions = True
log_plans = False
log_ratings = False
log_memories = False

print_locations = True
print_actions = True
print_plans = True
print_ratings = True
print_memories = False

# Start simulation loop
whole_simulation_output = ""

# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))  # This will point to LLMAgentsTown_Stability/
config_path = os.path.join(base_dir, 'Stability_Agents_Config.json')

# Load configuration
print(f"Loading config from: {config_path}")  # Add this to debug
with open(config_path, 'r') as f:
    town_data = json.load(f)

# Process nested location structure
locations = {}
for category, category_locations in town_data['town_areas'].items():
    for location_name, description in category_locations.items():
        locations[location_name] = Location(location_name, description)

print(f"Initialized locations: {list(locations.keys())}")  # Debug print

# Extract data from config
town_people = town_data['town_people']
town_areas = town_data['town_areas']

print(f"Loaded locations: {list(town_areas.keys())}")  # Add this to debug

# Create world graph
world_graph = nx.Graph()
last_town_area = None
for town_area in town_areas.keys():
    world_graph.add_node(town_area)
    if last_town_area is not None:
        world_graph.add_edge(town_area, last_town_area)
    last_town_area = town_area

# Complete the cycle
world_graph.add_edge(list(town_areas.keys())[0], last_town_area)

# Initialize metrics (simpler now)
metrics = FriedChickenMetrics(
    discount_value=experiment_settings['experiments']['stability_test']['discount_value']
)

# Create ONE memory manager for all agents
memory_mgr = MemoryManager()

# Initialize agents
agents = []
for name, info in town_data['town_people'].items():
    agent = Agent(
        name=name,
        description=info['basics'],
        starting_location=info['basics'].get('residence', 'Main Street Shops'),  # Add default
        residence=info['basics'].get('residence', 'Main Street Shops'),  # Add default
        metrics=metrics,
        memory_manager=memory_mgr  # Use the same memory manager for all agents
    )
    agents.append(agent)
    locations[info['basics'].get('residence', 'Main Street Shops')].add_agent(agent)  # Add to their starting location

# Main simulation loop
def run_simulation():
    try:
        global global_time
        print(f"\n=== Simulation Started at {datetime.now().strftime('%H:%M:%S')} ===")
        print("Day 1 begins")
        print("\nFried Chicken Shop Regular Hours: 10:00-22:00")
        print(f"Simulation will run for 7 days ({7 * 24} hours)")
        
        while global_time < (24 * 7):
            # Debug print
            if global_time % 1 == 0:  # Print every hour
                print(f"Current time: Day {(global_time//24)+1}, Hour {global_time%24}")
            
            # Calculate current day (1-7)
            current_day = (global_time // 24) + 1
            
            # Only print important time markers
            if global_time in [9, 14, 20]:
                print(f"\nTime now: {global_time}:00")
            
            # Discount announcements only on Days 3-4
            if current_day == 3 and global_time % 24 == 9:  # Day 3 (Wednesday) morning
                print("\n=== Special 2-Day Discount Starting Now! ===")
                print(f"Get {experiment_settings['experiments']['stability_test']['discount_value']}% off on all meals!")
                print("Valid today and tomorrow only!")
            
            # End of discount announcement
            if current_day == 4 and global_time % 24 == 21:  # Day 4 (Thursday) at closing
                print("\n=== Special Discount Period Ending ===")
                print("Last chance to get your discounted meals!")
            
            # Day transition
            if global_time % 24 == 0 and global_time > 0:
                print(f"\n=== Day {global_time//24} Completed ===")
                metrics.print_daily_summary(global_time//24)
                metrics.new_day()
                next_day = (global_time//24) + 1
                print(f"\nDay {next_day} begins at {datetime.now().strftime('%H:%M:%S')}")
                
                # Announce upcoming discount at end of Day 2
                if next_day == 3:
                    print("\nAnnouncement: Special discount promotion starting tomorrow morning!")
            
            # Run simulation
            for agent in agents:
                plan = agent.plan(global_time)
                action = agent.execute_action(agents, agent.location, global_time)
                agent.update_memory(action, global_time)
            
            time.sleep(1)
            global_time += 1
            
        print("\n=== Simulation Complete ===")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        
    finally:
        print(f"Simulation ended at time: Day {(global_time//24)+1}, Hour {global_time%24}")
        
        # Move the file saving to the main execution block
        output_dir = os.path.join(base_dir, 'simulation_outputs')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'simulation_{timestamp}.txt')
        
        with open(output_path, 'w') as f:
            f.write(whole_simulation_output)
        
        return output_path

# Add validation for experiment settings
def validate_experiment_settings():
    required_fields = ['simulation', 'fried_chicken_shop', 'experiments']
    with open('experiment_settings.json', 'r') as f:
        settings = json.load(f)
        
    for field in required_fields:
        if field not in settings:
            raise ValueError(f"Missing required field in experiment_settings.json: {field}")
    
    return settings

# Add to main simulation
if __name__ == "__main__":
    try:
        # Load experiment settings first
        with open('experiment_settings.json', 'r') as f:
            experiment_settings = json.load(f)

        # Initialize global variables
        global_time = 0
        SIMULATION_DURATION_DAYS = 7
        TIME_STEP = experiment_settings['simulation']['time_step']

        # Get configuration path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'Stability_Agents_Config.json')
        
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            town_data = json.load(f)

        # Initialize metrics and memory manager ONCE
        metrics = FriedChickenMetrics(
            discount_value=experiment_settings['experiments']['stability_test']['discount_value']
        )
        memory_mgr = MemoryManager()

        # Initialize locations and agents
        locations = {}
        for category, category_locations in town_data['town_areas'].items():
            for location_name, description in category_locations.items():
                locations[location_name] = Location(location_name, description)

        agents = []
        for name, info in town_data['town_people'].items():
            agent = Agent(
                name=name,
                description=info['basics'],
                starting_location=info['basics']['residence'],
                residence=info['basics']['residence'],
                metrics=metrics,
                memory_manager=memory_mgr  # Use the same memory manager for all agents
            )
            agents.append(agent)
            locations[info['basics']['residence']].add_agent(agent)

        # Start simulation
        run_simulation()

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        if 'memory_mgr' in locals():
            memory_file = memory_mgr.save_to_file()
        if 'metrics' in locals():
            metrics_file = metrics.save_metrics()

def run_single_experiment(experiment_type, discount_value, model='llama3.2'):
    """Run a single experiment with specified parameters"""
    # Initialize metrics and memory manager
    metrics = FriedChickenMetrics(discount_value)
    memory_manager = MemoryManager()
    
    # Load configuration
    with open('Stability_Agents_Config.json', 'r') as f:
        town_data = json.load(f)
    
    # Process nested location structure
    locations = {}
    for category, category_locations in town_data['town_areas'].items():
        for location_name, description in category_locations.items():
            locations[location_name] = Location(location_name, description)
    
    # Create world graph
    world_graph = nx.Graph()
    last_town_area = None
    for town_area in town_data['town_areas'].keys():
        world_graph.add_node(town_area)
        if last_town_area is not None:
            world_graph.add_edge(town_area, last_town_area)
        last_town_area = town_area
    
    # Complete the cycle
    world_graph.add_edge(list(town_data['town_areas'].keys())[0], last_town_area)
    
    # Initialize agents
    agents = []
    for name, info in town_data['town_people'].items():
        # Everyone starts at their residence in the morning
        starting_location = info['basics']['residence']
        
        agent = Agent(
            name=name,
            description=info['basics'],
            starting_location=starting_location,  # Start at residence
            residence=info['basics']['residence'],
            metrics=metrics,
            memory_manager=memory_manager
        )
        agents.append(agent)
        locations[starting_location].add_agent(agent)  # Add to their starting location
    
    print(f"\n=== Starting Experiment ===")
    print(f"Type: {experiment_type}")
    print(f"Discount Value: {discount_value}")
    print(f"Duration: {SIMULATION_DURATION_DAYS} days")
    print(f"Discount Days: {', '.join(experiment_settings['experiments'][experiment_type]['discount_days'])}")
    
    # Run simulation for specified duration
    for day in range(1, SIMULATION_DURATION_DAYS + 1):
        print(f"\n=== Day {day} ({datetime.now().strftime('%A')}) ===")
        
        for hour in range(24):
            global_time = hour
            
            # Each agent plans and executes their action
            for agent in agents:
                # Remove agent from current location
                locations[agent.location].remove_agent(agent)
                
                # Generate and execute plan
                plan = agent.plan(global_time)
                action = agent.execute_action(agents, agent.location, global_time)
                
                # Update agent's location based on action
                if "go to" in action.lower():
                    for location in locations:
                        if location.lower() in action.lower():
                            agent.location = location
                            break
                
                # Add agent to new location
                locations[agent.location].add_agent(agent)
            
            # Print current state at key times
            if hour in [7, 12, 18]:  # Meal times
                print(f"\nTime: {hour:02d}:00")
                for location in locations:
                    if locations[location].agents:
                        print(f"{location}: {', '.join([agent.name for agent in locations[location].agents])}")
        
        # Print daily summary
        metrics.print_daily_summary(day)
        
        # Check for significant changes or repeated conversations
        for agent in agents:
            if len(agent.conversation_history) > 1:
                last_conversation = agent.conversation_history[-1]
                prev_conversation = agent.conversation_history[-2]
                if (last_conversation['location'] == prev_conversation['location'] and
                    set(last_conversation['participants']) == set(prev_conversation['participants'])):
                    print(f"\nRepeated conversation detected at {last_conversation['location']} between {', '.join(last_conversation['participants'])}")
        
        # Start new day
        metrics.new_day()
        
        # Save memories and metrics at the end of each day
        memory_manager.save_to_file()
        metrics.save_metrics()
    
    print("\n=== Experiment Complete ===")
    print("All metrics and memories have been saved to the memory_records folder")

if __name__ == "__main__":
    # Example usage:
    # run_single_experiment("stability_test", 10)  # 10% off
    # run_single_experiment("ab_test", 5)         # $5 off
    pass