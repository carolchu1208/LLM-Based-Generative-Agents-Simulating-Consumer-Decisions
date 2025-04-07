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
        self.has_children = "kids" in description.lower() or "children" in description.lower()
        self.last_reflection_time = 0
        self.memory_count_since_reflection = 0
        
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
        
    def execute_action(self, all_agents, current_location, global_time):
        """Execute an action based on the current plan"""
        # Get list of other agents in same location
        others_here = [agent.name for agent in all_agents if agent.location == self.location and agent != self]
        
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
        
        # Let the LLM decide about meal times based on agent's schedule and preferences
        prompt = f"""You are {self.name}. Your plan is: {self.plans[-1]}. 
        You are in {self.location}{residence_context}{education_context}
        Current time: {global_time}:00
        Other agents here: {', '.join(others_here)}
        
        Consider your daily routine, preferences, and current context to decide your next action.
        What do you do?"""
        
        action = generate(prompt_meta.format(prompt))

        # Add error handling for None action
        if action is None:
            action = f"Stayed at {self.location}"  # Default action if generation fails
        
        # Track Fried Chicken Shop interactions
        if "Fried Chicken Shop" in action:
            # Analyze sentiment if it's a mention
            sentiment_prompt = f"Analyze the sentiment towards the Fried Chicken Shop in this text. Only respond with one word: 'positive', 'negative', or 'neutral'. Text: {action}"
            sentiment = generate(sentiment_prompt).strip().lower()
            
            # Calculate memory importance based on sentiment and context
            memory_importance = 0.5  # Base importance
            if sentiment == 'positive':
                memory_importance += 0.3
            elif sentiment == 'negative':
                memory_importance += 0.2
            if "buy" in action.lower() or "purchase" in action.lower():
                memory_importance += 0.2
            
            # Record the interaction
            self.metrics.record_interaction(
                self.name,
                self.location,
                "word_of_mouth",
                sentiment=sentiment
            )
            
            # Check if reflection is needed
            if self.should_reflect(global_time, memory_importance):
                reflection = self.reflect(global_time)
                print(f"\n{self.name} reflected: {reflection}")
        
        # Track store visits
        if "Fried Chicken Shop" in self.location:
            self.metrics.record_interaction(
                self.name,
                self.location,
                "store_visit"
            )
            
            # Track purchases if mentioned
            if any(word in action.lower() for word in ['buy', 'purchase', 'bought', 'order', 'get']):
                self.metrics.record_interaction(
                    self.name,
                    self.location,
                    "purchase"
                )
        
        # Record conversation if there are other agents present
        if others_here:
            conversation = {
                'time': global_time,
                'location': self.location,
                'participants': [self.name] + others_here,
                'action': action
            }
            
            # Add to memory stream
            self.add_memory('conversation', action)
            self.memory_count_since_reflection += 1
            
            # Check for repeated conversations
            if self.detect_repeated_conversation(conversation):
                print(f"\n⚠️ Repeated conversation pattern detected:")
                print(f"Location: {self.location}")
                print(f"Participants: {', '.join(conversation['participants'])}")
                print(f"Action: {action}")
                print("Relationship Context:")
                for ctx in self.get_relationship_context(conversation['participants']):
                    print(f"- {ctx}")
        
        return action

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
base_dir = os.path.dirname(os.path.abspath(__file__))  # This will point to Lululemon/
config_path = os.path.join(base_dir, 'config_lululemon.json')  # This will be Lululemon/config_lululemon.json

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

# Initialize metrics
metrics = FriedChickenMetrics()

# Initialize agents
agents = []
for name, info in town_people.items():
    agent = Agent(name, info['description'], info['starting_location'], 
                 info['residence'], metrics, MemoryManager())
    agents.append(agent)
    locations[info['starting_location']].add_agent(agent)

# memory manage
from memory_manager import MemoryManager

# Initialize memory manager
memory_mgr = MemoryManager()

# In your simulation loop:
def update_memories(agent, action, time):
    memory_mgr.add_memory(agent.name, action, time)
    if time % 3 == 0:  # Compress every 3 time steps
        memory_mgr.compress_memories(agent.name, generate)

# Main simulation loop
def run_simulation():
    try:
        global global_time
        print(f"\n=== Simulation Started at {datetime.now().strftime('%H:%M:%S')} ===")
        print("Day 1 begins")
        print("\nSpecial Event Today: Lululemon x Yoga VIP Event at 14:00")
        print("Location: Serenity Flow Yoga Studio")
        
        while True:
            # Only print important time markers
            if global_time in [9, 14, 20]:  # Key times: start of day, VIP event, evening
                print(f"\nTime now: {global_time}:00")
            
            # VIP event announcements
            if global_time == 13:
                print("\n=== VIP Event Starting in 1 hour! ===")
            elif global_time == 14:
                print("\n=== VIP Event Beginning Now ===")
            
            # Day transition
            if global_time % 24 == 0 and global_time > 0:
                print(f"\n=== Day {global_time//24} Completed ===")
                metrics.print_daily_summary(global_time//24)
                metrics.new_day()
                print(f"\nDay {(global_time//24)+1} begins at {datetime.now().strftime('%H:%M:%S')}")
            
            # Run simulation (without printing every action)
            for agent in agents:
                plan = agent.plan(global_time)
                action = agent.execute_action(agents, agent.location, global_time)
                update_memories(agent, action, global_time)
            
            time.sleep(1)
            global_time += 1
            
    except KeyboardInterrupt:
        print("\n\n=== Simulation Interrupted ===")
        print(f"Completed {global_time//24} days and {global_time%24} hours")
        
    finally:
        # Save memories and metrics
        memory_file = memory_mgr.save_to_file()
        metrics_file = metrics.save_metrics()
        
        # Save simulation output
        output_dir = os.path.join(base_dir, 'simulation_outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'simulation_{timestamp}.txt')
        
        with open(output_path, 'w') as f:
            f.write(whole_simulation_output)
        
        print(f"Simulation output saved to {output_path}")
        return output_path

# Run the simulation if this script is executed directly
if __name__ == "__main__":
    try:
        output_file = run_simulation()
        print(f"Simulation complete. Output saved to {output_file}")
    except KeyboardInterrupt:
        print("\n=== Final Lululemon Impact Report ===")
        # Fix the sales calculation
        total_sales = sum(day['sales']['amount'] for day in metrics.daily_metrics.values())
        total_visits = sum(len(day['store_visits']) for day in metrics.daily_metrics.values())
        
        print(f"\nSimulation Duration: {metrics.current_day} days")
        print(f"Total Store Visits: {total_visits}")
        print(f"Total Sales: ${total_sales:,.2f}")
    finally:
        memory_file = memory_mgr.save_to_file()
        metrics_file = metrics.save_metrics()

def run_single_experiment(experiment_type, discount_value, model='llama3.2'):
    """Run a single experiment with specified parameters"""
    # Initialize metrics and memory manager
    metrics = FriedChickenMetrics(experiment_type, discount_value)
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
        agent = Agent(name, info['description'], info['starting_location'], 
                     info['residence'], metrics, memory_manager)
        agents.append(agent)
        locations[info['starting_location']].add_agent(agent)
    
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