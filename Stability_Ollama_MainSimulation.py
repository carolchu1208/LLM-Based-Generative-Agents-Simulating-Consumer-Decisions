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
import traceback

# Load experiment settings
with open('experiment_settings.json', 'r') as f:
    experiment_settings = json.load(f)

# Initialize global variables
global_time = experiment_settings['simulation']['start_hour']  # Start at 5:00 on first day

# Add output control flags at the top
PRINT_CONFIG = {
    'daily_summary': True,      # Keep daily metrics summary
    'key_events': True,         # Important events like discount days
    'errors': True,             # Always show errors
    'debug': False,             # Detailed operation logs
    'conversations': False      # Individual conversation logs
}

# Then class definitions - movement system
class Location:
    def __init__(self, name, config_data):
        self.name = name
        # Use the config data directly
        self.type = config_data.get("type", "general")
        self.description = config_data.get("description", "")
        self.opening_hours = config_data.get("opening_hours")
        self.age_range = config_data.get("age_range")
        self.agents = []

    def add_agent(self, agent):
        """Add an agent to this location"""
        self.agents.append(agent)
        agent.memory_manager.add_memory(agent.name, "location_change", {
            'content': f"Arrived at {self.name}",
            'timestamp': global_time
        })

    def remove_agent(self, agent):
        """Remove an agent from this location"""
        if agent in self.agents:
            self.agents.remove(agent)
            agent.memory_manager.add_memory(agent.name, "location_change", {
                'content': f"Left {self.name}",
                'timestamp': global_time
            })

    def get_present_agents(self):
        """Get list of agents currently at this location"""
        return self.agents

    def is_open(self, current_time):
        """Check if location is open based on config"""
        if not self.opening_hours:
            return True  # Always open if no hours specified
        
        hour = current_time % 24
        return self.opening_hours["start"] <= hour < self.opening_hours["end"]

    def is_appropriate_for_age(self, age):
        """Check if location is appropriate for agent's age"""
        if not self.age_range:
            return True  # No age restriction
        return self.age_range[0] <= age <= self.age_range[1]

# Then class Agens - schedule production system
class Agent:
    # 1. Initialize agent
    def __init__(self, name, config, memory_manager):
        # Add world_graph as a class attribute
        self.world_graph = None  # Will be set later
        # Basic info
        self.name = name
        self.config = config
        self.memory_manager = memory_manager
        self.current_time = 0  # Add this line to track agent's time
        
        # Get person data from town_people
        person_data = config['town_people'][name]['basics']
        
        # Set basic attributes
        self.current_location = person_data.get('residence', 'Home')
        self.age = person_data.get('age')
        self.occupation = person_data.get('occupation')
        
        # Handle family data
        family_info = person_data.get('family', {})
        self.family_status = family_info.get('status', 'single')
        self.family = family_info.get('household_members', [])
        
        # Determine role based on age and occupation
        if self.age and self.age < 18:
            self.role = "Child"
        elif self.occupation == "student":
            self.role = "Student"
        elif any(member.startswith('child_') for member in self.family):
            self.role = "Parent"
        else:
            self.role = "Adult"
        
        # Financial context
        income_data = person_data.get('income', {})
        if income_data:
            if income_data['type'] == 'monthly':
                monthly_income = income_data['amount']
            elif income_data['type'] == 'annual':
                monthly_income = income_data['amount'] / 12
            elif income_data['type'] == 'hourly':
                # Assume 40 hours/week, 4 weeks/month for full-time
                monthly_income = income_data['amount'] * 40 * 4
            else:
                monthly_income = 0
        else:
            monthly_income = 0
        
        # Add daily food budget calculation (assuming 5.5% of monthly income for food)
        self.monthly_food_budget = monthly_income * 0.055
        self.daily_food_budget = self.monthly_food_budget / 30
        self.spent_today = 0  # Track daily spending
        
        # Food preferences and context - read price from settings
        self.food_options = {
            "Fried Chicken Shop": {
                "price": experiment_settings['fried_chicken_shop']['base_price'],
                "satisfaction": None
            },
            "Home Cooking": {"price": 5.00, "satisfaction": None},
            "Grocery Shopping": {"price": 10.00, "satisfaction": None},
            "Fast Food": {"price": 8.00, "satisfaction": None},
            "Restaurant": {"price": 20.00, "satisfaction": None}
        }
        
        # Track food experiences
        self.food_experiences = {
            "Fried Chicken Shop": [],
            "Home Cooking": [],
            "Fast Food": [],
            "Restaurant": []
        }
        
        # State tracking
        self.current_state = {
            "energy": 100,
            "hunger": 0,
            "last_meal_time": 0,
            "last_sleep_time": 0
        }
        self.last_meal = {
            "time": 0,
            "type": None,
            "satisfaction": None
        }
        self.grocery_stock = 100  # Percentage of grocery supplies at home

    def set_world_graph(self, graph):
        """Set the world graph for this agent"""
        self.world_graph = graph

    # 4. Location and movement methods
    def get_school_for_age(self, age):
        """Determine appropriate school based on age"""
        if age < 5:
            return "Sunshine Daycare Center"
        elif age < 12:
            return "Elementary School"
        else:
            return "Town Public High School"

    def move_to_location(self, new_location, locations_dict):
        """Move agent to a new location"""
        if self.current_location:
            locations_dict[self.current_location].remove_agent(self)
        self.current_location = new_location
        locations_dict[new_location].add_agent(self)
        return f"Moved to {new_location}"

    # 5. Food decision methods
    def make_first_time_food_decision(self, current_time, nearby_agents):
        """Decision making for food when no prior experience exists"""
        options_scores = {}
        
        for place, details in self.food_options.items():
            score = 0
            
            # Budget consideration
            if details["price"] <= self.daily_food_budget * 0.5:  # Using only 50% of daily budget
                score += 3
            elif details["price"] <= self.daily_food_budget:
                score += 1
                
            # Location convenience
            if place == self.current_location:
                score += 2
            
            # Check if anyone nearby has experience
            for agent in nearby_agents:
                if agent.name in self.family:
                    # Check family member's memories
                    family_memories = self.memory_manager.get_recent_memories(agent.name)
                    for memory in family_memories:
                        if place in memory['content'] and "enjoyed" in memory['content'].lower():
                            score += 3
                            break
            
            # Time-based choices
            hour = current_time % 24
            if hour in [11, 12, 13]:  # Lunch time
                if place in ["Fried Chicken Shop", "Local Diner", "The Coffee Shop"]:
                    score += 2  # Quick lunch options
            elif hour in [17, 18, 19]:  # Dinner time
                if place in ["Family Restaurant", "Home Cooking"]:
                    score += 2  # Traditional dinner options
            
            options_scores[place] = score
        
        # Choose highest scoring option
        best_option = max(options_scores.items(), key=lambda x: x[1])
        
        return best_option[0], f"First time trying {best_option[0]} based on {self.get_decision_reason(best_option[0], options_scores, nearby_agents)}"

    def get_decision_reason(self, place, scores, nearby_agents):
        """Get the main reason for the decision"""
        if place == self.current_location:
            return "convenience"
        elif self.food_options[place]["price"] <= self.daily_food_budget * 0.5:
            return "good price"
        elif any(agent.name in self.family for agent in nearby_agents):
            return "family recommendation"
        else:
            return "trying something new"

    def decide_food_choice(self, current_time, nearby_agents):
        """Decide what food option to choose"""
        # Check if we have any food experiences
        has_experiences = any(len(exp) > 0 for exp in self.food_experiences.values())
        
        if not has_experiences:
            # First time choosing food
            return self.make_first_time_food_decision(current_time, nearby_agents)
        # Use experience-based decision making
        return self.make_experienced_food_decision(current_time, nearby_agents)

    def decide_food_action(self, current_time, nearby_agents):
        """Execute the food-related decision"""
        # First check if we should even consider food
        if self.current_state["hunger"] < 40:
            return None, "not hungry enough"
            
        # Get the food choice
        choice, reason = self.decide_food_choice(current_time, nearby_agents)
        
        # Check if we can afford it
        if not self.can_afford_purchase(self.food_options[choice]["price"], current_time):
            return None, "cannot afford"
            
        # Execute the choice
        if choice == "Fried Chicken Shop":
            # Handle chicken shop purchase
            price = self.food_options[choice]["price"]
            # Apply discount if applicable
            day = (current_time // 24) + 1
            if day in [3, 4]:
                discount = self.experiment_settings['fried_chicken_shop']['discount_value']
                original_price = price
                price = price * (1 - discount/100)
            
            # 1. Record the visit
            visit_details = {
                'satisfaction': 4,
                'food_quality': 4,
                'price_satisfaction': 4,
                'service_rating': 4,
                'wait_time': 15,
                'feedback': reason,
                'would_recommend': True,
                'return_intention': 4,
                'experience': reason,
                'price_paid': price,  # Use the calculated price
                'used_discount': day in [3, 4],
                'timestamp': f"{day:02d}-{current_time%24:02d}:00"
            }
            
            # 2. Record in both systems
            self.memory_manager.add_memory(self.name, "store_visit", visit_details)
            metrics.record_interaction(self.name, "Fried Chicken Shop", "store_visit", visit_details)
            
            # 3. Record the purchase separately
            purchase_details = {
                'amount': price,
                'used_discount': day in [3, 4],
                'discount_amount': original_price - price if day in [3, 4] else 0
            }
            metrics.record_interaction(self.name, "Fried Chicken Shop", "purchase", purchase_details)
            
            memory = f"Bought food at {choice} for ${price:.2f} ({reason})"
            
        elif choice == "Home Cooking":
            # Check if we have groceries
            if self.grocery_stock < 20:
                return "Grocery Shopping", "need to buy groceries first"
            self.grocery_stock -= 20
            memory = f"Cooked meal at home ({reason})"
            
        elif choice == "Grocery Shopping":
            price = self.food_options[choice]["price"]
            self.record_purchase(price, current_time)
            self.grocery_stock = 100
            memory = f"Bought groceries for ${price:.2f}"
            
        else:  # Restaurant or Fast Food
            price = self.food_options[choice]["price"]
            self.record_purchase(price, current_time)
            memory = f"Had meal at {choice} for ${price:.2f} ({reason})"
        
        # Update state
        self.current_state["hunger"] = max(0, self.current_state["hunger"] - 50)
        self.last_meal = {
            "time": current_time,
            "type": choice,
            "satisfaction": None  # Will be updated after experience
        }
        
        # Create memory
        self.memory_manager.add_memory(self.name, memory, current_time)
        
        return choice, memory

    def process_food_experience(self, food_choice, current_time):
        """Process and remember the food experience"""
        satisfaction = self.food_options[food_choice]["satisfaction"]
        price = self.food_options[food_choice]["price"]
        
        # Record experience
        if satisfaction > 0.7 and price < self.daily_food_budget * 0.5:
            memory = f"Really enjoyed the {food_choice}, good value for money"
            importance = 0.7
        elif satisfaction > 0.7:
            memory = f"Enjoyed the {food_choice}, though it was a bit pricey"
            importance = 0.6
        else:
            memory = f"The {food_choice} was okay, might try something else next time"
            importance = 0.4
            
        self.memory_manager.add_memory(self.name, memory, current_time)
        
        # Update state
        self.last_meal = {
            "time": current_time,
            "type": food_choice,
            "satisfaction": satisfaction
        }
        
        if food_choice == "Grocery Shopping":
            self.grocery_stock = 100
        elif food_choice == "Home Cooking":
            self.grocery_stock -= 20

    # 6. State update methods
    def update_state(self, current_time):
        """Update agent's state based on time passed"""
        # Update energy
        if self.current_state["energy"] < 30:
            self.current_state["energy"] = min(100, self.current_state["energy"] + 50)
        else:
            self.current_state["energy"] = max(0, self.current_state["energy"] - 5)
            
        # Update hunger
        if self.current_state["hunger"] > 70:
            self.current_state["hunger"] = max(0, self.current_state["hunger"] - 50)
            self.current_state["last_meal_time"] = current_time
        else:
            hours_since_meal = (current_time - self.current_state["last_meal_time"]) / 24
            self.current_state["hunger"] = min(100, self.current_state["hunger"] + hours_since_meal * 10)

    def record_purchase(self, price, current_time):
        """Record a purchase and update budget"""
        self.spent_today += price
        self.current_state["hunger"] = max(0, self.current_state["hunger"] - 50)
        self.last_meal = {
            "time": current_time,
            "type": None,
            "satisfaction": None
        }

    def can_afford_purchase(self, price, current_time):
        """Check if agent can afford a purchase"""
        return self.spent_today + price <= self.monthly_food_budget

    def react_to_environment(self, current_time, nearby_agents):
        """React to environmental factors and other agents"""
        reactions = []
        
        # React to location
        if self.current_location == "Main Street Shops":
            if current_time % 24 in [12, 18]:  # Lunch or dinner time
                reactions.append("Notice food options")
                
        # React to other agents
        for agent in nearby_agents:
            if agent.name in self.family:
                reactions.append(f"Consider family member {agent.name}'s needs")
                
        return reactions

    def evaluate_food_options(self, current_time):
        """Evaluate all available food options based on current context"""
        hour = current_time % 24
        day = (current_time // 24) + 1
        
        # Get relevant memories
        recent_memories = self.memory_manager.get_recent_memories(self.name, window=24)
        
        options_scores = {}
        for option, details in self.food_options.items():
            score = 0
            
            # Base satisfaction score
            score += details["satisfaction"] * 10
            
            # Budget consideration
            budget_ratio = details["price"] / self.daily_food_budget
            if budget_ratio > 0.7:
                score -= 5  # Too expensive for daily meal
            
            # Time context
            if hour in [11, 12, 13]:  # Lunch time
                if option in ["Fast Food", "Fried Chicken Shop"]:
                    score += 3  # Quick lunch options
            elif hour in [17, 18, 19]:  # Dinner time
                if option in ["Home Cooking", "Restaurant"]:
                    score += 3  # Preferred dinner options
            
            # Location context
            if self.current_location == "Main Street Shops":
                if option == "Fried Chicken Shop":
                    score += 2  # Convenience bonus
            
            # Grocery status
            if option == "Home Cooking":
                if self.grocery_stock < 30:
                    score -= 5  # Need groceries
                else:
                    score += 2  # Have supplies
            
            # Special conditions
            if option == "Fried Chicken Shop" and day in [3, 4]:
                discount = self.experiment_settings['fried_chicken_shop']['discount_value']
                score += 4  # Discount bonus
            
            # Memory influence
            for memory in recent_memories:
                if option in memory['content']:
                    if "enjoyed" in memory['content'].lower():
                        score += 2
                    elif "regret" in memory['content'].lower():
                        score -= 2
                    elif "discount" in memory['content'].lower() and option == "Fried Chicken Shop":
                        score += 3
            
            options_scores[option] = score

        return options_scores

    def can_deviate_from_schedule(self, current_time):
        """Determine if agent can deviate based on family responsibilities"""
        current_activity = self.get_current_activity(current_time)
        
        # Can't deviate during child-related activities
        child_activities = ["drop", "pickup", "children", "family dinner"]
        if any(activity in current_activity.lower() for activity in child_activities):
            return False
            
        # Can't deviate during work/school
        fixed_activities = ["work", "school", "teaching", "sleep"]
        if any(activity in current_activity.lower() for activity in fixed_activities):
            return False
            
        # More likely to deviate during free times
        flexible_times = ["lunch break", "adult free time"]
        if any(time in current_activity.lower() for time in flexible_times):
            return True
            
        return random.random() < 0.2

    def coordinate_with_family(self, current_time):
        """Coordinate schedule changes with family members"""
        if self.dependents:
            # Check if spouse/partner is available
            for family_member in self.family:
                if family_member in self.config['agents']:
                    partner_role = self.config['agents'][family_member].get('role')
                    if partner_role in ['Father', 'Mother']:
                        partner_schedule = self.config['agents'][family_member].get('daily_schedule', {})
                        # Could coordinate child pickup/dropoff here
                        return True
        return False

    def plan_next_action(self, current_time):
        """Dynamically plan next action based on current state and context"""
        hour = current_time % 24
        day = current_time // 24
        
        # Get relevant memories
        recent_memories = self.memory_manager.get_recent_memories(self.name, window=24)
        
        # Consider basic needs
        if self.current_state["energy"] < 30:
            return "I need to rest"
        if self.current_state["hunger"] > 70:
            return "I need to find food"
            
        # Consider work/school obligations
        if 9 <= hour <= 17 and self.occupation:
            if "work" not in str(recent_memories):
                return f"I should go to {self.occupation}"
                
        # Consider family responsibilities
        for memory in recent_memories:
            if "children" in memory and "school" in memory:
                return "Need to take care of children"
                
        # Consider social and food opportunities
        if self.current_location == "Main Street Shops":
            for memory in recent_memories:
                if "discount" in memory and "Fried Chicken" in memory:
                    return "Maybe I should check out the chicken discount"
        
        return "Continue with current activity"

    def should_interact(self, nearby_agent, current_time, location):
        """Determine if agent should interact based on context"""
        # Check if location is conducive to interaction
        social_spaces = ["Main Street Shops", "Restaurant", "Park", "Community Center"]
        if location not in social_spaces:
            return False
        
        # Check relationship
        relationship_score = 0
        if nearby_agent.name in self.family:
            relationship_score += 3  # Family members more likely to interact
        
        # Check time context
        hour = current_time % 24
        if hour in [12, 13, 17, 18, 19]:  # Meal times
            relationship_score += 1  # More likely to chat during meals
        
        # Check if they have relevant topics (e.g., recent store visits)
        recent_memories = self.memory_manager.get_recent_memories(self.name, window=24)
        has_relevant_topic = any(
            memory['type'] == "store_visit" 
            for memory in recent_memories
        )
        if has_relevant_topic:
            relationship_score += 1
        
        # Higher threshold for strangers, lower for family/friends
        interaction_threshold = 4 - relationship_score
        
        return random.random() * 5 < interaction_threshold  # Scale of 0-5

    def get_accessible_locations(self, current_location, max_distance=2):
        """Get all locations accessible within a certain distance"""
        if not self.world_graph:
            return []  # Return empty list if world_graph not set
            
        accessible = set()
        
        # Get current location's area type
        current_area = None
        for area_type, locations in self.config['town_areas'].items():
            if current_location in locations:
                current_area = area_type
                break
        
        if current_area:
            # Get all areas within max_distance steps
            for area, distance in nx.single_source_shortest_path_length(
                self.world_graph, 
                current_area, 
                cutoff=max_distance
            ).items():
                # Add all locations in this area
                accessible.update(self.config['town_areas'][area].keys())
        
        return list(accessible)

    def decide_next_location(self, current_time, locations):
        """Decide next location based on location availability and agent needs"""
        hour = current_time % 24
        
        # Get accessible locations (removed world_graph parameter)
        accessible_locations = self.get_accessible_locations(self.current_location)
        
        # Filter for currently open locations
        open_locations = [loc for loc in accessible_locations 
                         if locations[loc].is_open(current_time)]
        
        # Filter for age-appropriate locations
        suitable_locations = [loc for loc in open_locations 
                            if locations[loc].is_appropriate_for_age(self.age)]
        
        # Consider work/school during appropriate hours
        if 8 <= hour <= 17 and self.occupation:
            workplace = self.config['town_people'][self.name]['basics']['workplace']
            if workplace in suitable_locations:
                return workplace
        
        # Priority 3: Food needs
        if self.current_state["hunger"] > 60:
            food_locations = [loc for loc in suitable_locations 
                             if loc in ["Fried Chicken Shop", "Local Diner", 
                                      "Family Restaurant", "The Coffee Shop"]]
            if food_locations:
                return random.choice(food_locations)
        
        # Default to staying at current location if it's still suitable
        if self.current_location in suitable_locations:
            return self.current_location
        
        # If current location becomes unsuitable, return home
        return "Home" if "Home" in accessible_locations else self.current_location

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

def log_event(event, log_file):
    """Log event to both console and file"""
    # Print to console
    print(event)
    
    # Write to file
    if log_file:
        log_file.write(event + "\n")
        log_file.flush()  # Ensure immediate write to file

def initialize_simulation():
    """Set up initial simulation state"""
    try:
        # Load configuration
        config_path = os.path.join(base_dir, 'Stability_Agents_Config.json')
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize components
        memory_mgr = MemoryManager(memory_limit=200)
        metrics = FriedChickenMetrics(
            discount_value=experiment_settings['fried_chicken_shop']['discount_value']
        )
        
        # Create world graph
        world_graph = nx.Graph()
        
        # Add nodes for each area type
        for area_type in config['town_areas'].keys():
            world_graph.add_node(area_type)
        
        # Add edges between connected areas (define connections)
        area_connections = [
            ('retail_and_grocery', 'work'),
            ('retail_and_grocery', 'dining'),
            ('retail_and_grocery', 'residences'),
            ('work', 'dining'),
            ('work', 'community'),
            ('residences', 'education'),
            ('residences', 'community'),
            ('dining', 'community'),
            ('education', 'community'),
            ('healthcare', 'community'),
            ('healthcare', 'residences')
        ]
        
        world_graph.add_edges_from(area_connections)
        
        # Initialize locations from town_areas
        locations = {}
        for category, category_locations in config['town_areas'].items():
            for location_name, location_info in category_locations.items():
                if isinstance(location_info, str):
                    location_info = {
                        "type": category,
                        "description": location_info
                    }
                locations[location_name] = Location(location_name, location_info)
        
        print(f"Initialized locations: {list(locations.keys())}")
        print(f"Loaded locations: {list(config['town_areas'].keys())}")
        
        # Initialize agents
        agents = []
        for name, info in config['town_people'].items():
            agent = Agent(name, config, memory_mgr)
            agent.set_world_graph(world_graph)  # Set world_graph for each agent
            agents.append(agent)
            residence = info['basics'].get('residence', 'Main Street Shops')
            if residence in locations:
                locations[residence].add_agent(agent)
            else:
                print(f"Warning: Residence {residence} not found for {name}, placing at Main Street Shops")
                locations['Main Street Shops'].add_agent(agent)
        
        return agents, metrics, memory_mgr, locations, world_graph

    except Exception as e:
        print(f"Error initializing simulation: {str(e)}")
        traceback.print_exc()
        return None, None, None, None, None

def run_simulation(agents, metrics, memory_mgr, locations, world_graph):
    """Main simulation loop"""
    try:
        # Create output directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'simulation_outputs')
        memory_dir = os.path.join(base_dir, 'memory_records')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(memory_dir, exist_ok=True)
        
        # Create simulation log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'simulation_{timestamp}.txt')
        
        with open(output_file, 'w') as log_file:
            # Write initial information
            log_event(f"=== Simulation Started at {datetime.now().strftime('%H:%M:%S')} ===", log_file)
            log_event("Day 1 begins\n", log_file)
            
            # Run for 7 days
            global global_time
            global_time = 0
            while global_time < (24 * 7):
                process_time_step(global_time, agents, locations, metrics, memory_mgr, world_graph, log_file)
                global_time += 1
                time.sleep(0.1)  # Reduced sleep time for faster simulation
            
            # Write final statistics
            log_event("\n=== Simulation Complete ===", log_file)
            log_event("\n=== Final Statistics ===", log_file)
            final_stats = metrics.get_final_statistics()
            log_event(final_stats, log_file)
            
            # Save all data at the end
            memory_mgr.save_to_file()
            metrics.save_to_file()
            
            print(f"\nSimulation output saved to: {output_file}")
            print(f"Memory records saved to: {memory_dir}")
            print(f"Metrics saved to: {output_dir}")

    except Exception as e:
        print(f"\nError in simulation: {str(e)}")
        traceback.print_exc()

def main():
    """Entry point of simulation"""
    # Initialize simulation with world_graph
    agents, metrics, memory_mgr, locations, world_graph = initialize_simulation()
    if not agents:
        print("Failed to initialize simulation")
        return
    
    # Run simulation with world_graph
    run_simulation(agents, metrics, memory_mgr, locations, world_graph)

def process_time_step(current_time, agents, locations, metrics, memory_mgr, world_graph, log_file):
    """Process one time step of simulation"""
    hour = current_time % 24
    day = (current_time // 24) + 1
    
    try:
        # Time announcements at key hours
        if hour in [9, 14, 20]:
            event = f"\nTime now: Day {day}, {hour:02d}:00"
            log_event(event, log_file)
        
        # Process each agent's actions
        for agent in agents:
            try:
                # 1. Update agent state
                agent.update_state(current_time)
                
                # 2. Get nearby agents
                nearby_agents = [
                    other for other in locations[agent.current_location].get_present_agents()
                    if other != agent
                ]
                
                # 3. Process food-related decisions
                if agent.current_state["hunger"] > 60:
                    choice, reason = agent.decide_food_action(current_time, nearby_agents)
                    if choice and reason:
                        if choice == "Fried Chicken Shop":
                            # Record visit metrics
                            visit_details = {
                                'agent': agent.name,
                                'time': f"{day:02d}-{hour:02d}:00",
                                'location': choice,
                                'reason': reason,
                                'satisfaction': 4  # Default satisfaction
                            }
                            metrics.record_interaction(agent.name, choice, "store_visit", visit_details)
                            
                            # Record purchase metrics if applicable
                            if "Bought food" in reason:
                                purchase_details = {
                                    'amount': agent.food_options[choice]["price"],
                                    'used_discount': day in [3, 4]
                                }
                                metrics.record_interaction(agent.name, choice, "purchase", purchase_details)
                        
                        log_event(f"{agent.name}: {reason}", log_file)
                
                # 4. Process social interactions
                if nearby_agents and random.random() < 0.3:
                    for nearby_agent in nearby_agents:
                        if agent.should_interact(nearby_agent, current_time, agent.current_location):
                            interaction_details = {
                                'from_agent': agent.name,
                                'to_agent': nearby_agent.name,
                                'location': agent.current_location,
                                'type': "word_of_mouth",
                                'sentiment': 'positive',
                                'content': "Shared experience about Fried Chicken Shop"
                            }
                            metrics.record_interaction(
                                agent.name,
                                agent.current_location,
                                "word_of_mouth",
                                interaction_details
                            )
                
                # 5. Move to new location if needed
                new_location = agent.decide_next_location(current_time, locations)
                if new_location != agent.current_location:
                    result = agent.move_to_location(new_location, locations)
                    log_event(f"{agent.name}: {result}", log_file)
                
            except Exception as e:
                log_event(f"Error processing agent {agent.name}: {str(e)}", log_file)
        
        # Generate and save metrics every hour
        if hour % 1 == 0:
            metrics_file = metrics.save_to_file()
            if metrics_file:
                log_event(f"Metrics saved to: {metrics_file}", log_file)
        
        # Update day and generate summary if needed
        if hour == 23:
            metrics.new_day()
            summary_file = generate_summary(metrics, memory_mgr, current_time)
            if summary_file:
                log_event(f"Daily summary saved to: {summary_file}", log_file)
                
    except Exception as e:
        log_event(f"Error in time step {current_time}: {str(e)}", log_file)

def generate_summary(metrics, memory_mgr, current_time):
    """Generate and save simulation summary"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        summary_dir = os.path.join(base_dir, 'memory_records', 'simulation_summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(summary_dir, f'summary_{timestamp}.txt')
        
        day = (current_time // 24) + 1
        hour = current_time % 24
        
        with open(summary_file, 'w') as f:
            f.write(f"=== Simulation Summary - Day {day}, Hour {hour:02d}:00 ===\n\n")
            
            # Get metrics summary
            metrics_summary = metrics.get_final_statistics()
            f.write(metrics_summary)
            
            # Add memory statistics
            f.write("\nMemory Statistics:\n")
            f.write("-" * 30 + "\n")
            # Add more detailed memory statistics here
            
        return summary_file
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return None

if __name__ == "__main__":
    main()
