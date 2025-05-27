from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime
import traceback
import collections # For deque in BFS
import re

from simulation_constants import ACTIVITY_TYPES, MEMORY_TYPES

class TownMap:
    def __init__(self, world_locations_data: Dict[str, List[int]], travel_paths_data: List[List[List[int]]]):
        self.world_locations: Dict[str, Tuple[int, int]] = { 
            name: tuple(coords) for name, coords in world_locations_data.items()
        }
        self.adjacency_list: Dict[Tuple[int, int], List[Tuple[int, int]]] = collections.defaultdict(list)
        self._build_graph(travel_paths_data)

    def _build_graph(self, travel_paths_data: List[List[List[int]]]):
        """Builds an adjacency list representation of the traversable grid."""
        for path_segment in travel_paths_data:
            for i in range(len(path_segment) - 1):
                u, v = tuple(path_segment[i]), tuple(path_segment[i+1])
                self.adjacency_list[u].append(v)
                self.adjacency_list[v].append(u) # Assuming paths are bidirectional

    def get_coordinates_for_location(self, location_name: str) -> Optional[Tuple[int, int]]:
        return self.world_locations.get(location_name)

    def get_location_name_at_coord(self, coord: Tuple[int, int]) -> Optional[str]:
        for name, loc_coord in self.world_locations.items():
            if loc_coord == coord:
                return name
        return None

    def find_path(self, start_coord: Tuple[int, int], end_coord: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Finds the shortest path using Breadth-First Search (BFS)."""
        if start_coord == end_coord:
            return [start_coord]
        
        queue = collections.deque([(start_coord, [start_coord])])
        visited = {start_coord}

        while queue:
            current_coord, path = queue.popleft()

            for neighbor in self.adjacency_list.get(current_coord, []):
                if neighbor == end_coord:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None # Path not found

class Location:
    """Location class for managing areas in the town"""
    def __init__(self, name: str, type: str, capacity: int, grid_coordinate: Optional[Tuple[int, int]] = None):
        self.name = name
        self.type = type
        self.capacity = capacity  # Each location has a capacity
        self.agents = []  # List of agents currently at the location
        self.queue = []  # Add queue for waiting agents
        self.base_price = 10.0  # Default price
        self.discount = None
        self.grid_coordinate = grid_coordinate # New attribute
        self.hours = {
            'open': 8,   # Default opening time
            'close': 22  # Default closing time
        }  # These will be overridden by the config during initialization
        
    def is_open(self, current_hour: int) -> bool:
        """Check if location is open at the given hour"""
        try:
            # Default hours if not set
            if not hasattr(self, 'hours'):
                self.hours = {'open': 8, 'close': 22}
            
            # Handle 24/7 locations
            if self.hours.get('always_open', False):
                return True
            
            # Get opening and closing hours
            open_hour = self.hours.get('open', 8)
            close_hour = self.hours.get('close', 22)
            
            # Check if current hour is within operating hours
            if close_hour > open_hour:
                # Normal case (e.g. 8:00-22:00)
                return open_hour <= current_hour < close_hour
            else:
                # Overnight case (e.g. 22:00-6:00)
                return current_hour >= open_hour or current_hour < close_hour
                
        except Exception as e:
            print(f"Error checking if location is open: {str(e)}")
            return False  # Default to closed on error
        
    def is_full(self) -> bool:
        """Check if location is at capacity"""
        return len(self.agents) >= self.capacity
        
    def add_agent(self, agent: 'Agent') -> bool:
        """Add agent to location"""
        if not self.is_full():
            self.agents.append(agent)
            return True
        return False
        
    def remove_agent(self, agent: 'Agent') -> None:
        """Remove agent from location"""
        self.agents = [a for a in self.agents if a.name != agent.name]
        
    def get_current_price(self, current_time: int) -> float:
        """Get current price with any active discounts"""
        if not self.discount:
            return self.base_price
            
        day = (current_time // 24) + 1
        if day in self.discount['days']:
            return self.base_price * (1 - self.discount['value'] / 100)
            
        return self.base_price

    def add_to_queue(self, agent: 'Agent') -> str:
        """Add agent to waiting queue"""
        self.queue.append(agent)
        return f"{agent.name} is waiting in line at {self.name}, position {len(self.queue)}"
        
    def process_queue(self) -> Optional[str]:
        """Process waiting queue when space becomes available"""
        while self.queue and not self.is_full():
            next_agent = self.queue.pop(0)
            if self.add_agent(next_agent):
                return f"{next_agent.name} has entered {self.name} from the queue"
        return None

class Agent:
    def __init__(self, name: str, residence: str, family_role: str, 
                 memory_manager: Any, model_manager: Any, prompt_manager: Any, 
                 town_map: Optional[TownMap] = None,
                 all_agents_list_for_perception: Optional[List['Agent']] = None):
        self.name = name
        self.residence = residence
        self.family_role = family_role
        self.current_location: Optional[Location] = None
        self.memory_manager = memory_manager
        self.model_manager = model_manager
        self.prompt_manager = prompt_manager
        self.town_map = town_map
        self.all_agents_list_for_perception: List[Agent] = all_agents_list_for_perception if all_agents_list_for_perception is not None else []
        
        # Basic attributes
        self.energy_level = 100
        self.grocery_level = 100
        self.money = 0
        self.daily_plan = None
        self.plans = []
        self.current_time = 0
        
        # Simplified relationship attributes
        self.relationship_status = None
        self.household_members = []  # List of dicts with name and relationship info
        self.relationships = {}  # Detailed relationship info for all connections
        
        # Other attributes
        self.personal_info = None
        self.workplace = None
        self.occupation = None
        self.work_schedule = None
        self.location_last_purchase = {}
        self.locations: Dict[str, Location] = {} # This will be populated by the main simulation
        self.current_activity = "idle"

        # Attributes for step-based movement
        self.current_grid_position: Optional[Tuple[int, int]] = None
        if self.town_map and self.residence:
            res_coord = self.town_map.get_coordinates_for_location(self.residence)
            if res_coord:
                self.current_grid_position = res_coord
            else:
                print(f"Warning: Agent {self.name}'s residence {self.residence} has no coordinates in town_map.")

        self.is_traveling: bool = False
        self.travel_destination_name: Optional[str] = None
        self.travel_destination_coord: Optional[Tuple[int, int]] = None
        self.travel_path_coords: Optional[List[Tuple[int, int]]] = None
        self.current_path_step_index: int = 0
        self.steps_per_action_phase: int = 3
        self.energy_cost_per_step: int = 1
        
        # For mid-travel decision making
        self.mid_travel_stimulus: Optional[Dict[str, Any]] = None

    def initialize_personal_context(self, person_data: Dict):
        """Initialize agent's personal context from configuration data"""
        try:
            # Set basic attributes
            basics = person_data['basics']
            self.age = basics.get('age', 30)
            self.occupation = basics.get('occupation', 'unemployed')
            
            # Handle multiple workplaces
            workplaces = basics.get('workplace', [])
            if isinstance(workplaces, str):
                workplaces = [workplaces]
            if workplaces:
                self.workplace = workplaces[0]  # Primary workplace
            
            # Set work schedule based on income info
            income_info = basics.get('income', {})
            if income_info:
                self.work_schedule = {
                    'start': 9,  # Default 9 AM
                    'end': 17,   # Default 5 PM
                }
                if income_info.get('type') == 'part-time':
                    self.work_schedule['end'] = 13  # Part-time ends at 1 PM
                
                # Calculate initial money based on income
                income_amount = income_info.get('amount', 0)
                income_type = income_info.get('type', 'hourly')
                if income_type == 'annual':
                    self.money = income_amount / 365  # Daily amount
                elif income_type == 'monthly':
                    self.money = income_amount / 30   # Daily amount
                else:  # hourly
                    self.money = income_amount * 8    # Daily amount for 8 hours
            
            # Set relationships
            relationships = person_data.get('relationships', {})
            self.relationship_status = relationships.get('status', 'single')
            
            # Set living arrangements
            living_with = relationships.get('living_with', [])
            if isinstance(living_with, str):
                living_with = [living_with]
            self.household_members = []
            for member in living_with:
                if isinstance(member, dict):
                    self.household_members.append(member)
                    self.relationships[member['name']] = member.get('relationship', 'household member')
                else:
                    self.household_members.append({'name': member, 'relationship': 'household member'})
                    self.relationships[member] = 'household member'
            
            # Handle spouse if married
            if self.relationship_status == 'married':
                spouse = relationships.get('spouse')
                if spouse:
                    if spouse not in self.relationships:
                        self.relationships[spouse] = 'spouse'
                    # Ensure spouse is also in household_members if they live together (usually true)
                    if not any(mem['name'] == spouse for mem in self.household_members):
                            self.household_members.append({'name': spouse, 'relationship': 'spouse'})
            
            # Store complete personal info for reference
            self.personal_info = person_data
            
            # Set current location to residence initially
            if self.residence in self.locations:
                self.current_location = self.locations[self.residence]
                # Initialize grid position if not already set (e.g. if town_map was None during __init__)
                if self.town_map and self.current_grid_position is None:
                    res_coord = self.town_map.get_coordinates_for_location(self.residence)
                    if res_coord:
                        self.current_grid_position = res_coord
                    else:
                        print(f"Warning (init_personal_context): Agent {self.name}'s residence {self.residence} has no coordinates.")
            
        except Exception as e:
            print(f"Error initializing personal context for {self.name}: {str(e)}")
            traceback.print_exc()

    def log_memory(self, memory_type_key: str, data: Dict):
        """Helper method to log memory for this agent."""
        if memory_type_key not in MEMORY_TYPES:
            print(f"AGENT_ERROR ({self.name}): Invalid memory_type_key '{memory_type_key}' used in log_memory. Data: {data}")
            # Potentially log this error to a generic error memory type if desired
            return
        try:
            # Ensure 'time' is in the data dictionary, as it's crucial for many memory types
            if 'time' not in data:
                data['time'] = self.current_time # Add current agent time if not present
            
            # Pass the memory_type_key directly, assuming MemoryManager expects the key string.
            self.memory_manager.add_memory(self.name, memory_type_key, data)
        except Exception as e:
            print(f"AGENT_ERROR ({self.name}): Failed to log_memory for type '{memory_type_key}'. Error: {e}. Data: {data}")
            traceback.print_exc()

    def get_social_context(self, nearby_agents: List[str]) -> Dict:
        """Get social context for interactions including relationships"""
        context = {
            'nearby_people': [],
            'household_members_present': [],
            'relationship_context': {}
        }
        
        for agent_name in nearby_agents:
            person_context = {
                'name': agent_name,
                'relationship': 'stranger'  # Default relationship
            }
            
            # Check if this is a household member
            for member in self.household_members:
                if member['name'] == agent_name:
                    person_context.update({
                        'relationship': member['relationship'],
                        'duration': member.get('duration', 'unknown'),
                        'living_arrangement': member.get('living_arrangement', 'same_household')
                    })
                    context['household_members_present'].append(agent_name)
                    break
            
            # Add any other relationship info
            if agent_name in self.relationships:
                person_context['relationship'] = self.relationships[agent_name]
            
            context['nearby_people'].append(person_context)
            
        return context

    def generate_conversation(self, context: Dict) -> str:
        """Generate conversation with nearby agents"""
        try:
            # Get social context for all nearby agents
            social_context = self.get_social_context(context['nearby_agents'])
            
            # Get recent interactions with these agents
            recent_interactions = self.memory_manager.retrieve_memories(
                self.name,
                int(context['time']),  # Ensure time is an integer
                memory_type_key='CONVERSATION_LOG_EVENT',
                limit=3
            )
            
            # Get shared activities with these agents
            shared_activities = self.memory_manager.retrieve_memories(
                self.name,
                int(context['time']),  # Ensure time is an integer
                memory_type_key='CONVERSATION_LOG_EVENT',
                limit=3
            )
            
            # Get relationship details
            relationships = {}
            living_arrangements = {}
            shared_history = {}
            for agent_name in context['nearby_agents']:
                rel = self.get_relationship_with(agent_name)
                relationships[agent_name] = rel.get('relationship_type', 'acquaintance')
                living_arrangements[agent_name] = rel.get('living_arrangement', 'separate')
                
                # Get shared history
                shared_mems = self.memory_manager.retrieve_memories(
                    self.name,
                    int(context['time']),  # Ensure time is an integer
                    memory_type_key='CONVERSATION_LOG_EVENT',
                    limit=5
                )
                shared_history[agent_name] = [m['data'].get('content', '') for m in shared_mems]
            
            # Get current location type
            location_type = (
                self.current_location.type 
                if isinstance(self.current_location, Location) 
                else "unknown"
            )
            
            # Get current location name
            location_name = (
                self.current_location.name
                if isinstance(self.current_location, Location)
                else str(self.current_location)
            )
            
            # Generate conversation using enhanced prompt
            conversation = self.model_manager.generate(
                self.prompt_manager.get_prompt(
                    "conversation",
                    name=self.name,
                    location=location_name,
                    time=int(context['time']),  # Ensure time is an integer
                    social_context=str(social_context),
                    nearby_agents=", ".join(context['nearby_agents']),
                    relationships=str(relationships),
                    shared_history=str(shared_history),
                    living_arrangements=str(living_arrangements),
                    location_type=location_type,
                    current_activity=context.get('current_activity', 'general'),
                    recent_interactions=[m['data'].get('content', '') for m in recent_interactions],
                    recent_shared_activities=[m['data'].get('content', '') for m in shared_activities],
                    ongoing_plans=self.daily_plan or "No current plans"
                )
            )
            
            if not conversation or conversation == "Error: Failed to generate response":
                return "Having a quiet moment..."
            
            # Record conversation in memory
            self.memory_manager.add_memory(
                self.name,
                'CONVERSATION_LOG_EVENT',
                {
                    'content': conversation,
                    'time': int(context['time']),  # Ensure time is an integer
                    'location': location_name,
                    'participants': context['nearby_agents'],
                    'social_context': social_context,
                    'relationships': relationships
                }
            )
            
            return conversation
            
        except Exception as e:
            print(f"Error generating conversation for {self.name}: {str(e)}")
            return "Error in conversation"

    def handle_household_interaction(self, current_time: int) -> str:
        """Handle interactions with household members"""
        try:
            # Check living arrangement first
            if not self.household_members:
                return self._handle_solo_evening_routine(current_time)
            
            # Get current location name and type
            current_location_name = (
                self.current_location.name
                if isinstance(self.current_location, Location)
                else str(self.current_location)
            )
            
            # Get household members present at the same location
            present_members = []
            for member in self.household_members:
                member_name = member['name']
                # Check if member is in the same location's agents list
                if any(a.name == member_name for a in self.current_location.agents):
                    present_members.append(member_name)
            
            if not present_members:
                return "No household members present"
            
            # Get recent household activities
            recent_activities = self.memory_manager.retrieve_memories(
                self.name,
                current_time,
                memory_type_key='CONVERSATION_LOG_EVENT',
                limit=3
            )
            activities_str = ", ".join([m['data'].get('content', '') for m in recent_activities])
            
            # Generate household coordination using prompt
            interaction = self.model_manager.generate(
                self.prompt_manager.get_prompt(
                    "household_coordination",
                    name=self.name,
                    location=current_location_name,
                    time=current_time % 24,
                    members=", ".join(present_members),
                    recent_activities=activities_str,
                    work_schedules=str(self.work_schedule),
                    shared_meals="Planned dinner together",
                    personal_activities="Various individual activities",
                    relationships=str(self.relationships)
                )
            )
            
            # Record interaction in memory
            self.memory_manager.add_memory(
                self.name,
                'CONVERSATION_LOG_EVENT',
                {
                    'content': interaction,
                    'activity_description': interaction,
                    'time': current_time,
                    'location': current_location_name,
                    'household_members': present_members
                }
            )
            
            return interaction
            
        except Exception as e:
            print(f"Error in household interaction for {self.name}: {str(e)}")
            return "Quiet time at home due to error"

    def _handle_solo_evening_routine(self, current_time: int) -> str:
        """Handle evening routine for people living alone"""
        try:
            # Get recent activities
            recent_activities = self.memory_manager.retrieve_memories(
                self.name,
                current_time,
                memory_type_key='ACTIVITY_EVENT',
                limit=3
            )
            
            # Get next day's schedule from daily plan
            next_day_schedule = self.daily_plan or "No specific plans for tomorrow"
            
            # Generate routine using prompt
            routine = self.model_manager.generate(
                self.prompt_manager.get_prompt(
                    "solo_evening_routine",
                    name=self.name,
                    location=self.get_current_location_name(),
                    time=current_time % 24,
                    energy=self.energy_level,
                    grocery_level=self.grocery_level,
                    recent_activities=str([m['data'].get('content', '') for m in recent_activities]),
                    next_day_schedule=next_day_schedule
                )
            )
            
            # Record the routine in memory
            self.memory_manager.add_memory(
                self.name,
                'CONVERSATION_LOG_EVENT',
                {
                    'content': routine,
                    'time': current_time,
                    'location': self.get_current_location_name(),
                    'household_members': [],
                    'relationship_context': {}
                }
            )
            
            return routine
            
        except Exception as e:
            print(f"Error in solo routine for {self.name}: {str(e)}")
            return "Error in solo routine"

    def needs_food(self, current_time: int) -> bool:
        """Check if agent needs food based on energy level"""
        try:
            # Very hungry if energy is low
            if self.energy_level <= 30:
                return True
                
            # Check if it's meal time
            is_meal_time, meal_type = self.is_meal_time(current_time)
            if is_meal_time:
                # Check if we've already had this meal today
                today_memories = self.memory_manager.get_memories_for_day(agent_name=self.name, current_sim_time=current_time)
                already_had_meal = any(
                    meal_type.lower() in memory_data['data'].get('content', '').lower() 
                    for memory_data in today_memories if isinstance(memory_data, dict) and 'data' in memory_data and isinstance(memory_data['data'], dict)
                )
                if not already_had_meal:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking food needs for {self.name}: {str(e)}")
            return False

    def is_meal_time(self, current_time: int) -> tuple[bool, Optional[str]]:
        """Check if current time is during a meal period"""
        hour = current_time % 24
        
        # Define meal time windows
        meal_times = {
            'breakfast': range(6, 9),    # 6-8 AM
            'lunch': range(11, 14),      # 11 AM - 1 PM
            'dinner': range(17, 20)      # 5-7 PM
        }
        
        # Check which meal time it is
        for meal_type, time_range in meal_times.items():
            if hour in time_range:
                return True, meal_type
                
        return False, None

    def make_meal(self, meal_type: str) -> Dict:
        """Make and consume a meal, either from groceries or purchased food"""
        try:
            result = {
                'success': False,
                'message': '',
                'energy_gained': 0,
                'grocery_used': 0
            }
            
            # Making meal at home from groceries
            if self.current_location == self.residence:
                if self.grocery_level >= 10:
                    self.grocery_level -= 10  # Use groceries
                    self.energy_level = min(100, self.energy_level + 50)  # All meals give +50 energy
                    result.update({
                        'success': True,
                        'message': f'Made {meal_type} at home using groceries',
                        'energy_gained': 50,
                        'grocery_used': 10
                    })
                else:
                    result['message'] = 'Not enough groceries to make meal'
            else:
                result['message'] = 'Not at home to make meal'
            
            # Record meal in memory
            if result['success']:
                self.memory_manager.add_memory(
                    self.name,
                    'ACTIVITY_EVENT',
                    {
                        'content': result['message'],
                        'time': self.current_time,
                        'location': str(self.current_location),
                        'meal_type': meal_type,
                        'energy_gained': result['energy_gained'],
                        'grocery_used': result['grocery_used']
                    }
                )
            
            return result
            
        except Exception as e:
            print(f"Error making meal for {self.name}: {str(e)}")
            return {
                'success': False,
                'message': 'Error making meal',
                'energy_gained': 0,
                'grocery_used': 0
            }

    def calculate_grocery_purchase_amount(self, available_money: float, base_price: float) -> int:
        """Calculate how many grocery points to purchase based on available money and price"""
        # Each point costs base_price
        max_points = int(available_money / base_price)
        # Round down to nearest 10
        points_to_add = (max_points // 10) * 10
        # Cap at 100 total grocery level
        points_possible = 100 - self.grocery_level
        return min(points_to_add, points_possible)

    def make_purchase(self, location_name, item_type='food', item_description='generic item', items_list=None, quantity: Optional[int] = None):
        """Make a purchase at the specified location"""
        try:
            if location_name not in self.locations:
                print(f"Error: {location_name} not found in locations")
                return {'success': False, 'message': f"Location {location_name} not found.", 'cost': 0, 'quantity': 0}
                
            location = self.locations[location_name]
            
            current_day = (self.current_time // 24) + 1
            used_discount_flag = False

            price_per_item_or_unit = location.get_current_price(self.current_time) # Already discount-adjusted
            
            final_cost = 0
            actual_quantity_purchased = 0 # For groceries or single items

            if item_type == 'food':
                final_cost = price_per_item_or_unit # Cost of one meal
                actual_quantity_purchased = 1
            elif item_type == 'groceries':
                if quantity is None or quantity <= 0:
                    return {'success': False, 'message': f"Invalid quantity ({quantity}) for grocery purchase at {location_name}.", 'cost': 0, 'quantity': 0}
                final_cost = price_per_item_or_unit * quantity
                actual_quantity_purchased = quantity
            else: # Fallback for other item types if not 'food' or 'groceries'
                 return {'success': False, 'message': f"Unknown item_type '{item_type}' for purchase at {location_name}.", 'cost': 0, 'quantity': 0}

            if location.base_price > price_per_item_or_unit : # If current price is less than base, discount was likely applied
                used_discount_flag = True
            
            if self.money < final_cost:
                return {'success': False, 'message': f"{self.name} doesn't have enough money (${self.money:.2f}) for purchase at {location_name} (cost: ${final_cost:.2f})", 'cost': final_cost, 'quantity': 0}
                
            self.money -= final_cost
            
            memory_key_for_purchase = 'FOOD_PURCHASE_EVENT' # Default string key for MemoryManager
            
            purchase_data_details = {
                'item_description': item_description, 
                'items_list': [], 
                'amount': final_cost, 
                'location': location_name,
                'time': self.current_time,
                'day': current_day,
                'used_discount': used_discount_flag,
                'quantity': actual_quantity_purchased
            }

            if item_type == 'food':
                self.energy_level = min(100, self.energy_level + 50)
                memory_key_for_purchase = 'FOOD_PURCHASE_EVENT'
                purchase_data_details['item_description'] = item_description if item_description else "Prepared Food"
            elif item_type == 'groceries': # quantity is already validated > 0 and actual_quantity_purchased is set
                self.grocery_level = min(100, self.grocery_level + actual_quantity_purchased)
                memory_key_for_purchase = 'GROCERY_PURCHASE_EVENT'
                purchase_data_details['items_list'] = items_list if items_list else [f"{actual_quantity_purchased} units of basic groceries"]
                purchase_data_details['item_description'] = "Groceries purchase"


            if hasattr(self, 'metrics') and self.metrics:
                self.metrics.record_interaction(
                    self.name,
                    location_name,
                    "purchase",
                    {
                        'purchase_type_tag': item_type, 
                        'description_for_metrics': purchase_data_details['item_description'], 
                        'items_for_metrics': purchase_data_details.get('items_list', []),                        
                        'location': location_name,
                        'agent': self.name,
                        'amount': final_cost, 
                        'time': self.current_time,
                        'day': current_day,
                        'used_discount': used_discount_flag,
                        'quantity': actual_quantity_purchased
                    }
                )
            
            self.memory_manager.add_memory(
                self.name,
                memory_key_for_purchase, # Use the string key
                purchase_data_details
            )
            
            return {'success': True, 'message': f"{self.name} purchased {purchase_data_details['item_description']} for ${final_cost:.2f} at {location_name}.", 'cost': final_cost, 'quantity': actual_quantity_purchased}
            
        except Exception as e:
            print(f"Error in make_purchase for {self.name} at {location_name}: {str(e)}")
            traceback.print_exc()
            return {'success': False, 'message': f"Error during purchase attempt at {location_name}.", 'cost': 0, 'quantity':0 }

    def record_store_visit(self, location_name: str) -> None:
        """Record a visit to a store location"""
        try:
            if location_name == "Fried Chicken Shop":
                visit_data = {
                    'agent': self.name,
                    'time': self.current_time,
                    'day': (self.current_time // 24) + 1,
                    'hour': self.current_time % 24
                }
                
                # Record in metrics if available
                if hasattr(self, 'metrics'):
                    self.metrics.record_interaction(
                        self.name,
                        location_name,
                        "store_visit",
                        visit_data
                    )
                
                # Record in memory
                self.memory_manager.add_memory(
                    self.name,
                    'ACTIVITY_EVENT',
                    {
                        'location': location_name,
                        'time': self.current_time,
                        'content': f"Visited {location_name}"
                    }
                )
        except Exception as e:
            print(f"Error recording store visit for {self.name}: {str(e)}")

    def handle_food_needs(self, current_time: int) -> str:
        """Unified food handling system with grocery integration"""
        try:
            # Check if agent needs food
            if not self.needs_food(current_time):
                return "Don't need food right now"
            
            # Get meal type for current time
            _, meal_type = self.is_meal_time(current_time)
            
            # If at home and have groceries, make meal
            if self.current_location.name == self.residence and self.grocery_level >= 10:
                meal_result = self.make_meal(meal_type or "snack")
                return meal_result['message']
            
            # If at a food location, make a purchase
            if isinstance(self.current_location, Location):
                if self.current_location.type in ['local_shop', 'grocery']:
                    if self.current_location.type == 'local_shop':
                        # Make food purchase - can be dine-in or takeout
                        purchase_result = self.make_purchase(self.current_location.name, 'food')
                        if purchase_result:
                            # Decide between dine-in and takeout based on time and social context
                            hour = current_time % 24
                            is_busy_hour = hour in [8, 9, 12, 13, 17, 18]  # Common work/rush hours
                            if is_busy_hour or self.energy_level < 40:
                                return f"I get takeout from {self.current_location.name} to save time"
                            else:
                                return f"I enjoy my meal at {self.current_location.name}"
                    else:
                        # Buy groceries to take home
                        purchase_result = self.make_purchase(self.current_location.name, 'groceries')
                        if purchase_result:
                            return f"I buy groceries at {self.current_location.name} to cook at home"
            
            # If at home but no groceries, need to go shopping or find food
            if self.current_location.name == self.residence and self.grocery_level < 10:
                # Decide whether to shop or eat out based on time and money
                hour = current_time % 24
                if hour < 10:  # Morning hours
                    # Early morning - go to coffee shop or diner
                    return self.move_to_location("The Coffee Shop" if hour < 8 else "Local Diner")
                else:
                    # Later hours - go shopping for groceries
                    return self.move_to_location("Target" if hour >= 8 and hour < 22 else "Local Market")
            
            # If none of the above, move to closest food location
            closest_food = self.find_closest_food_location(current_time)
            # This should now trigger travel, not instant move
            # return self.move_to_location(closest_food)
            if self.start_travel_to(closest_food):
                return f"Decided to travel to {closest_food} for food."
            else:
                return f"Could not start travel to {closest_food} for food."
            
        except Exception as e:
            print(f"Error in handle_food_needs for {self.name}: {str(e)}")
            return "Error handling food needs"

    def find_closest_food_location(self, current_time: int) -> str:
        """Find the closest available food location based on time"""
        hour = current_time % 24
        
        # Early morning (before 8 AM)
        if hour < 8:
            return "The Coffee Shop"  # Opens earliest at 6 AM
        
        # Morning to late night
        if hour < 22:
            if self.grocery_level < 10:
                return "Target"  # For groceries
            else:
                return "Local Diner"  # For prepared food
        
        # Late night
        return "Local Market"  # Usually open latest

    def move_to_location(self, location_name: str) -> str:
        """Deprecated: Use start_travel_to for grid-based movement."""
        # This method is now largely superseded by start_travel_to and _perform_travel_step
        # For direct, non-grid based movement (if ever needed again, or for fallback),
        # it would need to be re-evaluated.
        # For now, let's assume all movement is grid-based if town_map exists.
        # print(f"Warning: agent.move_to_location('{location_name}') called. This should use start_travel_to.")
        if self.town_map:
            if self.start_travel_to(location_name):
                return f"Initiating travel to {location_name}."
            else:
                return f"Failed to initiate travel to {location_name}."
        else: # Fallback to old logic if no town_map (should not happen in normal grid sim)
            try: # Ensuring this try has its block correctly indented
                if location_name in self.locations:
                    new_location = self.locations[location_name]
                    if not new_location.is_open(self.current_time % 24):
                        return f"Cannot move to {location_name} as it is closed"
                    
                    if not new_location.is_full():
                        if self.current_location:
                            self.current_location.remove_agent(self)
                        if new_location.add_agent(self):
                            self.current_location = new_location
                            # Log instant move
                            # Ensuring this line is correctly indented
                            self.memory_manager.add_memory(self.name, MEMORY_TYPES['LOCATION_CHANGE_EVENT'], {'time': self.current_time, 'location': location_name, 'content': f"Instantly moved to {location_name}"})
                            return f"Successfully moved to {location_name}"
                    else:
                        return new_location.add_to_queue(self)
                        
                return f"Cannot move to {location_name} as it does not exist"
            except Exception as e:
                print(f"Error in fallback move_to_location: {str(e)}")
                return "Error moving to location (fallback)"

    def get_current_location_name(self) -> str:
        """Get the name of the current location"""
        if self.current_location:
            return self.current_location.name
        return self.residence  # Default to residence if no current location

    def _calculate_energy_change(self, action_type: str, action_content: Optional[str] = None) -> int:
        """Calculate energy change based on activity type and content"""
        # Base energy loss per hour is -10
        energy_change = -10
        
        # Additional energy cost for main activities
        if action_type in [
            'EDUCATION',
            'PURCHASE',
            'RECREATION',
            'WORK'
        ]:
            energy_change -= 5  # Additional -5 for these activities
        
        # Energy recharge from eating
        if action_type == 'DINING' or (
            action_content and any(word in action_content.lower() for word in ['eat', 'lunch', 'dinner', 'breakfast', 'meal'])
        ):
            return +50  # Eating recharges energy significantly
        
        # Household activities don't cost additional energy
        if action_type == 'HOUSEHOLD':
            return -10  # Just the base hourly cost
        
        return energy_change 

    def create_daily_plan(self, current_time: int, planning_context: Optional[Dict] = None) -> str:
        """Create a daily plan for the agent"""
        try:
            day = (current_time // 24) + 1
            hour = current_time % 24
            
            # Get recent activities
            recent_activities = self.memory_manager.retrieve_memories(
                self.name, 
                current_time,
                limit=5
            )
            
            # Build comprehensive context string
            context_parts = []
            
            # Add work schedule and location
            if hasattr(self, 'work_schedule'):
                context_parts.append(f"Work schedule: {self.work_schedule}")
                context_parts.append(f"Work location: {self.workplace}")
            
            # Add recent activities
            activities_str = ", ".join([m['data'].get('content', '') for m in recent_activities])
            if activities_str:
                context_parts.append(f"Recent activities: {activities_str}")
            
            # Add social interactions if available
            if planning_context and 'social_interactions' in planning_context:
                social_str = ", ".join([m.get('content', '') for m in planning_context['social_interactions']])
                if social_str:
                    context_parts.append(f"Today's social interactions: {social_str}")
            
            # Add household plans if available
            if planning_context and 'household_plans' in planning_context:
                household_str = ", ".join([m.get('content', '') for m in planning_context['household_plans']])
                if household_str:
                    context_parts.append(f"Household coordination: {household_str}")
            
            # Add current status
            current_loc_str_name = self.get_current_location_name()
            context_parts.append(f"Current location: {current_loc_str_name}")
            context_parts.append(f"Energy level: {self.energy_level}")
            context_parts.append(f"Grocery level: {self.grocery_level}")
            
            # Add available locations
            available_locations = [loc for loc in self.locations.values() 
                                if loc.type not in ['residence', 'community']]
            context_parts.append(f"Available locations: {[loc.name for loc in available_locations]}")
            
            # Combine all context
            full_context = " | ".join(context_parts)
            
            # Generate plan using prompt
            prompt_for_llm = self.prompt_manager.get_prompt(
                    "daily_plan",
                    name=self.name,
                    day=day,
                    time=hour,
                    recent_activities=full_context,
                    location=current_loc_str_name,
                    energy=self.energy_level,
                    grocery_level=self.grocery_level,
                    available_locations=str([loc.name for loc in available_locations])
                )
            print(f"AGENT_LOG ({self.name}): Generating daily plan. About to call LLM at {datetime.now().strftime('%H:%M:%S')}. Prompt sample: {prompt_for_llm[:150]}...")
            
            plan = self.model_manager.generate(prompt_for_llm)
            
            print(f"AGENT_LOG ({self.name}): LLM call for daily plan returned at {datetime.now().strftime('%H:%M:%S')}. Raw plan: {plan}")
            
            self.daily_plan = plan
            # Log the plan to memory
            if plan:
                self.log_memory('PLANNING_EVENT', {
                    'time': current_time, # Keep specific current_time for planning event
                    'plan_content': plan,
                    'description': f'{self.name} generated their daily plan.'
                })
            else:
                 self.log_memory('PLANNING_EVENT', {
                    'time': current_time, # Keep specific current_time for planning event
                    'plan_content': 'Failed to generate plan',
                    'description': f'{self.name} failed to generate their daily plan.'
                })

            return plan
            
        except Exception as e:
            print(f"Error creating daily plan for {self.name}: {str(e)}")
            traceback.print_exc()  # Add this to get more detailed error info
            return None

    def needs_groceries(self, current_time: int) -> bool:
        """Check if agent needs to buy groceries"""
        try:
            # Priority 1: Very low on groceries, definitely need to shop.
            if self.grocery_level < 10:
                return True
                
            # Priority 2: Evening and getting low, good time to restock for dinner/next day.
            hour = current_time % 24
            if hour >= 17 and hour <= 20 and self.grocery_level < 20:
                return True
                
            # Priority 3: Planning to cook and groceries are not abundant.
            # Only consider this if not already covered by higher priority needs.
            if self.daily_plan and any(word in self.daily_plan.lower() for word in ['cook', 'make food', 'prepare meal', 'make dinner']):
                if self.grocery_level < 50: # e.g., less than half stock and planning to cook
                    return True
            # General threshold: If groceries are above a comfortable level (e.g., 30%), no immediate need unless specific conditions above met.
            if self.grocery_level > 30:
                return False
            # Fallback: if below 30 but no specific plan/evening condition, it's a moderate need.
            if self.grocery_level <= 30: # Catches the 10-30 range if other conditions didn't apply.
                return True
                
            return False # Default to false if no condition met
            
        except Exception as e:
            print(f"Error checking grocery needs for {self.name}: {str(e)}")
            return False

    def generate_contextual_action(self, context: Dict) -> str:
        self.current_time = context.get("current_simulation_time", self.current_time)
        current_hour = self.current_time % 24
        action_description = f"{self.name} is considering what to do."

        try:
            # Handle travel and encounters
            if self.is_traveling and self.mid_travel_stimulus:
                stimulus = self.mid_travel_stimulus
                self.mid_travel_stimulus = None 

                if stimulus['type'] == "location_encounter":
                    loc_obj: Location = stimulus['details']
                    
                    # Calculate time to next commitment and destination urgency
                    time_to_next = self._calculate_time_to_next_commitment(context['time'])
                    destination_urgency = self._calculate_destination_urgency(context['time'])
                    
                    # Get location relevance and history
                    location_relevance = self._assess_location_relevance(loc_obj, context)
                    last_visit = self._get_last_visit_time(loc_obj.name)
                    
                    prompt_context = {
                        'agent_name': self.name,
                        'original_destination_name': self.travel_destination_name,
                        'encountered_location_name': loc_obj.name,
                        'encountered_location_type': loc_obj.type,
                        'energy_level': self.energy_level,
                        'grocery_level': self.grocery_level,
                        'money': self.money,
                        'time': context['time'],
                        'time_to_next_commitment': time_to_next,
                        'destination_urgency': destination_urgency,
                        'daily_plan': self.daily_plan or "No specific plan.",
                        'location_offers': self._get_location_offers(loc_obj),
                        'location_relevance': location_relevance,
                        'last_visit_time': last_visit
                    }
                    
                    llm_decision_str = self.model_manager.generate(
                        self.prompt_manager.get_prompt('mid_travel_location_decision', **prompt_context)
                    )
                    self.log_memory('GENERIC_EVENT', {'time': self.current_time, 'content': f"Mid-travel decision at {loc_obj.name}. LLM output: {llm_decision_str}"})

                    # Parse the decision with new options
                    decision_match = re.match(r"([ABC]):", llm_decision_str.strip(), re.IGNORECASE)
                    if decision_match:
                        choice = decision_match.group(1).upper()
                        reason = llm_decision_str.split(":", 1)[1].strip() if ":" in llm_decision_str else "No reason provided"
                        
                        if choice == "A":  # Rush to destination
                            action_description = f"Decided to rush to {self.travel_destination_name}. Reason: {reason}"
                            self.energy_level = max(0, self.energy_level - 2)  # Extra energy cost for rushing
                            
                        elif choice in ["B", "C"]:  # Quick stop or Regular visit
                            visit_type = "quick" if choice == "B" else "regular"
                            action_description = f"Decided to make a {visit_type} stop at {loc_obj.name}. {reason}"
                            self.log_memory('GENERIC_EVENT', {
                                'time': self.current_time, 
                                'event_subtype': 'travel_interruption', 
                                'visit_type': visit_type,
                                'reason': reason, 
                                'new_destination': loc_obj.name, 
                                'original_destination': self.travel_destination_name, 
                                'content': action_description
                            })
                            
                            # Store original destination for resuming later
                            original_dest = self.travel_destination_name
                            original_coord = self.travel_destination_coord
                            
                            # Divert to the encountered location
                            self.travel_destination_name = loc_obj.name
                            self.travel_destination_coord = self.current_grid_position
                            arrival_at_diversion = self._complete_travel()
                            action_description += f" {arrival_at_diversion}"
                            
                            # For quick stops, automatically set up return travel after a short delay
                            if choice == "B":
                                self.log_memory('ACTIVITY_EVENT', {
                                    'time': self.current_time,
                                    'activity_type_tag': ACTIVITY_TYPES['QUICK_STOP'],
                                    'location': loc_obj.name,
                                    'duration': '5-10 minutes',
                                    'content': f"Making a quick stop at {loc_obj.name}"
                                })
                                # Will resume travel to original destination after this action phase
                                self.pending_travel_resume = {
                                    'destination': original_dest,
                                    'coordinate': original_coord,
                                    'after_quick_stop': True
                                }
                            
                        self.current_activity = action_description
                        return action_description
                    
                    # Default to continuing if decision parsing fails
                    action_description = f"Continuing to {self.travel_destination_name} due to unclear decision."

                elif stimulus['type'] == "agent_encounter":
                    encountered_agents = stimulus['encountered_agents']
                    encountered_agent_names = [a.name for a in encountered_agents]
                    
                    # Get relationship and history info
                    relationships = {a.name: self.get_relationship_with(a.name) for a in encountered_agents}
                    shared_history = self._get_shared_history(encountered_agent_names)
                    time_to_next = self._calculate_time_to_next_commitment(context['time'])
                    
                    # Calculate destination urgency
                    destination_urgency = self._calculate_destination_urgency(context['time'])
                    
                    decision_context = {
                        "agent_name": self.name,
                        "original_destination_name": self.travel_destination_name,
                        "current_grid_coord": str(self.current_grid_position),
                        "encountered_agent_names_list": ", ".join(encountered_agent_names),
                        "encountered_agent_details_list": self._get_agent_details_str(encountered_agents),
                        "energy_level": self.energy_level,
                        "time": context['time'],
                        "daily_plan": self.daily_plan or "No specific plan.",
                        "relationships": str(relationships),
                        "shared_history": str(shared_history),
                        "destination_urgency": destination_urgency,
                        "time_to_next_commitment": time_to_next
                    }
                    
                    llm_decision_raw = self.model_manager.generate(
                        self.prompt_manager.get_prompt('mid_travel_agent_encounter_decision', **decision_context)
                    )
                    
                    choice_match = re.search(r"CHOICE:\\s*([ABC])", llm_decision_raw, re.IGNORECASE)
                    reason_match = re.search(r"REASON:\\s*(.*)", llm_decision_raw, re.IGNORECASE)
                    
                    parsed_llm_choice = choice_match.group(1).upper() if choice_match else "A"
                    reason_text = reason_match.group(1).strip() if reason_match else "No specific reason provided."
                    
                    if parsed_llm_choice == "B":
                        action_description = f"{self.name} decided to stop traveling to potentially converse with {', '.join(encountered_agent_names)}. Reason: {reason_text}"
                        self.is_traveling = False
                        self.travel_path_coords = None 
                        self.current_path_step_index = 0
                        self.log_memory('ACTIVITY_EVENT', {'time': self.current_time, 'activity_type_tag': ACTIVITY_TYPES['TRAVEL'], 'description': action_description, 'involved_agents': encountered_agent_names})
                        self.current_activity = action_description
                        return self.current_activity
                    else:
                        action_description = f"{self.name} decided to {'briefly acknowledge' if parsed_llm_choice == 'A' else 'ignore'} {', '.join(encountered_agent_names)} and continue. Reason: {reason_text}"
                        self.log_memory('GENERIC_EVENT', {'time': self.current_time, 'content': action_description})
            
            # Continue travel if in progress
            if self.is_traveling:
                # Check if we need to resume travel after a quick stop
                if hasattr(self, 'pending_travel_resume') and self.pending_travel_resume:
                    resume_data = self.pending_travel_resume
                    self.pending_travel_resume = None  # Clear the pending resume
                    if resume_data['after_quick_stop']:
                        action_description = self.start_travel_to(resume_data['destination'])
                        self.log_memory('ACTIVITY_EVENT', {
                            'time': self.current_time,
                            'activity_type_tag': ACTIVITY_TYPES['TRAVEL'],
                            'description': f"Resuming travel to {resume_data['destination']} after quick stop.",
                            'destination': resume_data['destination']
                        })
                        return action_description

                total_log_messages = []
                needs_decision_after_steps = False
                for _ in range(self.steps_per_action_phase):
                    if not self.is_traveling:
                        break
                    step_log_msg, needs_mid_travel_decision_flag = self._perform_travel_step()
                    if step_log_msg:
                        total_log_messages.append(step_log_msg)
                    if needs_mid_travel_decision_flag:
                        needs_decision_after_steps = True
                        break
                    if not self.is_traveling:
                        break

                if total_log_messages:
                    action_description = f"{self.name} continued traveling: {' '.join(total_log_messages)}"
                elif self.is_traveling:
                    action_description = f"{self.name} continued traveling towards {self.travel_destination_name}."
                elif self.travel_destination_name:
                    action_description = f"{self.name} finished traveling and arrived at {self.travel_destination_name}."

                if needs_decision_after_steps and self.mid_travel_stimulus:
                    stim_desc = self.mid_travel_stimulus['location_name'] if self.mid_travel_stimulus['type'] == 'location_encounter' else f"agents: {', '.join(a.name for a in self.mid_travel_stimulus['encountered_agents'])}"
                    action_description += f" Encountered {stim_desc}."

                self.current_activity = action_description
                return action_description

            # Create daily plan at 7 AM
            if current_hour == 7 and not self.daily_plan:
                action_description = self.create_daily_plan(self.current_time, context)
                self.current_activity = action_description
                return action_description

            # Handle work schedule
            if self.occupation != 'unemployed' and self.work_schedule and self.workplace:
                if self.work_schedule['start'] <= current_hour < self.work_schedule['end']:
                    if self.get_current_location_name() != self.workplace:
                        action_description = self.start_travel_to(self.workplace)
                        self.log_memory('ACTIVITY_EVENT', {'time': self.current_time, 'activity_type_tag': ACTIVITY_TYPES['TRAVEL'], 'description': f"Traveling to work at {self.workplace}.", 'destination': self.workplace})
                    else:
                        action_description = f"{self.name} is working at {self.workplace}."
                        self.energy_level -= 5
                        self.log_memory('ACTIVITY_EVENT', {'time': self.current_time, 'activity_type_tag': ACTIVITY_TYPES['WORK'], 'description': action_description, 'location': self.workplace, 'energy_change': -5})
                    self.current_activity = action_description
                    return action_description

            # Handle food needs
            if self.needs_food(self.current_time):
                action_description = self.handle_food_needs(self.current_time)
                self.current_activity = action_description
                return action_description

            # Handle grocery needs
            if self.needs_groceries(self.current_time):
                grocery_store_name = self.find_closest_food_location(self.current_time)
                if grocery_store_name:
                    if self.get_current_location_name() != grocery_store_name:
                        action_description = self.start_travel_to(grocery_store_name)
                        self.log_memory('ACTIVITY_EVENT', {'time': self.current_time, 'activity_type_tag': ACTIVITY_TYPES['TRAVEL'], 'description': f"Traveling to {grocery_store_name} for groceries.", 'destination': grocery_store_name})
                    else:
                        grocery_loc = self.locations[grocery_store_name]
                        units_to_buy = self.calculate_grocery_purchase_amount(self.money, grocery_loc.get_current_price(self.current_time))
                        if units_to_buy > 0:
                            purchase_res = self.make_purchase(grocery_store_name, item_type='groceries', items_list=[f"{units_to_buy} units of basic groceries"], quantity=units_to_buy)
                            action_description = purchase_res['message']
                        else:
                            action_description = f"{self.name} is at {grocery_store_name}, but decided not to buy groceries now."
                            self.record_store_visit(grocery_store_name)
                else:
                    action_description = f"{self.name} needs groceries, but no grocery store is available/open."
                    self.log_memory('SYSTEM_EVENT', {'time': self.current_time, 'content': action_description})
                self.current_activity = action_description
                return action_description

            # Handle evening activities
            if current_hour >= 18 and self.get_current_location_name() == self.residence:
                if self.household_members and len(self.household_members) > 0:
                    other_members_exist = any(mem['name'] != self.name for mem in self.household_members)
                    action_description = self.handle_household_interaction(self.current_time) if other_members_exist else self._handle_solo_evening_routine(self.current_time)
                else:
                    action_description = self._handle_solo_evening_routine(self.current_time)
                self.current_activity = action_description
                return action_description

            # Default actions
            if self.get_current_location_name() == self.residence:
                # Night time resting (10 PM - 6 AM)
                if current_hour >= 22 or current_hour < 6:
                    action_description = f"{self.name} is sleeping at home."
                    self.energy_level = min(100, self.energy_level + 10)  # More energy recovery during sleep
                    self.log_memory('ACTIVITY_EVENT', {
                        'time': self.current_time, 
                        'activity_type_tag': ACTIVITY_TYPES['RESTING'], 
                        'description': action_description, 
                        'location': self.residence, 
                        'energy_change': 10
                    })
                else:
                    # Regular resting during day
                    action_description = f"{self.name} is resting at home ({self.residence})."
                    self.energy_level = min(100, self.energy_level + 5)
                    self.log_memory('ACTIVITY_EVENT', {
                        'time': self.current_time, 
                        'activity_type_tag': ACTIVITY_TYPES['RESTING'], 
                        'description': action_description, 
                        'location': self.residence, 
                        'energy_change': 5
                    })
            else:
                # If it's very late and not at home, try to return home
                if current_hour >= 23 or current_hour < 5:
                    action_description = self.start_travel_to(self.residence)
                    self.log_memory('ACTIVITY_EVENT', {
                        'time': self.current_time, 
                        'activity_type_tag': ACTIVITY_TYPES['TRAVEL'], 
                        'description': f"Heading home to {self.residence} for the night.", 
                        'destination': self.residence
                    })
                else:
                    action_description = self.start_travel_to(self.residence)
                    self.log_memory('ACTIVITY_EVENT', {
                        'time': self.current_time, 
                        'activity_type_tag': ACTIVITY_TYPES['TRAVEL'], 
                        'description': f"Traveling home to {self.residence}.", 
                        'destination': self.residence
                    })
            
            self.current_activity = action_description
            return action_description

        except Exception as e:
            print(f"Error in action generation for {self.name}: {str(e)}")
            traceback.print_exc()
            self.is_traveling = False
            return "I am assessing my surroundings and deciding what to do next after an unexpected event."

    def process_action_result(self, action_result: Dict):
        if action_result.get("energy_change"): self.energy_level = min(100, max(0, self.energy_level + action_result["energy_change"]))
        if action_result.get("money_change"): self.money += action_result["money_change"]
        if action_result.get("grocery_change"): self.grocery_level = min(100, max(0, self.grocery_level + action_result["grocery_change"]))
        if any(key in action_result for key in ["energy_change", "money_change", "grocery_change"]):
            # Use self.log_memory for consistency and to ensure time is added automatically
            log_data = {
                'energy_level': self.energy_level,
                'money': self.money,
                'grocery_level': self.grocery_level, 
                'triggering_action_result': action_result.get("message", "Unknown action effect")
            }
            self.log_memory('AGENT_STATE_UPDATE_EVENT', log_data)

    def get_relationship_with(self, other_agent_name: str) -> Optional[str]: return self.relationships.get(other_agent_name)
    def get_household_relationship(self, other_agent_name: str) -> Optional[Dict]:
         for member in self.household_members: 
             if member['name'] == other_agent_name: return member
         return None

    def should_interact_with(self, other_agent: 'Agent', current_time: int) -> bool:
        if any(mem['name'] == other_agent.name for mem in self.household_members): return True
        relationship = self.get_relationship_with(other_agent.name)
        if relationship in ['spouse', 'partner', 'close friend']: return True
        if self.is_traveling or self.energy_level < 30:
            if relationship not in ['spouse', 'partner']: return False
        if relationship: return True
        return False

    def generate_relationship_interaction(self, other_agent: 'Agent', context: Dict) -> str:
        interaction_context = {
            "agent_name": self.name, "other_agent_name": other_agent.name,
            "relationship": self.get_relationship_with(other_agent.name) or "acquaintance",
            "current_location": self.get_current_location_name(), "current_time": self.current_time,
            "shared_context": context.get("shared_context", "general encounter"),
            "agent1_recent_activity": self.get_recent_activities(limit=1),
            "agent2__recent_activity": other_agent.get_recent_activities(limit=1) if hasattr(other_agent, 'get_recent_activities') else "Unknown"
        }
        prompt = self.prompt_manager.get_prompt("relationship_interaction", **interaction_context)
        self.memory_manager.add_memory(self.name, 'ACTION_RAW_OUTPUT', {'time': self.current_time, 'action_type': 'relationship_interaction_prompt', 'content': prompt, 'target_agent': other_agent.name})
        interaction_text = self.model_manager.generate(prompt)
        self.memory_manager.add_memory(self.name, 'CONVERSATION_LOG_EVENT', {'time': self.current_time, 'location': self.get_current_location_name(), 'participants': [self.name, other_agent.name], 'content': interaction_text, 'context': f"Interaction between {self.name} and {other_agent.name}."})
        return f"{self.name} interacts with {other_agent.name}: {interaction_text}"

    def update_state(self):
        """Update agent's state based on current conditions and log it."""
        state_data = {
            'location': self.get_current_location_name(),
            'energy_level': self.energy_level,
            'money': self.money,
            'current_activity': self.current_activity,
            'grocery_level': self.grocery_level,
            'is_traveling': self.is_traveling,
            'travel_destination': self.travel_destination_name if self.is_traveling else None
        }
        self.log_memory('AGENT_STATE_UPDATE_EVENT', state_data)

    def can_afford(self, amount: float) -> bool: return self.money >= amount
    def spend_money(self, amount: float) -> bool:
        if self.can_afford(amount): self.money -= amount; return True
        return False

    def get_recent_activities(self, limit: int = 3) -> List[str]:
        activity_mems = self.memory_manager.retrieve_memories(self.name, self.current_time, memory_type_key='ACTIVITY_EVENT', limit=limit)
        convo_mems = self.memory_manager.retrieve_memories(self.name, self.current_time, memory_type_key='CONVERSATION_LOG_EVENT', limit=limit)
        combined_mems = sorted(activity_mems + convo_mems, key=lambda m: m['simulation_time'], reverse=True)
        descriptions = []
        for mem in combined_mems[:limit]:
            if mem['type'] == MEMORY_TYPES['ACTIVITY_EVENT']: descriptions.append(mem['data'].get('description', 'Unknown activity'))
            elif mem['type'] == MEMORY_TYPES['CONVERSATION_LOG_EVENT']:
                participants = mem['data'].get('participants', []); other_participants = [p for p in participants if p != self.name]
                if other_participants: desc = f"Conversed with {', '.join(other_participants)}"
                else: desc = "Had a conversation (details missing)"
                descriptions.append(desc)
        return descriptions if descriptions else ["no recent specific activities logged."]

    def start_travel_to(self, target_location_name: str) -> str:
        if not self.town_map: return f"{self.name} cannot travel: Town map is not available."
        if not self.current_grid_position:
            if self.current_location and self.current_location.grid_coordinate: self.current_grid_position = self.current_location.grid_coordinate
            else:
                res_coord_fallback = self.town_map.get_coordinates_for_location(self.residence)
                if res_coord_fallback:
                    self.current_grid_position = res_coord_fallback
                    if self.current_location and self.current_location.name != self.residence: print(f"Warning: Agent {self.name} current_grid_position was None, but current_location was {self.current_location.name}. Fell back to residence coord.")
                    elif not self.current_location:
                        res_loc_obj = self.locations.get(self.residence)
                        if res_loc_obj: self.current_location = res_loc_obj
                else: err_msg = f"{self.name} cannot travel: Current grid position is unknown and residence '{self.residence}' coord not found."; self.memory_manager.add_memory('SYSTEM_EVENT', {'time': self.current_time, 'content': err_msg}); return err_msg
        target_coord = self.town_map.get_coordinates_for_location(target_location_name)
        if not target_coord: return f"{self.name} cannot travel: Destination '{target_location_name}' not found in town map."
        if self.current_grid_position == target_coord:
            if not self.current_location or self.current_location.name != target_location_name:
                target_loc_obj = self.locations.get(target_location_name)
                if target_loc_obj:
                    if self.current_location: self.current_location.remove_agent(self)
                    target_loc_obj.add_agent(self)
                    self.current_location = target_loc_obj
            return f"{self.name} is already at {target_location_name}."
        path = self.town_map.find_path(self.current_grid_position, target_coord)
        if not path or len(path) <= 1: return f"{self.name} cannot find a path from {self.current_grid_position} to {target_location_name} ({target_coord})."
        self.is_traveling = True
        self.travel_destination_name = target_location_name
        self.travel_destination_coord = target_coord
        self.travel_path_coords = path
        self.current_path_step_index = 0 # Start at the first step of the path (which is current_grid_position)

        # Log the start of travel
        self.log_memory('ACTIVITY_EVENT', { 
            # 'time' will be added by log_memory if not present, using self.current_time
            'activity_type_tag': ACTIVITY_TYPES['TRAVEL'], 
            'description': f"Started traveling from {self.get_current_location_name()} (coord: {self.current_grid_position}) towards {target_location_name} (coord: {target_coord}). Path length: {len(path)-1} steps.", 
            'from_location': self.get_current_location_name(), 
            'to_location': target_location_name, 
            'from_coord': self.current_grid_position, 
            'to_coord': target_coord, 
            'path_steps': len(path)-1
        })
        
        return f"{self.name} is now traveling to {target_location_name}."

    def _perform_travel_step(self) -> Tuple[Optional[str], bool]:
        if not self.is_traveling or not self.travel_path_coords or not self.travel_destination_name:
            self.is_traveling = False
            return "Error: Travel not properly initialized.", False
        
        self.current_path_step_index += 1
        needs_mid_travel_decision = False
        log_message = None
        coord_before_this_step = self.current_grid_position # Store current position before moving

        if self.current_path_step_index < len(self.travel_path_coords):
            self.current_grid_position = self.travel_path_coords[self.current_path_step_index]
            self.energy_level = max(0, self.energy_level - self.energy_cost_per_step)
            log_message = f"Took step to {self.current_grid_position} towards {self.travel_destination_name}. Energy: {self.energy_level}."

            # Check for location stimulus
            location_name_at_step = self.town_map.get_location_name_at_coord(self.current_grid_position)
            if location_name_at_step and location_name_at_step != self.travel_destination_name: # Not the final destination
                loc_obj = self.locations.get(location_name_at_step)
                if loc_obj and loc_obj.is_open(self.current_time % 24):
                    self.mid_travel_stimulus = {
                        "type": "location_encounter", 
                        "location_name": location_name_at_step, 
                        "location_type": loc_obj.type, # For prompt
                        "is_open": True, # For prompt
                        "details": loc_obj, # For internal use if needed by decision logic
                        "coord_at_stimulus": self.current_grid_position,
                        "coord_before_stimulus": coord_before_this_step 
                    }
                    needs_mid_travel_decision = True
                    log_message += f" Passed by open {loc_obj.type} '{location_name_at_step}'."

            # Agent detection logic (inserted block)
            if not needs_mid_travel_decision and self.all_agents_list_for_perception and self.current_grid_position:
                encountered_agents_at_coord = []
                for other_agent in self.all_agents_list_for_perception:
                    # Ensure other_agent is not self and is at the same grid position
                    if other_agent.name != self.name and \
                       hasattr(other_agent, 'current_grid_position') and \
                       other_agent.current_grid_position == self.current_grid_position:
                        encountered_agents_at_coord.append(other_agent) # Add the full Agent object
                
                if encountered_agents_at_coord:
                    self.mid_travel_stimulus = {
                        "type": "agent_encounter", 
                        "encountered_agents": encountered_agents_at_coord, # Store list of Agent objects
                        "coord_at_stimulus": self.current_grid_position,
                        "coord_before_stimulus": coord_before_this_step 
                    }
                    needs_mid_travel_decision = True
                    agent_names_str = ", ".join([a.name for a in encountered_agents_at_coord])
                    
                    if log_message: # Append to existing log message if it exists
                        log_message += f" Encountered agent(s): {agent_names_str} at {self.current_grid_position}."
                    else: # Otherwise, create a new log message (should be rare if a step was taken)
                        log_message = f"Took step to {self.current_grid_position}. Encountered agent(s): {agent_names_str}."

            # Check for arrival at final destination
            if self.current_grid_position == self.travel_destination_coord:
                log_message = self._complete_travel() # This sets is_traveling to False
                # No decision needed if arrived, even if stimulus was also present at destination
                needs_mid_travel_decision = False 
        else: # Index out of bounds, means arrival
            log_message = self._complete_travel() # This sets is_traveling to False
            needs_mid_travel_decision = False

        return log_message, needs_mid_travel_decision

    def _complete_travel(self) -> str:
        self.is_traveling = False
        arrival_message = f"Arrived at {self.travel_destination_name} (coord: {self.current_grid_position})."
        target_loc_obj = self.locations.get(self.travel_destination_name)
        if target_loc_obj:
            if self not in target_loc_obj.agents:
                if target_loc_obj.add_agent(self):
                    self.current_location = target_loc_obj
                    arrival_message += f" Successfully entered {self.travel_destination_name}."
                else:
                    queue_msg = target_loc_obj.add_to_queue(self)
                    arrival_message += f" {self.travel_destination_name} is full. {queue_msg}"
                    self.log_memory('ACTIVITY_EVENT', {'activity_type_tag': ACTIVITY_TYPES['TRAVEL'], 'description': f"Arrived at {self.travel_destination_name} but it's full. {queue_msg}", 'location': self.travel_destination_name})
            else:
                self.current_location = target_loc_obj
                arrival_message += f" Confirmed presence at {self.travel_destination_name}."
        else:
            arrival_message += " This is a coordinate point, not a formal named location."
        
        self.log_memory('LOCATION_CHANGE_EVENT', {'from': "En route", 'to': self.travel_destination_name, 'from_coord': self.travel_path_coords[-2] if self.travel_path_coords and len(self.travel_path_coords) >1 else "Unknown", 'to_coord': self.current_grid_position, 'content': arrival_message, 'travel_type': 'arrival'})
        self.travel_path_coords = None
        self.current_path_step_index = 0
        return arrival_message

    def _calculate_time_to_next_commitment(self, current_time: int) -> float:
        """Calculate hours until next scheduled commitment."""
        if not self.daily_plan:
            return 24.0  # Default to plenty of time if no plan
            
        # Extract time mentions from daily plan using regex
        time_pattern = r'(\d{1,2})(?::00|:30| ?[AaPp][Mm])'
        matches = re.findall(time_pattern, self.daily_plan)
        
        if not matches:
            return 24.0
            
        future_times = []
        current_hour = current_time % 24
        
        for time_str in matches:
            try:
                hour = int(time_str)
                if "PM" in self.daily_plan[self.daily_plan.index(time_str):self.daily_plan.index(time_str)+10].upper():
                    hour += 12
                if hour > current_hour:
                    future_times.append(hour)
            except ValueError:
                continue
                
        if not future_times:
            return 24.0
            
        next_time = min(future_times)
        return next_time - current_hour

    def _assess_location_relevance(self, location: Location, context: Dict) -> str:
        """Assess how relevant a location is to the agent's current needs."""
        relevance_factors = []
        
        # Check energy and food needs
        if self.energy_level < 40 and location.type in ['local_shop', 'grocery']:
            relevance_factors.append("Could help with low energy")
            
        # Check grocery needs
        if self.grocery_level < 30 and location.type == 'grocery':
            relevance_factors.append("Could restock groceries")
            
        # Check if mentioned in daily plan
        if self.daily_plan and location.name in self.daily_plan:
            relevance_factors.append("Mentioned in daily plan")
            
        # Check if it's a frequent destination
        if location.name == self.workplace:
            relevance_factors.append("Your workplace")
            
        return "; ".join(relevance_factors) if relevance_factors else "No immediate relevance"

    def _get_last_visit_time(self, location_name: str) -> str:
        """Get the last time the agent visited this location."""
        recent_visits = self.memory_manager.retrieve_memories(
            self.name,
            self.current_time,
            memory_type_key='LOCATION_CHANGE_EVENT',
            limit=5
        )
        
        for visit in recent_visits:
            if visit['data'].get('to') == location_name:
                visit_time = visit['simulation_time']
                hours_ago = self.current_time - visit_time
                if hours_ago < 24:
                    return f"{hours_ago} hours ago"
                else:
                    days_ago = hours_ago // 24
                    return f"{days_ago} days ago"
        return "No recent visits recorded"

    def _get_location_offers(self, location: Location) -> str:
        """Get any current offers or discounts at the location."""
        if hasattr(location, 'discount') and location.discount:
            return f"{location.discount['value']}% discount available"
        return "No special offers"

    def _get_shared_history(self, agent_names: List[str]) -> str:
        """Get recent shared history with the encountered agents."""
        shared_events = []
        for name in agent_names:
            recent_interactions = self.memory_manager.retrieve_memories(
                self.name,
                self.current_time,
                memory_type_key='CONVERSATION_LOG_EVENT',
                limit=2
            )
            for interaction in recent_interactions:
                if name in interaction['data'].get('participants', []):
                    shared_events.append(f"Talked with {name}: {interaction['data'].get('content', 'details not recorded')}")
        return "; ".join(shared_events) if shared_events else "No recent shared history"

    def _calculate_destination_urgency(self, current_time: int) -> str:
        """Calculate how urgent it is to reach the destination."""
        if not self.travel_destination_name or not self.daily_plan:
            return "Low"
            
        # Check if destination is workplace during work hours
        if (self.travel_destination_name == self.workplace and 
            hasattr(self, 'work_schedule') and 
            self.work_schedule):
            work_start = self.work_schedule.get('start', 9)
            current_hour = current_time % 24
            if current_hour < work_start:
                time_until_work = work_start - current_hour
                if time_until_work <= 1:
                    return "Very High - Almost late for work"
                elif time_until_work <= 2:
                    return "High - Work starting soon"
                    
        # Check daily plan for urgent appointments
        time_to_next = self._calculate_time_to_next_commitment(current_time)
        if time_to_next <= 1:
            return "Very High - Upcoming commitment"
        elif time_to_next <= 2:
            return "High - Limited time"
        elif time_to_next <= 4:
            return "Medium - Some flexibility"
            
        return "Low - No immediate time pressure"

    def _get_agent_details_str(self, agents: List['Agent']) -> str:
        """Get detailed string about encountered agents."""
        details = []
        for agent in agents:
            detail = f"{agent.name} ("
            if agent.current_activity:
                detail += f"doing: {agent.current_activity}"
            if agent.is_traveling and agent.travel_destination_name:
                detail += f", heading to: {agent.travel_destination_name}"
            if agent.name in self.relationships:
                detail += f", {self.relationships[agent.name]}"
            detail += ")"
            details.append(detail)
        return "; ".join(details)