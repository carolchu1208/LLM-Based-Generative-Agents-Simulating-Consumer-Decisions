# Standard library imports
import os
import json
import random
import threading
import time
import traceback
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Union, 
    TYPE_CHECKING, TypeVar, Generic
)

# Third-party imports
import requests
from simulation_types import (
    MAX_CONVERSATION_TURNS, GROCERY_COST_HOME_MEAL, PromptManagerInterface
)
from simulation_constants import (
    ENERGY_MAX, ENERGY_MIN, ENERGY_DECAY_PER_HOUR, ENERGY_COST_WORK_HOUR,
    ENERGY_COST_PER_STEP, ENERGY_GAIN_RESTAURANT_MEAL,
    ENERGY_GAIN_SNACK, ENERGY_GAIN_HOME_MEAL,
    ENERGY_GAIN_NAP, ENERGY_THRESHOLD_LOW, ENERGY_THRESHOLD_FOOD
)
from memory_manager import MemoryManager
from metrics_manager import StabilityMetricsManager
if TYPE_CHECKING:
    from simulation_execution_classes import (
        ConversationManager, Agent, Location, TownMap,
        PlanExecutor, SimulationSettings
    )
from shared_trackers import (
    SharedLocationTracker, SharedResourceManager,
    LocationLockManager
)
from simulation_constants import (
    AgentError, LocationError,
    MemoryError, MetricsError, ThreadSafeBase
)

class PromptManager(PromptManagerInterface):
    _instance = None
    _unified_rules = {}  # Single source of truth for all rules
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, location_tracker=None, config_data=None):
        """Initialize the prompt manager."""
        if self._initialized:
            # If already initialized, just update the attributes
            if location_tracker is not None:
                self.location_tracker = location_tracker
            if config_data is not None:
                self.config_data = config_data
                self._load_menu_information()
            return
            
        self.location_tracker = location_tracker
        self.memory_mgr = None
        self.config_data = config_data or {}
        self.menu_data = {}
        # Cache for static dining info to avoid regenerating for each agent
        self._static_dining_cache = {}
        
        self._initialized = True
        self.prompt_templates = {}
        self._initialize_unified_rules()  # New unified system
        self._initialize_prompt_templates()
        self._load_menu_information()
        
    def _initialize_unified_rules(self):
        """Initialize unified rules system to prevent prompt conflicts."""
        
        # Constants are now imported at module level from simulation_constants and simulation_types
        # Energy constants come from simulation_constants (already imported)
        # Grocery and conversation constants come from simulation_types (already imported)
        from simulation_types import GROCERY_MAX, GROCERY_MIN, CONVERSATION_COOLDOWN_HOURS
        
        # ENERGY SYSTEM - Single definitive rules
        self._unified_rules['energy'] = {
            'range': f"Energy level range: {ENERGY_MIN}-{ENERGY_MAX} (starts at {ENERGY_MAX})",
            'natural_decay': f"Natural decay: -{ENERGY_DECAY_PER_HOUR} energy per hour automatically",
            'work_cost': f"Work cost: -{ENERGY_COST_WORK_HOUR} energy per hour worked",
            'travel_cost': f"Travel cost: -{ENERGY_COST_PER_STEP} energy per step moved (complete path in one hour)",
            'home_meal': f"Home meals: +{ENERGY_GAIN_HOME_MEAL} energy (requires grocery level > {GROCERY_COST_HOME_MEAL}, costs {GROCERY_COST_HOME_MEAL} grocery levels)",
            'restaurant_meal': f"Restaurant meals: +{ENERGY_GAIN_RESTAURANT_MEAL} energy",
            'snack': f"Snacks/beverages: +{ENERGY_GAIN_SNACK} energy",
            'sleep': f"Sleep at home (23:00-06:00): Energy set to {ENERGY_MAX} every hour (automatic)",
            'nap': f"Nap at workplace (11:00-15:00): +{ENERGY_GAIN_NAP} energy",
            'thresholds': {
                'low_energy': f"Energy < {ENERGY_THRESHOLD_LOW}: Plan urgent meals or rest",
                'food_threshold': f"Energy < {ENERGY_THRESHOLD_FOOD}: Must eat soon"
            }
        }
        
        # GROCERY SYSTEM - Single definitive rules with range specification
        # Get grocery store names dynamically from config
        grocery_stores = []
        if self.config_data:
            retail_grocery = self.config_data.get('town_areas', {}).get('retail_and_grocery', {})
            grocery_stores = list(retail_grocery.keys())
        grocery_locations_text = ', '.join(grocery_stores) if grocery_stores else "grocery stores"

        self._unified_rules['grocery'] = {
            'range': f"Grocery level range: {GROCERY_MIN}-{GROCERY_MAX} (starts at {GROCERY_MAX})",
            'home_cooking': f"Home meals require grocery level > {GROCERY_COST_HOME_MEAL} and cost {GROCERY_COST_HOME_MEAL} grocery levels",
            'shopping': "Grocery shopping: $1 per grocery level gained",
            'locations': f"Grocery shopping available at {grocery_locations_text}"
        }
        
        # ACTION SYSTEM - Single definitive rules
        self._unified_rules['actions'] = {
            'types': {
                'go_to': "Travel to any location",
                'shop': "Purchase items at stores", 
                'work': "Work-related activities at workplace",
                'eat': "Consume ANY food or drinks at ANY location (including home)",
                'rest': "For nap (11:00-15:00 at workplace) or sleep (23:00-06:00 at home)",
                'idle': "Relaxing/free time with no energy gain (use for evening relaxation, waiting, etc.)"
            },
            'rest_timing_rules': {
                'nap': f"rest action at workplace during 11:00-15:00 for +{ENERGY_GAIN_NAP} energy",
                'sleep': f"rest action at home during 23:00-06:00 for sleep (system sets energy to {ENERGY_MAX} automatically)",
                'other_times': "Use 'idle' action instead - no energy gain but valid activity"
            },
            'meal_planning': {
                'use_meal_types': "Plan with: breakfast, lunch, dinner, or snack",
                'no_item_names': "Do NOT specify exact menu items",
                'auto_selection': "System automatically selects appropriate items based on location and time",
                'examples': [
                    "eat [meal_type] at [restaurant name from valid locations]",
                    "eat [meal_type] at residence (home cooking)"
                ]
            }
        }
        
        # MEAL SYSTEM - Single definitive rules
        self._unified_rules['meals'] = {
            'timing': {
                'breakfast': "6:00-9:00",
                'lunch': "11:00-14:00", 
                'dinner': "17:00-20:00",
                'snack': "Any time"
            },
            'locations': {
                'home': f"Any eating/drinking at residence = home meal (+{ENERGY_GAIN_HOME_MEAL} energy, requires grocery level > {GROCERY_COST_HOME_MEAL}, costs {GROCERY_COST_HOME_MEAL} grocery levels)"
            },
            'critical_planning': f"Working 8 hours costs {ENERGY_COST_WORK_HOUR * 8} energy (work activity) + {ENERGY_DECAY_PER_HOUR * 8} energy (natural decay) = {(ENERGY_COST_WORK_HOUR + ENERGY_DECAY_PER_HOUR) * 8} total - plan meals accordingly"
        }
        
        # FORMAT SYSTEM - Single definitive rules
        self._unified_rules['format'] = {
            'requirements': [
                "Plan for EVERY HOUR from 07:00 to 23:00 (17 hours total)",
                "Each hour must have: time, action, target, description", 
                "Actions must be: go_to, shop, work, rest, eat",
                "Targets must be exact valid location names",
                "Use first person perspective",
                "No gaps or skipped hours allowed"
            ],
            'structure': "HH:00\nAction: [action]\nTarget: [location]\nReasoning: [brief explanation]"
        }

    def get_unified_rules(self, context_type: str, agent_state: dict = None) -> str:
        """Get unified rules based on context type and agent state."""
        rules_text = []
        
        if context_type == 'planning':
            # Core energy rules - always needed for planning
            energy_rules = self._unified_rules['energy']
            rules_text.append("🔋 ENERGY SYSTEM:")
            rules_text.append(f"• {energy_rules['range']}")
            rules_text.append(f"• {energy_rules['natural_decay']}")
            rules_text.append(f"• {energy_rules['work_cost']}")
            rules_text.append(f"• {energy_rules['travel_cost']}")
            rules_text.append(f"• {energy_rules['home_meal']}")
            rules_text.append(f"• {energy_rules['restaurant_meal']}")
            rules_text.append(f"• {energy_rules['snack']}")
            rules_text.append(f"• {energy_rules['sleep']}")
            rules_text.append(f"• {energy_rules['nap']}")
            
            # Grocery system rules
            grocery_rules = self._unified_rules['grocery']
            rules_text.append("\n🛒 GROCERY SYSTEM:")
            rules_text.append(f"• {grocery_rules['range']}")
            rules_text.append(f"• {grocery_rules['home_cooking']}")
            rules_text.append(f"• {grocery_rules['shopping']}")
            rules_text.append(f"• {grocery_rules['locations']}")
            
            # Add thresholds if energy is low
            if agent_state and agent_state.get('energy_level', ENERGY_MAX) < ENERGY_THRESHOLD_LOW:
                rules_text.append("\n⚠️ ENERGY THRESHOLDS:")
                for threshold_type, threshold_rule in energy_rules['thresholds'].items():
                    rules_text.append(f"• {threshold_rule}")
            
            # Meal system
            meal_rules = self._unified_rules['meals']
            rules_text.append("\n🕐 MEAL TIMING:")
            for meal_type, timing in meal_rules['timing'].items():
                rules_text.append(f"• {meal_type}: {timing}")
            
            rules_text.append("\n🏠 HOME MEALS:")
            rules_text.append(f"• {meal_rules['locations']['home']}")
            
            # Add dynamic dining information from config (actual locations only)
            dynamic_dining_info = self._get_dynamic_dining_info()
            if dynamic_dining_info and "Error" not in dynamic_dining_info:
                rules_text.append(f"\n{dynamic_dining_info}")
            
            rules_text.append(f"\n⚠️ {meal_rules['critical_planning']}")
            
            # Action system
            action_rules = self._unified_rules['actions']
            rules_text.append("\n🎯 ACTIONS:")
            for action_type, action_rule in action_rules['types'].items():
                rules_text.append(f"• {action_type}: {action_rule}")
            
            # Add specific rest timing rules
            rules_text.append("\n⏰ REST TIMING RULES:")
            for timing_type, timing_rule in action_rules['rest_timing_rules'].items():
                rules_text.append(f"• {timing_rule}")
            
            # Format system
            format_rules = self._unified_rules['format']
            rules_text.append("\n📋 FORMAT REQUIREMENTS:")
            for requirement in format_rules['requirements']:
                rules_text.append(f"• {requirement}")
            
            return '\n'.join(rules_text)

        elif context_type == 'conversation':
            # Include energy rules for travel decision-making
            energy_rules = self._unified_rules['energy']
            rules_text.append("🔋 ENERGY RULES:")
            rules_text.append(f"• {energy_rules['travel_cost']}")
            rules_text.append(f"• {energy_rules['natural_decay']}")
            rules_text.append(f"• Current energy: {agent_state.get('energy_level', 'Unknown') if agent_state else 'Unknown'}")

            rules_text.append(f"\n🗣️ CONVERSATION RULES:")
            rules_text.append(f" per turn (max {MAX_CONVERSATION_TURNS} turns)")
            rules_text.append(f"• Duration: 5-15 minutes per turn")
            rules_text.append(f"• Can occur at any location")
            rules_text.append(f"• Let your relationship naturally guide the conversation")
            
        return '\n'.join(rules_text)

    def _initialize_prompt_templates(self):
        """Initialize all prompt templates."""
        # Daily Plan Template
        self.prompt_templates["daily_plan"] = {
            "template": """You are {name}, a {age}-year-old {occupation} living in {residence}. You work at {workplace}.

Current State:
- Current Time: {current_time}:00
- Current Day: {current_day}
- Current Location: {current_location}
- Energy Level: {energy_level} (range: {ENERGY_MIN}-{ENERGY_MAX}, capped at {ENERGY_MAX})
- Grocery Level: {grocery_level} (range: 0-100)
- Available Money: ${money:.2f}

{system_info}

⚠️ CRITICAL LOCATION RULE:
You MUST ONLY use locations that exist in the simulation. Do NOT create, invent, or mention any new locations.
ONLY use the exact location names listed above in "Available Locations".
Example INVALID locations: "new bistro", "nearby cafe", "local restaurant", "downtown mall"
Example VALID: Use exact names from the "Available Locations" list above

Your Task:
Create a detailed 17-hour daily plan from {current_time}:00 to 23:00 (hours {current_time} through 23).

CRITICAL REQUIREMENTS:
- Plan EXACTLY 17 consecutive hours: {current_time}, {current_time}+1, {current_time}+2, ..., 23
- Each hour must have exactly one activity
- ONLY use locations from the "Available Locations" list above - NO made-up locations allowed
- Include appropriate meals: breakfast (6-9), lunch (11-14), dinner (17-20)
- Work during business hours (9-17) to earn your daily wage
- Manage your energy carefully (starts at {energy_level}, decays {ENERGY_DECAY_PER_HOUR}/hour)
- End at your residence at hour 23 (automatic sleep system handles hours 23-6)

Format your response as a JSON object with this exact structure:
{{
  "activities": [
    {{
      "time": {current_time},
      "action": "action_name",
      "target": "location_name",
      "description": "Brief description of what you're doing and why. For work hours, specify occupation-specific tasks."
    }},
    // ... exactly 17 activities total
    {{
      "time": 23,
      "action": "rest",
      "target": "{residence}",
      "description": "Preparing for sleep at home. Automatic sleep system will handle energy recovery through the night."
    }}
  ]
}}

⚠️ RESPONSE FORMAT: Return ONLY the JSON object, no markdown code blocks, no ```json or ``` markers. Start with {{ and end with }}.

⚠️ FINAL CHECK: Before submitting your plan, verify that EVERY "target" field contains a location name from the "Available Locations" list above. If you use any location not on that list, the plan will fail."""
        }

        # Emergency Replan Template
        self.prompt_templates["emergency_replan"] = {
            "template": """You are {name}.

Current Situation:
- Time: {current_time}:00 on Day {current_day}
- Your Location: {current_location}
- Your Money: ${money:.2f}
- Failure Reason: {reason}

Available Option:
- A nearby restaurant, '{target_location}', is open and you can afford a meal there.

Your Task:
Create a 2-hour plan starting at {current_time}:00 to travel to '{target_location}' and eat a meal.
Follow the exact JSON format rules provided for daily plans, but ONLY for the next two hours.

Format Rules:
- The plan must cover exactly two consecutive hours: {current_time}:00 and {next_hour}:00.
- Each hour must have an 'action' ('go_to' or 'eat') and a 'target'.
- The first hour's action must be 'go_to' with the target '{target_location}'.
- The second hour's action must be 'eat' with the target '{target_location}'.

⚠️ RESPONSE FORMAT: Return ONLY the JSON object, no markdown code blocks, no ```json or ``` markers. Start with {{ and end with }}.

Example Format:
{{
  "activities": [
    {{
      "time": {current_time},
      "action": "go_to",
      "target": "{target_location}",
      "description": "Traveling to the restaurant because I need to eat.",
    }},
    {{
      "time": {next_hour},
      "action": "eat",
      "target": "{target_location}",
      "description": "Eating a meal to regain energy.",
    }}
  ]
}}
"""
        }

    def _load_menu_information(self):
        """Load menu information from config - all data comes from agent_configuration.json."""
        try:
            # Load menu data from config's town_areas.dining section
            # Menu information is already in self.config_data
            self.menu_data = self.config_data.get('town_areas', {}).get('dining', {})
        except Exception as e:
            print(f"Error loading menu information: {str(e)}")
            self.menu_data = {}

    def _generate_static_dining_info(self, current_hour: int, current_day: int) -> Dict[str, Any]:
        """
        Generate static dining information that's the same for all agents.
        Focus on meal types (breakfast, lunch, dinner, snack) instead of specific item names.
        """
        try:
            # Check cache first
            cache_key = f"{current_day}_{current_hour}"
            if cache_key in self._static_dining_cache:
                return self._static_dining_cache[cache_key]
            
            if not self.menu_data:
                return {'error': "No dining information available."}
            
            # Get current meal period for context
            from simulation_types import get_meal_period
            meal_period = get_meal_period(current_hour)
            
            static_info = {
                'current_hour': current_hour,
                'current_day': current_day,
                'meal_period': meal_period,
                'locations': []
            }
            
            for location_name, location_data in self.menu_data.items():
                # Check if location is open
                hours = location_data.get('hours', {})
                open_hour = hours.get('open', 0)
                close_hour = hours.get('close', 24)
                
                if not (open_hour <= current_hour < close_hour):
                    continue
                
                # Get location info
                description = location_data.get('description', '')
                menu = location_data.get('menu', {})
                
                # Check for discounts
                discount_info = ""
                discount_data = None
                if location_name in self.config_data.get('town_areas', {}).get('dining', {}):
                    location_config = self.config_data['town_areas']['dining'][location_name]
                    discount_data = location_config.get('discount', {})
                    if discount_data and current_day in discount_data.get('days', []):
                        discount_value = discount_data.get('value', 0)
                        discount_type = discount_data.get('type', 'percentage')
                        if discount_type == 'percentage':
                            discount_info = f" 🎉 {discount_value}% OFF TODAY!"
                        else:
                            discount_info = f" 🎉 ${discount_value} OFF TODAY!"
                
                # Process menu to show available meal types with their time windows
                available_meal_types = []
                
                for meal_type, menu_item in menu.items():
                    available_hours = menu_item.get('available_hours', [])
                    base_price = menu_item.get('base_price', 0)
                    
                    # Calculate final price with discount
                    final_price = base_price
                    if discount_data and current_day in discount_data.get('days', []):
                        discount_value = discount_data.get('value', 0)
                        discount_type = discount_data.get('type', 'percentage')
                        if discount_type == 'percentage':
                            final_price = base_price * (1 - discount_value / 100)
                        else:
                            final_price = max(0, base_price - discount_value)
                    
                    # Format time windows for display
                    time_windows = []
                    if available_hours:
                        # Group consecutive hours
                        ranges = []
                        start = available_hours[0]
                        end = start
                        
                        for hour in available_hours[1:]:
                            if hour == end + 1:
                                end = hour
                            else:
                                ranges.append(f"{start}:00-{end+1}:00" if start != end else f"{start}:00")
                                start = end = hour
                        ranges.append(f"{start}:00-{end+1}:00" if start != end else f"{start}:00")
                        time_windows = ranges
                    
                    meal_type_info = {
                        'meal_type': meal_type,
                        'available_hours': available_hours,
                        'time_windows': time_windows,
                        'base_price': base_price,
                        'final_price': final_price,
                        'has_discount': bool(discount_info),
                        'available_now': current_hour in available_hours
                    }
                    available_meal_types.append(meal_type_info)
                
                if available_meal_types:
                    location_info = {
                        'name': location_name,
                        'description': description,
                        'discount_info': discount_info,
                        'open_hour': open_hour,
                        'close_hour': close_hour,
                        'available_meal_types': available_meal_types
                    }
                    static_info['locations'].append(location_info)
            
            # Cache the result
            self._static_dining_cache[cache_key] = static_info
            
            # Clean old cache entries (keep only last 3 hours to prevent memory buildup)
            if len(self._static_dining_cache) > 3:
                oldest_key = min(self._static_dining_cache.keys())
                del self._static_dining_cache[oldest_key]
            
            return static_info
            
        except Exception as e:
            print(f"Error generating static dining info: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': "Error retrieving dining information."}

    def get_dining_information_for_planning(self, current_hour: int, current_day: int) -> str:
        """Get dining information with current time context for planning."""
        try:
            dining_info = []
            
            # Time-sensitive dining information
            dining_info.append(f"Current Time: {current_hour}:00 on Day {current_day}")
            
            # Meal timing guidance
            if 6 <= current_hour <= 9:
                dining_info.append("🌅 Breakfast Time (6:00-9:00)")
            elif 11 <= current_hour <= 14:
                dining_info.append("🌞 Lunch Time (11:00-14:00)")
            elif 17 <= current_hour <= 20:
                dining_info.append("🌆 Dinner Time (17:00-20:00)")
            else:
                dining_info.append("🍪 Snack Time (meals outside normal hours)")
            
            return "\n".join(dining_info)
            
        except Exception as e:
            print(f"Error getting dining information for planning: {str(e)}")
            return "Error retrieving dining information."

    def _get_dynamic_dining_info(self) -> str:
        """Get dynamic dining location information from config."""
        try:
            if not self.config_data:
                return "No dining information available."
            
            dining_locations = self.config_data.get('town_areas', {}).get('dining', {})
            if not dining_locations:
                return "No dining locations found in config."
            
            dining_info = []
            dining_info.append("🍽️ AVAILABLE DINING LOCATIONS:")
            
            for location_name, location_data in dining_locations.items():
                location_type = location_data.get('type', 'unknown')
                description = location_data.get('description', '')
                menu = location_data.get('menu', {})
                
                if menu:
                    meal_info = []
                    for meal_type, meal_data in menu.items():
                        price = meal_data.get('base_price', 'Price not set')
                        available_hours = meal_data.get('available_hours', [])
                        if available_hours:
                            hours_str = f"{min(available_hours)}:00-{max(available_hours)}:00"
                        else:
                            hours_str = "Hours not set"
                        meal_info.append(f"{meal_type} (${price}, {hours_str})")
                    
                    dining_info.append(f"• {location_name}: {', '.join(meal_info)}")
                else:
                    dining_info.append(f"• {location_name}: No menu available")
            
            return '\n'.join(dining_info)
            
        except Exception as e:
            print(f"Error getting dynamic dining info: {str(e)}")
            return "Error retrieving dining information."

    # Interface implementation methods
    def get_prompt(self, prompt_type: str, context: dict) -> str:
        """Get a formatted prompt based on type and context."""
        try:
            if prompt_type == "daily_plan":
                return self.get_planning_prompt(
                    agent_name=context.get('name', ''),
                    current_time=context.get('current_time', 0),
                    context=context
                )
            elif prompt_type == "emergency_replan":
                return self.get_emergency_replan_prompt(context)
            elif prompt_type == "conversation":
                return self.get_conversation_prompt(
                    speaker=context.get('speaker', ''),
                    listener=context.get('listener', ''),
                    context=context
                )
            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
        except Exception as e:
            print(f"Error getting prompt: {str(e)}")
            traceback.print_exc()
            raise

    def get_prompt_type(self, prompt: str) -> str:
        """Determine the type of a given prompt."""
        try:
            if "Create a detailed 17-hour daily plan" in prompt:
                return "daily_plan"
            elif "Create a 2-hour plan starting" in prompt:
                return "emergency_replan"
            elif "conversation" in prompt.lower():
                return "conversation"
            else:
                return "unknown"
        except Exception:
            return "unknown"

    def get_discount_info(self, location_name: str, current_day: int) -> str:
        """Get discount information for a location."""
        # Currently no discount system implemented
        return ""

    def get_location_context(self, location_name: str, current_day: int) -> str:
        """Get context information for a location."""
        try:
            return self._get_location_specific_info(location_name)
        except Exception as e:
            print(f"Error getting location context: {str(e)}")
            return f"Location: {location_name}"

    def validate_context(self, prompt_type: str, context: dict) -> bool:
        """Validate context for a prompt type."""
        try:
            if prompt_type == "daily_plan":
                required_fields = ['name', 'age', 'occupation', 'residence', 'workplace', 
                                 'current_time', 'current_day', 'current_location', 
                                 'energy_level', 'grocery_level', 'money']
                return all(field in context for field in required_fields)
            elif prompt_type == "emergency_replan":
                required_fields = ['name', 'current_time', 'current_day', 'current_location', 
                                 'money', 'reason', 'target_location']
                return all(field in context for field in required_fields)
            elif prompt_type == "conversation":
                required_fields = ['speaker', 'listener']
                return all(field in context for field in required_fields)
            else:
                return False
        except Exception:
            return False

    def get_planning_prompt(self, agent_name: str, current_time: int, context: dict) -> str:
        """Get a planning prompt for an agent."""
        try:
            template = self.prompt_templates.get("daily_plan")
            if not template:
                raise ValueError("Daily plan template not found")
            
            # Get unified rules for planning context (no energy warnings during planning)
            unified_rules = self.get_unified_rules('planning', context)
            
            # Add system info with unified rules and pass context for location information
            system_info = self._get_system_info(context).format(unified_rules=unified_rules)
            
            context['system_info'] = system_info
            
            # Ensure agent_name and current_time are in context
            context['name'] = agent_name
            context['current_time'] = current_time
            
            # Validate required context fields
            if not self.validate_context('daily_plan', context):
                raise ValueError("Missing required context fields for daily plan")

            return template["template"].format(**context)
            
        except Exception as e:
            print(f"Error getting planning prompt: {str(e)}")
            traceback.print_exc()
            raise

    def get_plan_schema(self) -> dict:
        """Get the schema for planning prompts."""
        return {
            "type": "object",
            "properties": {
                "activities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "time": {"type": "integer", "minimum": 7, "maximum": 23},
                            "action": {"type": "string"},
                            "target": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["time", "action", "target", "description"]
                    },
                    "minItems": 17,
                    "maxItems": 17
                }
            },
            "required": ["activities"]
        }

    def get_conversation_prompt(self, speaker: str, listener: str, context: dict) -> str:
        """Get a conversation prompt between two agents."""
        try:
            # Build conversation history context
            previous_interaction = context.get('previous_interaction', [])
            conversation_turn = context.get('conversation_turn', 0)
            
            # Determine if this is an initiator or response turn
            is_initiator_turn = len(previous_interaction) == 0
            
            if is_initiator_turn:
                # First turn - speaker is initiating the conversation                
                prompt = f"""You are {speaker} starting a conversation with {listener}.

Current context:
- Location: {context.get('location', 'Unknown')}
- Time: {context.get('time', 'Unknown')}:00
- Your energy: {context.get('energy_level', 'Unknown')}
- Your money: ${context.get('money', 'Unknown')}
- Relationship: {context.get('relationship', 'neighbor')}
- Location Info: {context.get('location_hours_info', 'No hours info available')}

Your Background:
- Occupation: {context.get('speaker_occupation', 'Unknown')}
- Employment: {context.get('typical_work_hours', 'Unknown schedule')}
- Workplace: {context.get('speaker_workplace', 'Unknown')}
- Current Activity: {context.get('current_activity', 'Unknown')}

🔋 ENERGY RULES FOR PLANNING:
- Travel costs {ENERGY_COST_PER_STEP} energy per step moved (complete path in one hour)
- Natural decay: -{ENERGY_DECAY_PER_HOUR} energy per hour automatically
- Consider travel costs when suggesting meeting places

⚠️ VALID LOCATIONS ONLY:
If you make plans to meet somewhere, you MUST use locations from this list:
{', '.join(context.get('valid_locations', []))}
Do NOT invent new locations like "new bistro", "nearby cafe", "local place" - use exact names from the list above.

You are initiating a conversation with {listener}. Generate a natural greeting or opening statement.
- Keep it brief (1-2 sentences) and appropriate for the setting
- Consider your relationship with {listener}
- Stay in character as {speaker}
- Be aware of current time and location availability when discussing plans
- IMPORTANT: You are a working adult - do NOT mention classes, school, or student activities
- Base conversations on your actual occupation and work schedule
- If suggesting meeting places, ONLY use valid location names from the list above

If you make any specific plans or commitments with {listener} (like meeting somewhere at a specific time), format your response as JSON:
{{
  "dialogue": "Your conversation text here",
  "commitments": [
    {{
      "with": ["{listener}"],
      "schedule_update": {{
        "18:00": {{"action": "go_to", "target": "Location Name", "reasoning": "Brief reason"}},
        "19:00": {{"action": "eat", "target": "Location Name", "reasoning": "Brief reason"}}
      }}
    }}
  ]
}}

⚠️ COMMITMENT VALIDATION: If you include commitments, the "target" MUST be an exact location name from the valid locations list above.

Otherwise, just respond with your dialogue text directly.

Your opening statement:"""
                
            else:
                # Response turn - speaker is responding to previous dialogue
                last_turn = previous_interaction[-1] if previous_interaction else {}
                last_speaker = last_turn.get('speaker', 'Unknown')
                last_dialogue = last_turn.get('dialogue', '')
                                
                # Build conversation history for context
                history_text = "\n\nConversation so far:\n"
                for turn in previous_interaction:
                    turn_speaker = turn.get('speaker', 'Unknown')
                    turn_dialogue = turn.get('dialogue', '')
                    history_text += f"- {turn_speaker}: \"{turn_dialogue}\"\n"
                
                prompt = f"""You are {speaker} having a conversation with {listener}.

Current context:
- Location: {context.get('location', 'Unknown')}
- Time: {context.get('time', 'Unknown')}:00
- Your energy: {context.get('energy_level', 'Unknown')}
- Your money: ${context.get('money', 'Unknown')}
- Relationship: {context.get('relationship', 'neighbor')}
- Location Info: {context.get('location_hours_info', 'No hours info available')}

Your Background:
- Occupation: {context.get('speaker_occupation', 'Unknown')}
- Employment: {context.get('typical_work_hours', 'Unknown schedule')}
- Workplace: {context.get('speaker_workplace', 'Unknown')}
- Current Activity: {context.get('current_activity', 'Unknown')}

🔋 ENERGY RULES FOR PLANNING:
- Travel costs {ENERGY_COST_PER_STEP} energy per step moved (complete path in one hour)
- Natural decay: -{ENERGY_DECAY_PER_HOUR} energy per hour automatically
- Consider travel costs when suggesting meeting places

⚠️ VALID LOCATIONS ONLY:
If you make plans to meet somewhere, you MUST use locations from this list:
{', '.join(context.get('valid_locations', []))}
Do NOT invent new locations like "new bistro", "nearby cafe", "local place" - use exact names from the list above.

{history_text}

{last_speaker} just said: "{last_dialogue}"

Generate a natural response to what {last_speaker} just said.
- Keep it brief (1-2 sentences) and appropriate for the setting
- Respond directly to what was just said
- Stay in character as {speaker}
- Be aware of current time and location availability when discussing plans
- IMPORTANT: You are a working adult - do NOT mention classes, school, or student activities
- Base conversations on your actual occupation and work schedule
- If suggesting meeting places, ONLY use valid location names from the list above

If you make any specific plans or commitments with {listener} (like agreeing to meet somewhere at a specific time), format your response as JSON:
{{
  "dialogue": "Your conversation text here",
  "commitments": [
    {{
      "with": ["{listener}"],
      "schedule_update": {{
        "18:00": {{"action": "go_to", "target": "Location Name", "reasoning": "Brief reason"}},
        "19:00": {{"action": "eat", "target": "Location Name", "reasoning": "Brief reason"}}
      }}
    }}
  ]
}}

⚠️ COMMITMENT VALIDATION: If you include commitments, the "target" MUST be an exact location name from the valid locations list above.

Otherwise, just respond with your dialogue text directly.

Your response:"""
            
            return prompt
            
        except Exception as e:
            print(f"Error getting conversation prompt: {str(e)}")
            traceback.print_exc()
            raise

    def get_emergency_replan_prompt(self, context: dict) -> str:
        """Generate emergency replan prompt when an agent needs immediate food."""
        try:
            template = self.prompt_templates.get("emergency_replan")
            if not template:
                raise ValueError("Emergency replan template not found")
            
            # Validate required context fields
            if not self.validate_context('emergency_replan', context):
                raise ValueError("Missing required context fields for emergency replan")
            
            # Add the next hour to the context
            context['next_hour'] = (context['current_time'] + 1) % 24

            return template["template"].format(**context)
            
        except Exception as e:
            print(f"Error getting emergency replan prompt: {str(e)}")
            traceback.print_exc()
            raise

    def _get_location_specific_info(self, location_name: str) -> str:
        """Get location-specific rules and information."""
        context = ""
        
        if not self.config_data:
            return f"Location: {location_name}"
        
        # Check what type of location this is from config
        town_areas = self.config_data.get('town_areas', {})
        
        # Check if it's a dining location
        dining_locations = town_areas.get('dining', {})
        if location_name in dining_locations:
            location_data = dining_locations[location_name]
            menu = location_data.get('menu', {})
            if menu:
                meal_types = list(menu.keys())
                context += f"\nRestaurant/Cafe Information:\n"
                context += f"- Serves: {', '.join(meal_types)}\n"
                context += f"- Description: {location_data.get('description', '')}\n"
                
                # Add pricing information from config
                for meal_type, meal_data in menu.items():
                    price = meal_data.get('base_price', 'Price not set')
                    available_hours = meal_data.get('available_hours', [])
                    if available_hours:
                        hours_str = f"{min(available_hours)}:00-{max(available_hours)}:00"
                        context += f"- {meal_type}: ${price} (available {hours_str})\n"
                    else:
                        context += f"- {meal_type}: ${price} (hours not set)\n"
            else:
                context += f"\nDining Location:\n"
                context += f"- No menu available\n"
        
        # Check if it's a grocery location
        grocery_locations = town_areas.get('retail_and_grocery', {})
        if location_name in grocery_locations:
            location_data = grocery_locations[location_name]
            base_price = location_data.get('base_price', 'Price not set')
            context += f"\nGrocery Store Information:\n"
            context += f"- PRIMARY PURPOSE: Buy groceries at ${base_price} per level\n"
            context += f"- Description: {location_data.get('description', '')}\n"
            context += f"- ⚠️ CANNOT eat meals here - this is for shopping only\n"
            context += f"- For meals, go to restaurants in the dining section\n"
        
        # Check if it's a residence
        residence_locations = town_areas.get('residences', {})
        if location_name in residence_locations:
            context += f"\nHome Information:\n"
            context += f"- Can eat home meals if grocery level > {GROCERY_COST_HOME_MEAL}\n"
            context += f"- Home meals cost {GROCERY_COST_HOME_MEAL} grocery levels, provide +{ENERGY_GAIN_HOME_MEAL} energy\n"
            context += f"- Sleep here at night to reset energy to {ENERGY_MAX}\n"
        
        # Check if it's a work location
        work_locations = town_areas.get('work', {})
        if location_name in work_locations:
            context += f"\nWorkplace Information:\n"
            context += f"- Work here to earn daily wage\n"
            context += f"- Can take naps during midday hours (11-15)\n"
            
        return context

    def _get_system_info(self, context: dict = None):
        """Get system information for all prompts."""
        # Build available locations from context if provided
        locations_text = "Available Locations:\n"
        if context and 'valid_locations' in context:
            # Categorize locations dynamically from config
            dining_locations = set()
            grocery_locations = set()
            residence_locations = set()
            work_locations = set()
            
            if self.config_data:
                # Get dining locations from config
                dining_data = self.config_data.get('town_areas', {}).get('dining', {})
                dining_locations = set(dining_data.keys())
                
                # Get grocery locations from config
                grocery_data = self.config_data.get('town_areas', {}).get('retail_and_grocery', {})
                grocery_locations = set(grocery_data.keys())
                
                # Get residence locations from config
                residence_data = self.config_data.get('town_areas', {}).get('residences', {})
                residence_locations = set(residence_data.keys())
                
                # Get work locations from config
                work_data = self.config_data.get('town_areas', {}).get('work', {})
                work_locations = set(work_data.keys())
            
            for location in context['valid_locations']:
                # Categorize based on config data
                if location in dining_locations:
                    locations_text += f"- {location} (RESTAURANT/CAFE - for meals and snacks)\n"
                elif location in grocery_locations:
                    locations_text += f"- {location} (GROCERY STORE - for shopping only, NO meals)\n"
                elif location in residence_locations:
                    locations_text += f"- {location} (RESIDENCE - for home meals and sleep)\n"
                elif location in work_locations:
                    locations_text += f"- {location} (WORKPLACE - for work and naps)\n"
                else:
                    locations_text += f"- {location}\n"
        else:
            # Fallback - should not happen if context is properly provided
            locations_text += "- Residence (your home)\n"
            locations_text += "- [Locations should be provided dynamically from configuration]\n"
        
        return f"""IMPORTANT SYSTEM RULES - READ CAREFULLY:

{locations_text}
{{unified_rules}}

Available Actions:
- go_to: Travel to a location (costs 1 energy per step + natural decay)
- work: Work at office (costs {ENERGY_COST_WORK_HOUR} energy/hour + natural decay, earns daily wage)
- eat: Eat at current location (ONLY works at restaurants/cafes with menus, or at home with groceries)
- shop: Buy groceries or snacks
- rest: Rest/nap/sleep at current location
  * Sleep at home (22:00-06:00): resets energy to {ENERGY_MAX}
  * Nap at workplace (11:00-15:00): +{ENERGY_GAIN_NAP} energy
  * Invalid at other times/locations
- idle: Relaxing/free time with no energy gain (for general relaxation, waiting, etc.)
- converse: Talk with another agent (if at same location, respects cooldown)

⚠️ CRITICAL ACTION RULES:
1. Use "eat" action ONLY for actual meals - never for napping!
2. Use "rest" action for napping (workplace 11-15) or sleep (home 23-6)
3. Use "idle" action for relaxing/waiting at other times (no energy gain)
4. "eat" action requires being at a restaurant with a menu OR at home with groceries
5. Don't confuse eating, resting, and idle activities!

Travel Times:
- Most trips take 1 step (1 energy + natural decay)
- Plan travel time into your schedule

Financial System:
- Restaurant meals cost varies by location and meal type (see dining locations above)
- Snacks/beverages cost varies by location and item type
- Groceries cost $1 per level purchased
- You earn a daily wage by working

CRITICAL PLANNING RULES:
1. Plan ALL 17 hours (7:00-23:00) - no gaps allowed!
2. Include meals during appropriate times: breakfast (6-9), lunch (11-14), dinner (17-20)
3. ⚠️ NEVER skip lunch! Working 8 hours costs {ENERGY_COST_WORK_HOUR * 8} energy - you MUST eat lunch to survive
4. Monitor energy - you lose {ENERGY_DECAY_PER_HOUR} energy per hour automatically + {ENERGY_COST_WORK_HOUR} per hour working
5. Work standard hours (9:00-17:00) to earn money
   ⚠️ IMPORTANT: Describe work activities based on your specific occupation
   - Include specific tasks typical for your role (e.g., chef: preparing meals, managing kitchen; software engineer: coding, meetings; retail: serving customers, stocking)
   - Vary descriptions throughout the day to reflect different work tasks
   - Make work descriptions realistic and occupation-specific
6. Use 'rest' action for napping (workplace 11-15) or sleep (home 23-6)
7. Use 'idle' action for evening relaxation, waiting, general free time (no energy gain)
8. Each hour must have exactly one action with a target location
9. ⚠️ CRITICAL: Only eat at RESTAURANTS/CAFES (with menus) or at HOME - NOT at grocery stores or workplaces!
10. Examples: "action": "idle" at home during hours 20-22 for evening relaxation before sleep
"""
