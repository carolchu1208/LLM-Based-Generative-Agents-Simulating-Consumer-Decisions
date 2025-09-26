import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Any, TYPE_CHECKING, Optional
import traceback
from collections import defaultdict
import time
import copy

from simulation_types import TimeManager, METRICS_SAVE_INTERVAL

if TYPE_CHECKING:
    from simulation_execution_classes import Agent

class StabilityMetricsManager:
    def __init__(self, memory_manager, agents: Dict[str, 'Agent'] = None):
        self.memory_manager = memory_manager
        self.agents = agents or {}  # Store agents dictionary
        
        # Create metrics directory in the workspace root
        main_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'LLMAgentsTown_memory_records')
        metrics_dir = os.path.join(main_dir, 'simulation_metrics')
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            
        # Use simulation timestamp for file naming
        self.simulation_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(metrics_dir, f'metrics_log_{self.simulation_start_time}.jsonl')
        print(f"Metrics log file will be saved to: {self.metrics_file}")
        
        # Set up daily summaries directory and file path
        self.daily_summaries_dir = os.path.join(main_dir, 'simulation_daily_summaries')
        if not os.path.exists(self.daily_summaries_dir):
            os.makedirs(self.daily_summaries_dir)
        self.daily_summaries_file = os.path.join(self.daily_summaries_dir, f'daily_summary_{self.simulation_start_time}.json')
        print(f"Daily summaries will be saved to: {self.daily_summaries_file}")
        
        # Initialize time from TimeManager
        self.current_day = TimeManager.get_current_day()
        self.current_time = TimeManager.get_current_hour()
        
        # This dictionary holds the live, in-memory aggregated state. It is not saved directly.
        self.metrics_state = {
            'shop_metrics': {},
            'customer_metrics': {},
        }
        
        # Buffer for pending metric events to be written to the log file.
        self._pending_metrics = []
        
        # Initialize metrics dictionaries
        self.metrics = {
            'simulation_info': {
                'current_day': 1
            },
            'shop_metrics': {},  # Will store metrics for each shop
            'customer_metrics': {},  # Will store metrics for each customer
            'purchase_records': [],  # Will store detailed purchase records
            'location_visits': {},  # Will store location visit records
            'energy_levels': {},  # Will store energy levels
            'grocery_levels': {},  # Will store grocery levels
            'financial_states': {},  # Will store financial states
            'activities': {},  # Will store activities
        }
        
        # Initialize templates
        self._initialize_templates()
        
        self.daily_metrics_data = {}
        self.start_time = datetime.now()
        
        # Initialize first day
        self.ensure_day_initialized(1)
        print("Metrics system initialized with timestamp:", self.simulation_start_time)

        self.agent_metrics = defaultdict(lambda: defaultdict(list))  # agent_name -> metric_type -> list of metrics
        self.location_metrics = defaultdict(lambda: defaultdict(list))  # location_name -> metric_type -> list of metrics
        self.simulation_metrics = defaultdict(list)  # metric_type -> list of metrics

    def _initialize_templates(self):
        """Initialize metric templates."""
        self.shop_metrics_template = {
            'total_revenue': 0.0,
            'total_transactions': 0,
            'customer_count': 0,
            'average_transaction_value': 0.0,
            'popular_items': {},
            'peak_hours': {},
            'daily_sales': {},
            'discount_usage': 0,
            'inventory_levels': {}
        }
        
        self.customer_metrics_template = {
            'total_spent': 0.0,
            'visit_count': 0,
            'average_spend': 0.0,
            'favorite_items': {},
            'visit_times': {},
            'discount_usage': 0,
            'purchase_frequency': 0.0
        }
        
        # Initialize purchase record template
        self.purchase_record_template = {
            'simulation_day': None,
            'current_time': None,
            'shop_name': None,
            'customer_name': None,
            'purchase_item': None,
            'purchase_quantity': 0,
            'base_price': 0.0,
            'discount': 0.0,
            'final_price': 0.0,
            'energy_level': {
                'before_meal': 0.0,
                'after_meal': 0.0
            },
            'financial_state': {
                'before_meal': 0.0,
                'after_meal': 0.0
            }
        }
        
    def initialize_metrics(self) -> None:
        """Initialize all metrics structures."""
        self.daily_metrics_data = {}
        self.initialize_day_metrics(1)

    def initialize_day_metrics(self, day: int) -> None:
        """Initialize metrics for a new day."""
        if day not in self.daily_metrics_data:
            self.daily_metrics_data[day] = {
                'energy': {},
                'travel': {},
                'grocery': {},
                'financial': {},
                'activity': {},
                'location_visits': {},
                'relationship_trends': {}
            }

    def ensure_day_initialized(self, day: int) -> None:
        """Ensure metrics are initialized for a specific day."""
        if day not in self.daily_metrics_data:
            self.initialize_day_metrics(day)

    def _record_metric(self, metric_type: str, data: Dict[str, Any], day: Optional[int] = None) -> None:
        """Generic method to record a metric with proper locking and saving."""
        try:
            # Add the event to the pending metrics buffer
            event = {
                'event_type': metric_type,
                'day': day or TimeManager.get_current_day(),
                'hour': TimeManager.get_current_hour(),
                'data': data
            }
            self._pending_metrics.append(event)
            self.save_metrics()
        except Exception as e:
            print(f"Error recording {metric_type} metric: {str(e)}")
            traceback.print_exc()
            
    def record_agent_metric(self, agent_name: str, metric_type: str, value: Any, timestamp: Optional[Dict[str, int]] = None) -> None:
        """Record a metric for a specific agent."""
        try:
            if agent_name not in self.agent_metrics:
                self.agent_metrics[agent_name] = defaultdict(list)
            self.agent_metrics[agent_name][metric_type].append({
                'value': value,
                'timestamp': timestamp or {'day': TimeManager.get_current_day(), 'time': TimeManager.get_current_hour()}
            })
            self.save_metrics(force=True)
        except Exception as e:
            print(f"Error recording agent metric: {str(e)}")
            traceback.print_exc()
        
    def record_location_metric(self, location_name: str, metric_type: str, value: Any, timestamp: Optional[Dict[str, int]] = None) -> None:
        """Record a metric for a specific location."""
        try:
            if location_name not in self.location_metrics:
                self.location_metrics[location_name] = defaultdict(list)
            self.location_metrics[location_name][metric_type].append({
                'value': value,
                'timestamp': timestamp or {'day': TimeManager.get_current_day(), 'time': TimeManager.get_current_hour()}
            })
            self.save_metrics(force=True)
        except Exception as e:
            print(f"Error recording location metric: {str(e)}")
            traceback.print_exc()

    def record_simulation_metric(self, metric_type: str, value: Any, timestamp: Optional[Dict[str, int]] = None) -> None:
        """Record a simulation-wide metric."""
        try:
            self.simulation_metrics[metric_type].append({
                'value': value,
                'timestamp': timestamp or {'day': TimeManager.get_current_day(), 'time': TimeManager.get_current_hour()}
            })
            self.save_metrics(force=True)
        except Exception as e:
            print(f"Error recording simulation metric: {str(e)}")
            traceback.print_exc()

    def record_energy_level(self, agent_name: str, level: str, day: Optional[int] = None) -> None:
        """Record energy level for an agent."""
        self._record_metric('energy_levels', {agent_name: level}, day)

    def record_travel_time(self, agent_name: str, time: float, day: Optional[int] = None) -> None:
        """Record travel time for an agent."""
        self._record_metric('financial_states', {agent_name: {'travel_time': time}}, day)

    def record_grocery_level(self, agent_name: str, level: str, day: Optional[int] = None) -> None:
        """Record grocery level for an agent."""
        self._record_metric('grocery_levels', {agent_name: level}, day)

    def record_financial_state(self, agent_name: str, state: str, day: Optional[int] = None) -> None:
        """Record financial state for an agent."""
        self._record_metric('financial_states', {agent_name: {'state': state}}, day)

    def record_activity_type(self, agent_name: str, activity: str, day: Optional[int] = None) -> None:
        """Record activity type for an agent."""
        self._record_metric('activities', {agent_name: activity}, day)

    def record_location_visit(self, agent_name: str, location: str, day: int, hour: int) -> None:
        """Record a location visit for an agent."""
        try:
            event_data = {
                'agent': agent_name,
                'location': location,
                'day': day,
                'hour': hour
            }
            self._pending_metrics.append({'event_type': 'location_visit', 'data': event_data})
            self.save_metrics()
        except Exception as e:
            print(f"Error recording location visit: {str(e)}")
            traceback.print_exc()

    def record_sale(self, location_name: str, agent_name: str, amount: float, items: List[Dict], has_discount: bool, simulation_day: int, simulation_hour: int) -> None:
        """Record a sale at a location using simplified meal type system."""
        try:
            # Update in-memory aggregated state
            if location_name not in self.metrics_state['shop_metrics']:
                self.metrics_state['shop_metrics'][location_name] = copy.deepcopy(self.shop_metrics_template)
            
            # Update shop metrics
            shop_metrics = self.metrics_state['shop_metrics'][location_name]
            shop_metrics['total_revenue'] += amount
            shop_metrics['total_transactions'] += 1
            shop_metrics['customer_count'] += 1 # This might overcount unique customers
            shop_metrics['average_transaction_value'] = shop_metrics['total_revenue'] / shop_metrics['total_transactions']
            if has_discount:
                shop_metrics['discount_usage'] += 1
            
            # Update popular meal types (simplified approach)
            for item in items:
                meal_type = item.get('meal_type', 'unknown')
                shop_metrics['popular_items'][meal_type] = shop_metrics['popular_items'].get(meal_type, 0) + 1
            
            # Update peak hours
            shop_metrics['peak_hours'][simulation_hour] = shop_metrics['peak_hours'].get(simulation_hour, 0) + 1
            
            # Update daily sales
            shop_metrics['daily_sales'][simulation_day] = shop_metrics['daily_sales'].get(simulation_day, 0) + amount
            
            # Update customer metrics
            if agent_name not in self.metrics_state['customer_metrics']:
                self.metrics_state['customer_metrics'][agent_name] = copy.deepcopy(self.customer_metrics_template)
            
            customer_metrics = self.metrics_state['customer_metrics'][agent_name]
            customer_metrics['total_spent'] += amount
            customer_metrics['visit_count'] += 1
            customer_metrics['average_spend'] = customer_metrics['total_spent'] / customer_metrics['visit_count']

            # Update customer favorite meal types (simplified approach)
            for item in items:
                meal_type = item.get('meal_type', 'unknown')
                customer_metrics['favorite_items'][meal_type] = customer_metrics['favorite_items'].get(meal_type, 0) + 1
            
            # Update customer visit times
            customer_metrics['visit_times'][simulation_hour] = customer_metrics['visit_times'].get(simulation_hour, 0) + 1
            
            # Create the raw event log for this sale with simplified data
            sale_event = {
                'shop_name': location_name,
                'customer_name': agent_name,
                'amount': amount,
                'meal_types': [item.get('meal_type', 'unknown') for item in items],  # Track meal types, not complex item names
                'base_prices': [item.get('base_price', 0) for item in items],
                'final_prices': [item.get('final_price', item.get('base_price', 0)) for item in items],
                'has_discount': has_discount,
                'day': simulation_day,
                'hour': simulation_hour
            }
            self._pending_metrics.append({'event_type': 'sale', 'data': sale_event})
            self.save_metrics()
            
        except Exception as e:
            print(f"Error recording sale: {str(e)}")
            traceback.print_exc()

    def record_purchase(self, shop_name: str, customer_name: str, purchase_data: Dict[str, Any]) -> None:
        """Record a purchase with detailed information."""
        try:
            # Update in-memory aggregated state
            if shop_name not in self.metrics_state['shop_metrics']:
                self.metrics_state['shop_metrics'][shop_name] = copy.deepcopy(self.shop_metrics_template)
            if customer_name not in self.metrics_state['customer_metrics']:
                self.metrics_state['customer_metrics'][customer_name] = copy.deepcopy(self.customer_metrics_template)
            
            # Update metrics
            self.metrics_state['purchase_records'].append({
                'simulation_day': TimeManager.get_current_day(),
                'current_time': TimeManager.get_current_hour(),
                'shop_name': shop_name,
                'customer_name': customer_name,
                **purchase_data
            })
            self.metrics_state['shop_metrics'][shop_name]['total_revenue'] += purchase_data.get('final_price', 0.0)
            self.metrics_state['shop_metrics'][shop_name]['total_transactions'] += 1
            self.metrics_state['shop_metrics'][shop_name]['customer_count'] += 1
            self.metrics_state['shop_metrics'][shop_name]['average_transaction_value'] = (
                self.metrics_state['shop_metrics'][shop_name]['total_revenue'] / self.metrics_state['shop_metrics'][shop_name]['total_transactions']
            )
            
            # Update customer metrics
            self.metrics_state['customer_metrics'][customer_name]['total_spent'] += purchase_data.get('final_price', 0.0)
            self.metrics_state['customer_metrics'][customer_name]['visit_count'] += 1
            self.metrics_state['customer_metrics'][customer_name]['average_spend'] = (
                self.metrics_state['customer_metrics'][customer_name]['total_spent'] / self.metrics_state['customer_metrics'][customer_name]['visit_count']
            )
            
            # Create the raw event log for this purchase
            purchase_event = {
                'shop_name': shop_name,
                'customer_name': customer_name,
                **purchase_data
            }
            self._pending_metrics.append({'event_type': 'purchase', 'data': purchase_event})
            self.save_metrics()
            
        except Exception as e:
            print(f"Error recording purchase: {str(e)}")
            traceback.print_exc()

    def record_visit(self, agent_name: str, location: str) -> None:
        """Record a visit to a location."""
        try:
            event_data = {
                'agent': agent_name,
                'location': location,
                'day': TimeManager.get_current_day(),
                'hour': TimeManager.get_current_hour()
            }
            self._pending_metrics.append({'event_type': 'visit', 'data': event_data})
            self.save_metrics()
        except Exception as e:
            print(f"Error recording visit: {str(e)}")
            traceback.print_exc()

    def record_plan_creation(self, agent_name: str, plan_type: str, target_location: str) -> None:
        """Record plan creation for an agent."""
        # This is primarily a memory event. Metrics can be derived from other logs if needed.
        pass

    def update_simulation_time(self, day: int, time: int) -> None:
        """Update the current simulation time."""
        self.current_day = day
        self.current_time = time
        self.metrics['simulation_info']['current_day'] = self.current_day

    def clear_daily_metrics(self):
        """Clear metrics data for the current day after saving."""
        try:
            self.save_metrics(force=True)
            
            # Clear daily metrics
            if self.current_day in self.daily_metrics_data:
                del self.daily_metrics_data[self.current_day]
            
            # Clear location visits
            for location in self.metrics['location_visits']:
                self.metrics['location_visits'][location] = [
                    visit for visit in self.metrics['location_visits'][location]
                    if visit['day'] != self.current_day
                ]
            
            # Clear energy levels
            for agent in self.metrics['energy_levels']:
                if self.current_day in self.metrics['energy_levels'][agent]:
                    del self.metrics['energy_levels'][agent][self.current_day]
            
            # Clear grocery levels
            for agent in self.metrics['grocery_levels']:
                if self.current_day in self.metrics['grocery_levels'][agent]:
                    del self.metrics['grocery_levels'][agent][self.current_day]
            
            # Clear financial states
            for agent in self.metrics['financial_states']:
                if self.current_day in self.metrics['financial_states'][agent]:
                    del self.metrics['financial_states'][agent][self.current_day]
            
            # Clear activities
            for agent in self.metrics['activities']:
                if self.current_day in self.metrics['activities'][agent]:
                    del self.metrics['activities'][agent][self.current_day]
            
            print(f"Cleared metrics data for Day {TimeManager.get_current_day()}")
        
        except Exception as e:
            print(f"Error clearing daily metrics: {str(e)}")
            traceback.print_exc()

    def _get_location_type(self, location: str) -> str:
        """Get the type of a location."""
        if 'Coffee Shop' in location:
            return 'cafe'
        elif 'Restaurant' in location:
            return 'restaurant'
        elif 'Grocery' in location:
            return 'grocery'
        else:
            return 'other'

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics data."""
        return self.metrics_state

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics data."""
        return {
            'day': TimeManager.get_current_day(),
            'time': TimeManager.get_current_hour(),
            'metrics': self.metrics_state
        }

    def get_daily_metrics(self, agent_name: str, day: int) -> Dict[str, Any]:
        """Get daily metrics for a specific agent and day."""
        try:
            # Get all metrics for the agent
            agent_metrics = self.agent_metrics.get(agent_name, {})
            
            # Filter metrics for the specific day
            day_metrics = {
                metric_type: [
                    metric for metric in metrics
                    if metric.get('timestamp', {}).get('day') == day
                ]
                for metric_type, metrics in agent_metrics.items()
            }
            
            return day_metrics
            
        except Exception as e:
            print(f"Error getting daily metrics: {str(e)}")
            return {}
            
    def get_agent_metrics(self, agent_name: str, metric_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics for a specific agent."""
        try:
            if metric_type:
                return {metric_type: self.agent_metrics[agent_name][metric_type]}
            return dict(self.agent_metrics[agent_name])
        except Exception as e:
            print(f"Error getting agent metrics: {str(e)}")
            return {}
            
    def get_location_metrics(self, location_name: str, metric_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics for a specific location."""
        try:
            if metric_type:
                return {metric_type: self.location_metrics[location_name][metric_type]}
            return dict(self.location_metrics[location_name])
        except Exception as e:
            print(f"Error getting location metrics: {str(e)}")
            return {}
            
    def get_simulation_metrics(self, metric_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get simulation-wide metrics."""
        try:
            if metric_type:
                return {metric_type: self.simulation_metrics[metric_type]}
            return dict(self.simulation_metrics)
        except Exception as e:
            print(f"Error getting simulation metrics: {str(e)}")
            return {}

    def save_metrics(self, force: bool = False) -> None:
        """Saves the pending metrics buffer to the append-only log file."""
        if not self._pending_metrics:
            return

        try:
            with open(self.metrics_file, 'a') as f:
                for event in self._pending_metrics:
                    f.write(json.dumps(event) + '\n')
            
            # Clear the buffer after a successful write
            self._pending_metrics.clear()
            
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
            traceback.print_exc()

    def save_final_metrics(self) -> None:
        """Save final metrics at the end of simulation."""
        try:
            self.save_metrics(force=True)
            print("Final metrics saved successfully")
        except Exception as e:
            print(f"Error saving final metrics: {str(e)}")
            traceback.print_exc()

    def clear(self):
        """Reset all metrics data."""
        self.current_day = 1
        self.current_time = 7
        self.metrics_state = {
            'shop_metrics': {},
            'customer_metrics': {},
        }
        self._pending_metrics = []
        
        # Initialize first day
        self.ensure_day_initialized(TimeManager.get_current_day())
        print("Metrics data cleared and reset.")

    def generate_daily_summary(self, day: int) -> Dict[str, Any]:
        """Generate summary of metrics for a specific day."""
        day_data = {
            'energy': {},
            'grocery': {},
            'financial': {},
            'activities': {},
            'locations': {}
        }

        # Collect data for each agent
        for agent_name, agent in self.agents.items():
            # Energy levels
            energy_level = agent.energy_system.get_energy(agent.name)
            day_data['energy'][agent_name] = energy_level

            # Grocery levels
            grocery_level = agent.grocery_system.get_level(agent.name)
            day_data['grocery'][agent_name] = grocery_level

            # Financial state
            day_data['financial'][agent_name] = {
                'money': agent.money,
                'daily_income': agent.daily_income,
                'daily_expenses': agent.daily_expenses
            }

            # Activities
            day_data['activities'][agent_name] = agent.current_activity

            # Locations
            day_data['locations'][agent_name] = agent.get_current_location_name()

        return day_data

    def store_daily_summary(self, day: int, summary: Dict[str, Any]) -> None:
        """Store the generated daily summary."""
        try:
            # Add metadata to the summary
            summary['metadata'] = {
                'day': day,
            }
            
            # Read existing summaries if file exists
            existing_summaries = {}
            if os.path.exists(self.daily_summaries_file):
                try:
                    with open(self.daily_summaries_file, 'r') as f:
                        existing_summaries = json.load(f)
                except json.JSONDecodeError:
                    print("Warning: Could not read existing daily summaries file, starting fresh")
            
            # Update or add the new summary
            existing_summaries[f'day_{day}'] = summary
            
            # Write back to file
            with open(self.daily_summaries_file, 'w') as f:
                json.dump(existing_summaries, f, indent=2)
                
            print(f"Daily summary for Day {day} saved to: {self.daily_summaries_file}")
            
        except Exception as e:
            print(f"Error storing daily summary: {str(e)}")
            traceback.print_exc()

