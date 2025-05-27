from datetime import datetime
import os
import json
import uuid
from typing import Dict, List, Optional, Any
import traceback

class StabilityMetrics:
    """Track and analyze metrics for the entire town simulation"""
    
    def __init__(self):
        """Initialize the metrics tracker"""
        self.current_day = 0  # Will be incremented to 1 in first new_day() call by the main simulation loop
        self.daily_metrics_data = {}  # Renamed from daily_metrics to avoid confusion with methods
        self.all_interactions = []  # Renamed from interactions to avoid confusion
        self.daily_summary_cache = {}  # Renamed from daily_summaries to store raw data for file saving
        self.conversation_topics_by_day = {}  # Renamed for clarity
        self.relationship_trends_by_day = {}  # Renamed for clarity
        # self.discount_trends = {} # This wasn't actively used, can be re-added if needed
        # self.new_day()  # REMOVED: Initialize first day - This will be called by the main simulation loop

    def record_interaction(self, agent_name: str, location: str, event_type: str, details: Dict) -> None:
        """Record an interaction or event"""
        try:
            interaction = {
                'agent': agent_name,
                'location': location,
                'type': event_type,
                'details': details,
                'day': self.current_day,
                'time': details.get('time', 0)  # Ensure time is present
            }
            self.all_interactions.append(interaction)
            
            if self.current_day not in self.daily_metrics_data:
                # This case should ideally not happen if new_day() is called correctly
                print(f"Warning: Metrics for day {self.current_day} not initialized before recording interaction. Calling new_day().")
                self.new_day(force_day=self.current_day if self.current_day > 0 else 1)

            # Ensure day structure exists
            if self.current_day not in self.daily_metrics_data:
                self.daily_metrics_data[self.current_day] = {
                    'store_visits': {"Fried Chicken Shop": []},
                    'transactions': [],
                    'conversations': [],
                    'social_interactions': []
                }

            daily_data = self.daily_metrics_data[self.current_day]
            
            # Initialize lists if not present (defensive coding)
            daily_data.setdefault('store_visits', {"Fried Chicken Shop": []})
            daily_data.setdefault('transactions', [])
            daily_data.setdefault('conversations', [])
            daily_data.setdefault('social_interactions', [])
            if "Fried Chicken Shop" not in daily_data['store_visits']:
                daily_data['store_visits']["Fried Chicken Shop"] = []

            if event_type == "store_visit" and location == "Fried Chicken Shop":
                daily_data['store_visits']["Fried Chicken Shop"].append(details)
                # print(f"Day {self.current_day}: Recorded store visit for {agent_name} at Fried Chicken Shop")
                
            elif event_type == "purchase" and location == "Fried Chicken Shop":  # Assuming purchase implies food here
                # Ensure 'amount' is present, default to 0 if not for safety
                details.setdefault('amount', 0.0)
                daily_data['transactions'].append(details)
                # print(f"Day {self.current_day}: Recorded purchase for {agent_name} at Fried Chicken Shop: ${details['amount']:.2f}")

            elif event_type == "conversation":
                daily_data['conversations'].append(details)
                self._analyze_conversation_topics(details)
                
            elif event_type == "social_interaction":
                daily_data['social_interactions'].append(details)
                self._track_relationship_development(details)
                
        except Exception as e:
            print(f"Error recording interaction: {str(e)}")
            traceback.print_exc()

    def _analyze_conversation_topics(self, conversation_data: Dict) -> None:
        """Analyze and track conversation topics for the current day"""
        if 'content' not in conversation_data:
            return
            
        content = conversation_data['content'].lower()
        day = self.current_day
        
        if day not in self.conversation_topics_by_day:
            self.conversation_topics_by_day[day] = {
                'food': 0, 'work': 0, 'social': 0, 'shopping': 0, 
                'family': 0, 'entertainment': 0, 'other': 0
            }
            
        topics = {
            'food': ['eat', 'food', 'lunch', 'dinner', 'breakfast', 'restaurant', 'chicken', 'hungry', 'meal'],
            'work': ['work', 'job', 'office', 'meeting', 'project', 'deadline', 'career', 'colleague'],
            'social': ['friend', 'together', 'meet', 'hangout', 'chat', 'party', 'event', 'community'],
            'shopping': ['shop', 'buy', 'store', 'purchase', 'discount', 'mall', 'market', 'groceries'],
            'family': ['family', 'home', 'spouse', 'partner', 'sister', 'brother', 'roommate', 'girlfriend', 'boyfriend', 'children', 'parents'],
            'entertainment': ['movie', 'game', 'fun', 'hobby', 'exercise', 'gym', 'park', 'music', 'book']
        }
        
        found_topic = False
        for topic, keywords in topics.items():
            if any(keyword in content for keyword in keywords):
                self.conversation_topics_by_day[day][topic] += 1
                found_topic = True
        if not found_topic:
            self.conversation_topics_by_day[day]['other'] +=1

    def _track_relationship_development(self, interaction_data: Dict) -> None:
        """Track relationship development between agents for the current day"""
        participants = interaction_data.get('participants')
        if not participants or not isinstance(participants, list) or len(participants) < 2:
            return # Needs at least two participants
            
        day = self.current_day
        
        if day not in self.relationship_trends_by_day:
            self.relationship_trends_by_day[day] = {}
            
        # Create unique pair key for the interaction (sorted tuple of names)
        # Ensure all participants are strings
        string_participants = sorted([str(p) for p in participants])

        for i in range(len(string_participants)):
            for j in range(i + 1, len(string_participants)):
                agent1 = string_participants[i]
                agent2 = string_participants[j]
                pair_key = tuple(sorted((agent1, agent2))) # Ensures consistency
                
                if pair_key not in self.relationship_trends_by_day[day]:
                    self.relationship_trends_by_day[day][pair_key] = {
                        'interaction_count': 0,
                        'locations': set(), # Store location names
                        'activities': set() # Store activity types/content snippets
                    }
                
                self.relationship_trends_by_day[day][pair_key]['interaction_count'] += 1
                if interaction_data.get('location'):
                    self.relationship_trends_by_day[day][pair_key]['locations'].add(str(interaction_data['location']))
                # Add more detail to activity, e.g. first few words of conversation or type of interaction
                activity_detail = interaction_data.get('type', 'unknown_interaction')
                if 'content' in interaction_data and isinstance(interaction_data['content'], str):
                     activity_detail += ": " + interaction_data['content'][:30] # first 30 chars

                self.relationship_trends_by_day[day][pair_key]['activities'].add(activity_detail)

    def get_daily_summary_data(self, day: int) -> Dict:
        """Generate a comprehensive daily summary data structure and cache it."""
        if day in self.daily_summary_cache:
            return self.daily_summary_cache[day]

        if day not in self.daily_metrics_data:
            # Return a default empty structure if no data for the day
            empty_summary = {
                "date": f"Day {day}", "error": "No metrics data recorded for this day.",
                "fried_chicken_shop": {"total_visits": 0, "total_sales": 0, "total_revenue": 0.0, "discount_usage": 0, "customer_details": []},
                "social_insights": {"conversation_topics": {}, "active_relationships": [], "popular_locations": {}},
                "town_activity": {"location_stats": {}}
            }
            self.daily_summary_cache[day] = empty_summary
            return empty_summary
            
        daily_data = self.daily_metrics_data[day]
        
        # Fried Chicken Shop metrics
        fcs_visits = len(daily_data['store_visits'].get("Fried Chicken Shop", []))
        fcs_transactions = [t for t in daily_data.get('transactions', []) if t.get('location') == "Fried Chicken Shop"]
        fcs_revenue = sum(t.get('amount', 0.0) for t in fcs_transactions)
        fcs_discount_usage = sum(1 for t in fcs_transactions if t.get('used_discount', False))
        
        fcs_customer_details = []
        for t in fcs_transactions:
            time_val = t.get('time', 0) # Get time, default to 0
            hour_of_day = time_val % 24 if isinstance(time_val, int) else 0 # Handle if time is not int
            fcs_customer_details.append({
                "time": f"{hour_of_day:02d}:00",
                "agent": t.get('agent', 'Unknown Agent'),
                "amount": t.get('amount', 0.0),
                "used_discount": t.get('used_discount', False)
            })

        # Location activity summary (all locations)
        location_activity = {}
        for interaction in self.all_interactions: # Use all_interactions for comprehensive view
            if interaction['day'] == day and interaction.get('location'):
                loc_name = str(interaction['location']) # Ensure loc_name is a string
                if loc_name not in location_activity:
                    location_activity[loc_name] = {
                        'total_visits': 0, 'visitors': set(), 
                        'peak_hours': self._get_peak_hours(day, loc_name), 'activities': []
                    }
                location_activity[loc_name]['total_visits'] += 1
                location_activity[loc_name]['visitors'].add(str(interaction['agent']))
                location_activity[loc_name]['activities'].append(str(interaction['type']))
        
        # Conversation topics for the day
        topics_today = self.conversation_topics_by_day.get(day, {})
        popular_topics = sorted(topics_today.items(), key=lambda x: x[1], reverse=True)
        
        # Relationship insights for the day
        relationships_today = self.relationship_trends_by_day.get(day, {})
        active_relationships = []
        for pair, data in relationships_today.items():
            if data['interaction_count'] >= 2: # Threshold for 'active'
                active_relationships.append({
                    'agents': list(pair), 'interaction_count': data['interaction_count'],
                    'shared_locations': list(data['locations']), 'activities': list(data['activities'])
                })
        
        summary_data = {
            "date": f"Day {day}",
            "fried_chicken_shop": {
                "total_visits": fcs_visits, "total_sales": len(fcs_transactions),
                "total_revenue": fcs_revenue, "discount_usage": fcs_discount_usage,
                "customer_details": fcs_customer_details
            },
            "social_insights": {
                "conversation_topics": {topic: count for topic, count in popular_topics if count > 0},
                "active_relationships": active_relationships,
                "popular_locations": { # Activity at popular locations
                    loc_name: {
                        "total_visits": data['total_visits'], 
                        "unique_visitors": len(data['visitors']),
                        "common_activities": self._get_common_activities(data['activities'])
                    } for loc_name, data in location_activity.items() if data['total_visits'] > 0 # Only show active locs
                }
            },
            "town_activity": { # General town stats
                "total_interactions_recorded": len([inter for inter in self.all_interactions if inter['day'] == day]),
                "location_stats": {
                    loc_name: {
                        "total_visits": data['total_visits'], 
                        "unique_visitors": len(data['visitors']),
                        "peak_hours_display": ", ".join(f"{h:02d}:00" for h in data['peak_hours']) if data['peak_hours'] else "N/A"
                    } for loc_name, data in location_activity.items() if data['total_visits'] > 0
                }
            }
        }
        self.daily_summary_cache[day] = summary_data # Cache it
        return summary_data

    def print_daily_summary(self, summary_data: Dict, day: int) -> None:
        """Prints the daily summary to the console."""
        if not summary_data or summary_data.get("error"):
            print(f"\n=== No Summary Data to Print for Day {day} ===\n")
            if summary_data.get("error"): print(summary_data["error"])
            return

        print(f"\n=== Daily Summary for Day {day} ({summary_data.get('date', '')}) ===\n")
        
        fcs = summary_data.get("fried_chicken_shop", {})
        print("\nFried Chicken Shop Activity:")
        print(f"  Total Visits: {fcs.get('total_visits', 0)}")
        print(f"  Total Sales: {fcs.get('total_sales', 0)}")
        print(f"  Total Revenue: ${fcs.get('total_revenue', 0.0):.2f}")
        print(f"  Discount Usage: {fcs.get('discount_usage', 0)} times")
        if fcs.get('customer_details'):
            print("\nCustomer Details:")
            for cust in fcs['customer_details']:
                disc_text = " (discount)" if cust.get('used_discount') else ""
                print(f"    {cust.get('time', 'N/A')} - {cust.get('agent', 'N/A')}: ${cust.get('amount', 0.0):.2f}{disc_text}")
        
        social = summary_data.get("social_insights", {})
        print("\nSocial Insights:")
        if social.get("conversation_topics"):
            print("\nPopular Conversation Topics:")
            for topic, count in social["conversation_topics"].items():
                print(f"    - {topic.capitalize()}: {count} conversations")
        else:
            print("  No specific conversation topics recorded.")

        if social.get("active_relationships"):
            print("\nActive Social Connections (2+ interactions):")
            for rel in social["active_relationships"]:
                agents = " & ".join(rel.get('agents', ['?A', '?B']))
                print(f"    - {agents}: {rel.get('interaction_count', 0)} interactions")
                # print(f"      Locations: {list(rel.get('shared_locations', []))}") # Optional: more detail
        else:
            print("  No significant social connections formed today.")

        town_act = summary_data.get("town_activity", {})
        print("\nTown Activity Overview:")
        print(f"  Total Interactions Recorded Today: {town_act.get('total_interactions_recorded', 0)}")
        if town_act.get("location_stats"):
            print("\nLocation Activity:")
            for loc, data in town_act["location_stats"].items():
                print(f"    {loc}:")
                print(f"      Total Visits: {data.get('total_visits', 0)}")
                print(f"      Unique Visitors: {data.get('unique_visitors', 0)}")
                print(f"      Peak Hours: {data.get('peak_hours_display', 'N/A')}")
                # Common activities at popular locations
                pop_loc_data = social.get('popular_locations', {}).get(loc, {})
                if pop_loc_data.get('common_activities'):
                    acts = pop_loc_data['common_activities']
                    act_str = ", ".join([f"{k} ({v})" for k,v in acts.items()])
                    print(f"      Common Activities: {act_str}")
        else:
            print("  No specific location activity recorded.")
        print("======================================\n")

    def _get_common_activities(self, activities: List[str]) -> Dict[str, int]:
        """Get most common activities from a list of activity strings"""
        activity_counts: Dict[str, int] = {}
        for activity in activities:
            # Normalize or simplify activity string if needed here
            act_key = str(activity).strip() # Ensure string and strip
            activity_counts[act_key] = activity_counts.get(act_key, 0) + 1
        # Return top 3 or so, sorted
        return dict(sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:3])

    def _get_peak_hours(self, day: int, location_name: str) -> List[int]:
        """Get peak hours for a specific location on a given day"""
        hour_counts: Dict[int, int] = {}
        for interaction in self.all_interactions: # Analyze all interactions for the day
            if interaction['day'] == day and str(interaction.get('location')) == location_name:
                time_val = interaction.get('time')
                if isinstance(time_val, int): # Ensure time is an int
                    hour = time_val % 24
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if not hour_counts: return []
        max_visits = max(hour_counts.values())
        return sorted([h for h, count in hour_counts.items() if count == max_visits])

    def new_day(self, force_day: Optional[int] = None) -> int:
        """Initialize metrics for a new day or specified day."""
        if force_day is not None:
            self.current_day = force_day
        else:
            self.current_day += 1
        
        # Ensure current day structure is initialized
        if self.current_day not in self.daily_metrics_data:
            self.daily_metrics_data[self.current_day] = {
                'store_visits': {"Fried Chicken Shop": []}, # Specific for FCS
                'transactions': [], # All transactions, will be filtered for FCS later
                'conversations': [],
                'social_interactions': [] 
                # Add other general daily metrics categories if needed
            }
        # Also initialize for topic/relationship tracking if not present
        if self.current_day not in self.conversation_topics_by_day:
             self.conversation_topics_by_day[self.current_day] = {'food': 0, 'work': 0, 'social': 0, 'shopping': 0, 'family': 0, 'entertainment': 0, 'other': 0}
        if self.current_day not in self.relationship_trends_by_day:
            self.relationship_trends_by_day[self.current_day] = {}

        print(f"Metrics initialized for Day {self.current_day}")
        return self.current_day

    def save_final_metrics(self, filepath: str) -> None:
        """Save final aggregated sales metrics for the entire simulation, focusing on Fried Chicken Shop."""
        try:
            # Aggregate Fried Chicken Shop data across all days
            total_fcs_visits = 0
            total_fcs_sales_count = 0
            total_fcs_revenue = 0.0
            total_fcs_discount_usage = 0
            
            daily_fcs_breakdown = {}

            for day in range(1, self.current_day + 1):
                if day in self.daily_metrics_data:
                    day_data = self.daily_metrics_data[day]
                    fcs_day_visits = len(day_data['store_visits'].get("Fried Chicken Shop", []))
                    
                    fcs_day_transactions = [
                        t for t in day_data.get('transactions', []) 
                        if t.get('location') == "Fried Chicken Shop"
                    ]
                    fcs_day_sales_count = len(fcs_day_transactions)
                    fcs_day_revenue = sum(t.get('amount', 0.0) for t in fcs_day_transactions)
                    fcs_day_discount_usage = sum(1 for t in fcs_day_transactions if t.get('used_discount', False))
                    
                    total_fcs_visits += fcs_day_visits
                    total_fcs_sales_count += fcs_day_sales_count
                    total_fcs_revenue += fcs_day_revenue
                    total_fcs_discount_usage += fcs_day_discount_usage
                    
                    daily_fcs_breakdown[f'day_{day}'] = {
                        'visits': fcs_day_visits,
                        'sales_count': fcs_day_sales_count,
                        'revenue': round(fcs_day_revenue, 2),
                        'discount_usage': fcs_day_discount_usage
                    }

            final_metrics_data = {
                'simulation_run_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'total_simulated_days': self.current_day,
                'fried_chicken_shop_overall_performance': {
                    'total_visits': total_fcs_visits,
                    'total_sales_count': total_fcs_sales_count,
                    'total_revenue': round(total_fcs_revenue, 2),
                    'total_discount_usage': total_fcs_discount_usage,
                    'average_revenue_per_day': round(total_fcs_revenue / self.current_day if self.current_day > 0 else 0, 2),
                },
                'fried_chicken_shop_daily_breakdown': daily_fcs_breakdown
            }
            
            # Create directory for the filepath if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(final_metrics_data, f, indent=2)
            print(f"\nFinal sales metrics saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving final metrics: {str(e)}")
            traceback.print_exc()

    def save_all_daily_summaries_to_file(self, base_dir_override: Optional[str] = None) -> None:
        """Save all cached daily summaries (comprehensive insights) to a single JSON file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if base_dir_override:
                metrics_dir = base_dir_override
            else:
                # Default path construction
                current_script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_script_dir) # Assuming Stability_Metrics.py is in LLMAgentsTown_experiment
                if not project_root.endswith("LLMAgentsTown_Stability"): # Adjust if script is deeper
                     project_root = os.path.dirname(project_root)

                metrics_dir = os.path.join(project_root, "LLMAgentsTown_memory_records", "simulation_daily_summaries")
            
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Ensure all days are processed for summary data if not already cached
            for day_num in range(1, self.current_day + 1):
                if day_num not in self.daily_summary_cache:
                    # This will generate and cache it
                    self.get_daily_summary_data(day_num) 

            # Prepare data for saving
            all_summaries_data = {
                'simulation_run_timestamp': timestamp,
                'total_simulated_days': self.current_day,
                'daily_insights': self.daily_summary_cache # Use the cached summaries
            }
            
            summary_filename = f"all_daily_insights_{timestamp}.json"
            summary_filepath = os.path.join(metrics_dir, summary_filename)
            
            with open(summary_filepath, 'w') as f:
                json.dump(all_summaries_data, f, indent=2)
            print(f"\nAll daily insight summaries saved to: {summary_filepath}")
            
        except Exception as e:
            print(f"Error saving all daily summaries: {str(e)}")
            traceback.print_exc()

    def record_hour_metrics(self, current_day: int, current_hour: int, agents: List) -> None:
        """Record metrics for the current hour, including agent states and interactions."""
        try:
            if current_day not in self.daily_metrics_data:
                self.new_day(force_day=current_day)

            # Get the metrics data for the current day
            day_data = self.daily_metrics_data[current_day]

            # Record agent states for this hour
            hour_key = f"{current_hour:02d}:00"
            if 'hourly_states' not in day_data:
                day_data['hourly_states'] = {}
            
            day_data['hourly_states'][hour_key] = []
            
            # Record each agent's state
            for agent in agents:
                agent_state = {
                    'name': agent.name,
                    'location': agent.current_location,
                    'energy_level': agent.energy_level,
                    'grocery_level': agent.grocery_level,
                    'money': agent.money,
                    'current_activity': agent.current_activity if hasattr(agent, 'current_activity') else 'unknown'
                }
                day_data['hourly_states'][hour_key].append(agent_state)

            # Record any pending interactions or purchases
            if 'pending_interactions' in day_data:
                for interaction in day_data['pending_interactions']:
                    interaction['hour'] = current_hour
                    self.record_interaction(
                        interaction['agent'],
                        interaction['location'],
                        interaction['type'],
                        interaction
                    )
                day_data['pending_interactions'] = []  # Clear after processing

        except Exception as e:
            print(f"Error recording hour metrics: {str(e)}")
            traceback.print_exc()