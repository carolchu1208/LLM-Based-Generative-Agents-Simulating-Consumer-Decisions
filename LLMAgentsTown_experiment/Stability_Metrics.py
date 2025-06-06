import json
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import uuid

class StabilityMetrics:
    """Track and analyze metrics for the entire town simulation"""
    
    def __init__(self):
        """Initialize the metrics tracker"""
        self.current_day = 1
        self.daily_metrics_data = {}
        self.all_interactions = []
        self.daily_summary_cache = {}
        self.conversation_topics_by_day = {}
        self.relationship_trends_by_day = {}
        self.satisfaction_ratings_by_day = {}
        self.simulation_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.start_time = datetime.now()
        
        # Standardized metric categories
        self.metric_categories = {
            'store_visits': {},
            'purchases': {},
            'conversations': {},
            'satisfaction': {},
            'energy': {},
            'travel': {}
        }
        
        # Initialize first day
        self.ensure_day_initialized(1)
        print("Metrics system initialized with timestamp:", self.simulation_start_time)

    def clear(self):
        """Clear all metrics data to start fresh."""
        self.current_day = 1
        self.daily_metrics_data = {}
        self.all_interactions = []
        self.daily_summary_cache = {}
        self.conversation_topics_by_day = {}
        self.relationship_trends_by_day = {}
        self.satisfaction_ratings_by_day = {}
        self.simulation_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        print("Metrics data cleared and reset.")

    def ensure_day_initialized(self, day: int) -> None:
        """Initialize metrics data structure for a given day."""
        if day not in self.daily_metrics_data:
            self.daily_metrics_data[day] = {
                'store_visits': defaultdict(lambda: {
                    'visits': [],
                    'purchase_records': [],
                    'total_revenue': 0.0,
                    'hourly_visits': defaultdict(int),
                    'satisfaction_ratings': []
                }),
                'purchases': defaultdict(lambda: {
                    'total_amount': 0.0,
                    'count': 0,
                    'items': [],
                    'hourly_distribution': defaultdict(int)
                }),
                'conversations': defaultdict(lambda: {
                    'count': 0,
                    'participants': set(),
                    'topics': defaultdict(int),
                    'hourly_distribution': defaultdict(int)
                }),
                'satisfaction': defaultdict(lambda: {
                    'ratings': [],
                    'average': 0.0,
                    'count': 0
                }),
                'energy': defaultdict(lambda: {
                    'average': 0.0,
                    'min': 100,
                    'max': 0,
                    'hourly_averages': defaultdict(list)
                }),
                'travel': defaultdict(lambda: {
                    'count': 0,
                    'total_distance': 0,
                    'hourly_distribution': defaultdict(int)
                })
            }
            print(f"DEBUG: Initialized metrics for day {day} with real data tracking")

    def record_store_visit(self, agent_name: str, location: str, time: int, day: int, details: Dict = None):
        """Record a store visit with standardized metrics."""
        self.ensure_day_initialized(day)
        hour = time % 24
        
        visit_data = {
            'agent': agent_name,
            'time': time,
            'hour': hour,
            'day': day,
            'details': details or {}
        }
        
        self.daily_metrics_data[day]['store_visits'][location]['visits'].append(visit_data)
        self.daily_metrics_data[day]['store_visits'][location]['hourly_visits'][hour] += 1

    def record_purchase(self, agent_name: str, location: str, time: int, day: int, amount: float, details: Dict = None):
        """Record a purchase with standardized metrics."""
        self.ensure_day_initialized(day)
        hour = time % 24
        
        purchase_data = {
            'agent': agent_name,
            'time': time,
            'hour': hour,
            'day': day,
            'amount': amount,
            'details': details or {}
        }
        
        self.daily_metrics_data[day]['purchases'][location]['total_amount'] += amount
        self.daily_metrics_data[day]['purchases'][location]['count'] += 1
        self.daily_metrics_data[day]['purchases'][location]['hourly_distribution'][hour] += 1
        
        if 'items' in details:
            self.daily_metrics_data[day]['purchases'][location]['items'].extend(details['items'])

    def record_interaction(self, agent_name: str, location: str, event_type: str, details: Dict) -> None:
        """Record a social interaction with standardized metrics."""
        day = self.current_day
        self.ensure_day_initialized(day)
        time = details.get('time', 0)
        hour = time % 24
        
        interaction_data = {
            'agent': agent_name,
            'location': location,
            'event_type': event_type,
            'time': time,
            'hour': hour,
            'day': day,
            'details': details
        }
        
        self.daily_metrics_data[day]['conversations'][location]['count'] += 1
        self.daily_metrics_data[day]['conversations'][location]['hourly_distribution'][hour] += 1
        
        if 'participants' in details:
            self.daily_metrics_data[day]['conversations'][location]['participants'].update(details['participants'])
        
        if 'topic' in details:
            self.daily_metrics_data[day]['conversations'][location]['topics'][details['topic']] += 1

    def record_satisfaction_rating(self, agent_name: str, location: str, rating_data: Dict) -> None:
        """Record a satisfaction rating with standardized metrics."""
        day = self.current_day
        self.ensure_day_initialized(day)
        
        rating = rating_data.get('rating', 0)
        self.daily_metrics_data[day]['satisfaction'][location]['ratings'].append(rating)
        self.daily_metrics_data[day]['satisfaction'][location]['count'] += 1
        
        # Update average
        ratings = self.daily_metrics_data[day]['satisfaction'][location]['ratings']
        self.daily_metrics_data[day]['satisfaction'][location]['average'] = sum(ratings) / len(ratings)

    def get_daily_summary_data(self, day: int) -> Dict[str, Any]:
        """Get summary data for a specific day."""
        try:
            if day not in self.daily_metrics_data:
                return {
                    'day': day,
                    'conversations': [],
                    'activities': [],
                    'locations_visited': [],
                    'energy_levels': {},
                    'grocery_levels': {},
                    'money_levels': {}
                }

            daily_data = self.daily_metrics_data[day]
            
            # Get conversations, handling missing transactions key
            conversations = []
            if 'transactions' in daily_data and 'all' in daily_data['transactions']:
                conversations = [t for t in daily_data['transactions']['all'] 
                               if t.get('type') == 'conversation']
            
            # Get activities
            activities = daily_data.get('activities', [])
            
            # Get location visits
            locations_visited = daily_data.get('locations_visited', [])
            
            # Get agent states
            energy_levels = daily_data.get('energy_levels', {})
            grocery_levels = daily_data.get('grocery_levels', {})
            money_levels = daily_data.get('money_levels', {})
            
            return {
                'day': day,
                'conversations': conversations,
                'activities': activities,
                'locations_visited': locations_visited,
                'energy_levels': energy_levels,
                'grocery_levels': grocery_levels,
                'money_levels': money_levels
            }
            
        except Exception as e:
            print(f"Error getting daily summary data for day {day}: {str(e)}")
            traceback.print_exc()
            return {
                'day': day,
                'conversations': [],
                'activities': [],
                'locations_visited': [],
                'energy_levels': {},
                'grocery_levels': {},
                'money_levels': {}
            }

    def _get_peak_hour(self, hourly_data: Dict[int, int]) -> Optional[int]:
        """Get the hour with the highest activity."""
        if not hourly_data:
            return None
        return max(hourly_data.items(), key=lambda x: x[1])[0]

    def _calculate_average(self, values: List[float]) -> float:
        """Calculate average of a list of values."""
        return sum(values) / len(values) if values else 0.0

    def _get_top_items(self, items: Dict[str, int], limit: int) -> List[Tuple[str, int]]:
        """Get top N items by count."""
        return sorted(items.items(), key=lambda x: x[1], reverse=True)[:limit]

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
        """Get simplified daily summary with other shops stats and basic conversation insights"""
        if day not in self.daily_metrics_data:
            self.ensure_day_initialized(day)
        
        # Calculate shop activity for ALL OTHER SHOPS (exclude Fried Chicken Shop)
        other_shops_stats = {}
        
        for location_name, store_data in self.daily_metrics_data[day]['store_visits'].items():
            # Skip Fried Chicken Shop - that goes to focused metrics file
            if location_name == 'Fried Chicken Shop':
                continue
                
            location_visits = len(store_data.get('visits', []))
            location_purchases = len(store_data.get('purchase_records', []))
            location_revenue = store_data.get('total_revenue', 0.0)
            
            # Calculate peak hour for this location
            hourly_visits = store_data.get('hourly_visits', {})
            peak_hour = None
            if hourly_visits:
                max_visits = max(v for v in hourly_visits.values() if isinstance(v, int))
                if max_visits > 0:
                    peak_hours = [int(h) for h, v in hourly_visits.items() if isinstance(v, int) and v == max_visits]
                    peak_hour = peak_hours[0] if peak_hours else None
            
            # Record stats for any location with activity
            if location_visits > 0 or location_purchases > 0 or location_revenue > 0:
                other_shops_stats[location_name] = {
                    'visits': location_visits,
                    'purchases': location_purchases,
                    'revenue': round(location_revenue, 2),
                    'peak_hour': peak_hour
                }
        
        # Basic conversation insights - just count and categorize
        conversations = [t for t in self.daily_metrics_data[day]['transactions']['all'] 
                        if t.get('event_type') in ['conversation', 'social_interaction']]
        
        # Simple conversation categorization
        conversation_themes = {
            'household': 0,
            'food': 0,
            'work': 0,
            'shopping': 0,
            'social': 0
        }
        
        # Simple relationship tracking
        agent_interactions = {}
        conversation_locations = {}
        
        theme_keywords = {
            'household': ['family', 'home', 'house', 'spouse', 'wife', 'husband', 'together'],
            'food': ['eat', 'food', 'meal', 'cook', 'restaurant', 'lunch', 'dinner', 'breakfast', 'fried chicken'],
            'work': ['work', 'job', 'office', 'meeting', 'project', 'busy', 'schedule'],
            'shopping': ['buy', 'shop', 'store', 'purchase', 'money', 'price', 'groceries'],
            'social': ['hello', 'how are you', 'nice', 'good', 'chat', 'talk', 'friend']
        }
        
        for conv in conversations:
            conv_data = conv.get('details', {})
            participants = conv_data.get('participants', [])
            content = conv_data.get('content', '').lower()
            location = conv_data.get('location', 'Unknown')
            
            # Count conversations by location
            conversation_locations[location] = conversation_locations.get(location, 0) + 1
            
            # Simple theme detection
            for theme, keywords in theme_keywords.items():
                if any(keyword in content for keyword in keywords):
                    conversation_themes[theme] += 1
                    break
            else:
                conversation_themes['social'] += 1  # Default to social if no specific theme found
            
            # Track agent interactions
            for participant in participants:
                agent_interactions[participant] = agent_interactions.get(participant, 0) + 1
        
        # Simple relationship pairs (just count interactions between agents)
        relationship_pairs = {}
        for conv in conversations:
            conv_data = conv.get('details', {})
            participants = conv_data.get('participants', [])
            if len(participants) >= 2:
                for i, agent1 in enumerate(participants):
                    for agent2 in participants[i+1:]:
                        pair_key = tuple(sorted([agent1, agent2]))
                        relationship_pairs[pair_key] = relationship_pairs.get(pair_key, 0) + 1
        
        return {
            'date': f"Day {day}",
            'other_shops_stats': other_shops_stats,
            'conversation_insights': {
                'total_conversations': len(conversations),
                'conversation_themes': conversation_themes,
                'active_agents': agent_interactions,
                'conversation_locations': conversation_locations,
                'agent_pairs': relationship_pairs
            }
        }

    def _get_most_social_agents(self, day: int) -> List[Dict]:
        """Get agents with the most social interactions on a given day"""
        agent_interaction_count = {}
        
        conversations = [t for t in self.daily_metrics_data[day]['transactions']['all'] 
                        if t.get('event_type') in ['conversation', 'social_interaction']]
        
        for conv in conversations:
            participants = conv.get('details', {}).get('participants', [])
            for agent in participants:
                agent_interaction_count[agent] = agent_interaction_count.get(agent, 0) + 1
        
        # Sort and return top 3
        sorted_agents = sorted(agent_interaction_count.items(), key=lambda x: x[1], reverse=True)
        return [{'name': agent, 'interactions': count} for agent, count in sorted_agents[:3]]

    def print_daily_summary(self, day: int) -> None:
        """Print simplified daily summary with shop stats and basic conversation insights"""
        try:
            summary_data = self.get_daily_summary_data(day)
            
            print(f"\n{'='*50}")
            print(f"🏛️  DAILY SUMMARY - {summary_data['date']}")
            print(f"{'='*50}")
            
            # Other Shops Stats (excluding Fried Chicken Shop)
            other_shops = summary_data.get('other_shops_stats', {})
            
            print(f"\n🏪 OTHER SHOPS PERFORMANCE")
            if other_shops:
                for shop_name, stats in other_shops.items():
                    peak_info = f" (peak: {stats['peak_hour']}:00)" if stats['peak_hour'] is not None else " (no peak)"
                    print(f"   • {shop_name}: {stats['visits']} visits, {stats['purchases']} purchases, ${stats['revenue']:.2f} revenue{peak_info}")
            else:
                print("   • No other shop activity recorded")
            
            # Conversation Insights
            conv_insights = summary_data.get('conversation_insights', {})
            
            print(f"\n💬 CONVERSATION INSIGHTS")
            print(f"   Total Conversations: {conv_insights.get('total_conversations', 0)}")
            
            # Conversation themes
            themes = conv_insights.get('conversation_themes', {})
            if any(themes.values()):
                print(f"\n   🎯 Conversation Themes:")
                for theme, count in themes.items():
                    if count > 0:
                        print(f"     • {theme.capitalize()}: {count}")
            
            # Active agents
            active_agents = conv_insights.get('active_agents', {})
            if active_agents:
                print(f"\n   👤 Most Talkative Agents:")
                # Sort by interaction count
                sorted_agents = sorted(active_agents.items(), key=lambda x: x[1], reverse=True)
                for agent_name, count in sorted_agents[:3]:  # Top 3
                    print(f"     • {agent_name}: {count} conversations")
            
            # Conversation locations
            conv_locations = conv_insights.get('conversation_locations', {})
            if conv_locations:
                print(f"\n   📍 Conversation Locations:")
                for location, count in conv_locations.items():
                    print(f"     • {location}: {count} conversations")
            
            # Agent pairs (relationships)
            agent_pairs = conv_insights.get('agent_pairs', {})
            if agent_pairs:
                print(f"\n   👥 Agent Interactions:")
                # Sort by interaction count
                sorted_pairs = sorted(agent_pairs.items(), key=lambda x: x[1], reverse=True)
                for pair, count in sorted_pairs[:5]:  # Top 5 pairs
                    agents = " & ".join(pair)
                    print(f"     • {agents}: {count} interactions")
            
            print(f"\n{'='*50}")
            
        except Exception as e:
            print(f"Error printing daily summary: {str(e)}")
            traceback.print_exc()

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
            # Reset to specified day (used for new simulation starts)
            self.current_day = force_day
            if force_day == 1:
                # If we're starting a new simulation (day 1), clear old data
                print(f"DEBUG: Starting new simulation - resetting metrics to Day 1")
                self.clear()
                self.current_day = 1
        else:
            self.current_day += 1
        
        # Ensure current day structure is initialized
        if self.current_day not in self.daily_metrics_data:
            self.daily_metrics_data[self.current_day] = {
                'store_visits': defaultdict(lambda: {
                    'visits': [],
                    'purchase_records': [],
                    'total_revenue': 0.0,
                    'hourly_visits': defaultdict(int),
                    'satisfaction_ratings': []
                }),
                'purchases': defaultdict(lambda: {
                    'total_amount': 0.0,
                    'count': 0,
                    'items': [],
                    'hourly_distribution': defaultdict(int)
                }),
                'conversations': defaultdict(lambda: {
                    'count': 0,
                    'participants': set(),
                    'topics': defaultdict(int),
                    'hourly_distribution': defaultdict(int)
                }),
                'satisfaction': defaultdict(lambda: {
                    'ratings': [],
                    'average': 0.0,
                    'count': 0
                }),
                'energy': defaultdict(lambda: {
                    'average': 0.0,
                    'min': 100,
                    'max': 0,
                    'hourly_averages': defaultdict(list)
                }),
                'travel': defaultdict(lambda: {
                    'count': 0,
                    'total_distance': 0,
                    'hourly_distribution': defaultdict(int)
                })
            }
            
        # Also initialize for topic/relationship tracking if not present
        if self.current_day not in self.conversation_topics_by_day:
             self.conversation_topics_by_day[self.current_day] = {'food': 0, 'work': 0, 'social': 0, 'shopping': 0, 'family': 0, 'entertainment': 0, 'other': 0}
        if self.current_day not in self.relationship_trends_by_day:
            self.relationship_trends_by_day[self.current_day] = {}

        print(f"Metrics initialized for Day {self.current_day}")
        return self.current_day

    def reset_for_new_simulation(self) -> None:
        """Reset metrics for a fresh simulation start."""
        print("DEBUG: Resetting metrics for new simulation")
        self.clear()
        self.current_day = 0  # Will be set to 1 when new_day() is called

    def save_final_metrics(self, filepath: str) -> None:
        """Save focused Fried Chicken Shop business metrics for detailed performance analysis."""
        try:
            # Focus only on Fried Chicken Shop for detailed business analysis
            fcs_metrics = {}
            fcs_total_visits = 0
            fcs_total_purchases = 0
            fcs_total_revenue = 0.0
            fcs_total_discount_usage = 0
            fcs_daily_breakdown = {}
            fcs_hourly_patterns = {str(h): 0 for h in range(24)}
            
            # Aggregate Fried Chicken Shop data across all days
            for day in range(1, self.current_day + 1):
                if day in self.daily_metrics_data:
                    day_data = self.daily_metrics_data[day]
                    fcs_day_data = day_data['store_visits'].get('Fried Chicken Shop', {})
                    
                    day_visits = len(fcs_day_data.get('visits', []))
                    day_purchases = len(fcs_day_data.get('purchase_records', []))
                    day_revenue = fcs_day_data.get('total_revenue', 0.0)
                    day_discount_usage = fcs_day_data.get('discount_usage', 0)
                    
                    fcs_total_visits += day_visits
                    fcs_total_purchases += day_purchases
                    fcs_total_revenue += day_revenue
                    fcs_total_discount_usage += day_discount_usage
                    
                    # Aggregate hourly patterns
                    hourly_visits = fcs_day_data.get('hourly_visits', {})
                    for hour, visits in hourly_visits.items():
                        if isinstance(visits, int):
                            fcs_hourly_patterns[hour] = fcs_hourly_patterns.get(hour, 0) + visits
                    
                    # Daily breakdown with detailed analysis
                    fcs_daily_breakdown[f'day_{day}'] = {
                        'visits': day_visits,
                        'purchases': day_purchases,
                        'revenue': round(day_revenue, 2),
                        'discount_usage': day_discount_usage,
                        'conversion_rate': round((day_purchases / day_visits * 100) if day_visits > 0 else 0, 2),
                        'average_transaction': round(day_revenue / day_purchases if day_purchases > 0 else 0, 2),
                        'discount_day': day in [3, 4],  # Days 3 and 4 have 20% discount
                        'discount_impact': {
                            'expected_normal_revenue': day_purchases * 20.0 if day_purchases > 0 else 0,
                            'actual_revenue': day_revenue,
                            'discount_savings': round((day_purchases * 20.0 - day_revenue) if day_purchases > 0 else 0, 2)
                        } if day in [3, 4] else None
                    }
            
            # Calculate peak hours for Fried Chicken Shop
            fcs_peak_hours = []
            if fcs_hourly_patterns:
                max_visits = max(v for v in fcs_hourly_patterns.values() if isinstance(v, int))
                if max_visits > 0:
                    fcs_peak_hours = [int(h) for h, v in fcs_hourly_patterns.items() if isinstance(v, int) and v == max_visits]
            
            # Detailed Fried Chicken Shop business intelligence
            fried_chicken_metrics = {
                'simulation_metadata': {
                    'focus': 'Fried Chicken Shop Performance Analysis',
                    'run_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'total_simulated_days': self.current_day,
                    'discount_days': [3, 4],
                    'discount_percentage': 20
                },
                'overall_performance': {
                    'total_visits': fcs_total_visits,
                    'total_purchases': fcs_total_purchases,
                    'total_revenue': round(fcs_total_revenue, 2),
                    'total_discount_usage': fcs_total_discount_usage,
                    'overall_conversion_rate': round((fcs_total_purchases / fcs_total_visits * 100) if fcs_total_visits > 0 else 0, 2),
                    'average_revenue_per_day': round(fcs_total_revenue / self.current_day if self.current_day > 0 else 0, 2),
                    'average_transaction_value': round(fcs_total_revenue / fcs_total_purchases if fcs_total_purchases > 0 else 0, 2),
                    'average_visits_per_day': round(fcs_total_visits / self.current_day if self.current_day > 0 else 0, 2)
                },
                'temporal_analysis': {
                    'hourly_patterns': fcs_hourly_patterns,
                    'peak_hours': fcs_peak_hours,
                    'daily_breakdown': fcs_daily_breakdown
                },
                'discount_impact_analysis': {
                    'regular_days_performance': {
                        'days': [1, 2, 5, 6, 7],
                        'avg_visits': 0,
                        'avg_purchases': 0,
                        'avg_revenue': 0,
                        'avg_conversion_rate': 0
                    },
                    'discount_days_performance': {
                        'days': [3, 4],
                        'avg_visits': 0,
                        'avg_purchases': 0,
                        'avg_revenue': 0,
                        'avg_conversion_rate': 0
                    },
                    'discount_effectiveness': {
                        'visit_increase_percentage': 0,
                        'purchase_increase_percentage': 0,
                        'revenue_impact_percentage': 0,
                        'total_discount_given': 0
                    }
                },
                'business_insights': {
                    'best_performing_day': None,
                    'worst_performing_day': None,
                    'most_profitable_hour': None,
                    'discount_roi': 0,
                    'customer_behavior_notes': []
                }
            }
            
            # Calculate discount impact analysis
            regular_days = [1, 2, 5, 6, 7]
            discount_days = [3, 4]
            
            # Regular days analysis
            regular_visits = sum(fcs_daily_breakdown.get(f'day_{d}', {}).get('visits', 0) for d in regular_days if d <= self.current_day)
            regular_purchases = sum(fcs_daily_breakdown.get(f'day_{d}', {}).get('purchases', 0) for d in regular_days if d <= self.current_day)
            regular_revenue = sum(fcs_daily_breakdown.get(f'day_{d}', {}).get('revenue', 0) for d in regular_days if d <= self.current_day)
            regular_days_count = len([d for d in regular_days if d <= self.current_day])
            
            if regular_days_count > 0:
                fried_chicken_metrics['discount_impact_analysis']['regular_days_performance'].update({
                    'avg_visits': round(regular_visits / regular_days_count, 2),
                    'avg_purchases': round(regular_purchases / regular_days_count, 2),
                    'avg_revenue': round(regular_revenue / regular_days_count, 2),
                    'avg_conversion_rate': round((regular_purchases / regular_visits * 100) if regular_visits > 0 else 0, 2)
                })
            
            # Discount days analysis
            discount_visits = sum(fcs_daily_breakdown.get(f'day_{d}', {}).get('visits', 0) for d in discount_days if d <= self.current_day)
            discount_purchases = sum(fcs_daily_breakdown.get(f'day_{d}', {}).get('purchases', 0) for d in discount_days if d <= self.current_day)
            discount_revenue = sum(fcs_daily_breakdown.get(f'day_{d}', {}).get('revenue', 0) for d in discount_days if d <= self.current_day)
            discount_days_count = len([d for d in discount_days if d <= self.current_day])
            
            if discount_days_count > 0:
                fried_chicken_metrics['discount_impact_analysis']['discount_days_performance'].update({
                    'avg_visits': round(discount_visits / discount_days_count, 2),
                    'avg_purchases': round(discount_purchases / discount_days_count, 2),
                    'avg_revenue': round(discount_revenue / discount_days_count, 2),
                    'avg_conversion_rate': round((discount_purchases / discount_visits * 100) if discount_visits > 0 else 0, 2)
                })
                
                # Calculate discount effectiveness
                if regular_days_count > 0:
                    reg_avg_visits = regular_visits / regular_days_count
                    reg_avg_purchases = regular_purchases / regular_days_count
                    reg_avg_revenue = regular_revenue / regular_days_count
                    
                    disc_avg_visits = discount_visits / discount_days_count
                    disc_avg_purchases = discount_purchases / discount_days_count
                    disc_avg_revenue = discount_revenue / discount_days_count
                    
                    fried_chicken_metrics['discount_impact_analysis']['discount_effectiveness'].update({
                        'visit_increase_percentage': round(((disc_avg_visits - reg_avg_visits) / reg_avg_visits * 100) if reg_avg_visits > 0 else 0, 2),
                        'purchase_increase_percentage': round(((disc_avg_purchases - reg_avg_purchases) / reg_avg_purchases * 100) if reg_avg_purchases > 0 else 0, 2),
                        'revenue_impact_percentage': round(((disc_avg_revenue - reg_avg_revenue) / reg_avg_revenue * 100) if reg_avg_revenue > 0 else 0, 2),
                        'total_discount_given': round(sum(fcs_daily_breakdown.get(f'day_{d}', {}).get('discount_impact', {}).get('discount_savings', 0) for d in discount_days if d <= self.current_day), 2)
                    })
            
            # Business insights
            if fcs_daily_breakdown:
                # Best/worst performing days
                daily_revenues = {day: data['revenue'] for day, data in fcs_daily_breakdown.items()}
                best_day = max(daily_revenues.keys(), key=lambda x: daily_revenues[x]) if daily_revenues else None
                worst_day = min(daily_revenues.keys(), key=lambda x: daily_revenues[x]) if daily_revenues else None
                
                fried_chicken_metrics['business_insights'].update({
                    'best_performing_day': best_day,
                    'worst_performing_day': worst_day,
                    'most_profitable_hour': fcs_peak_hours[0] if fcs_peak_hours else None
                })
            
            # Create directory for the filepath if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(fried_chicken_metrics, f, indent=2)
            print(f"\n🍗 Fried Chicken Shop focused business metrics saved to: {filepath}")
            print(f"📈 Total Revenue: ${fcs_total_revenue:.2f} | Visits: {fcs_total_visits} | Conversion: {fried_chicken_metrics['overall_performance']['overall_conversion_rate']}%")
            
        except Exception as e:
            print(f"Error saving Fried Chicken Shop metrics: {str(e)}")
            traceback.print_exc()

    def save_all_daily_summaries_to_file(self, base_dir_override: Optional[str] = None) -> None:
        """Save all daily summaries (comprehensive insights) to a single JSON file."""
        try:
            if base_dir_override:
                metrics_dir = base_dir_override
            else:
                # Default path construction
                current_script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_script_dir)
                if not project_root.endswith("LLMAgentsTown_Stability"):
                     project_root = os.path.dirname(project_root)

                metrics_dir = os.path.join(project_root, "LLMAgentsTown_memory_records", "simulation_daily_summaries")
            
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Ensure all days are processed for summary data if not already cached
            for day_num in range(1, 8):  # Always process 7 days
                if day_num not in self.daily_summary_cache:
                    # This will generate and cache it
                    self.get_daily_summary_data(day_num)

            # Prepare data for saving
            all_summaries_data = {
                'simulation_start_time': self.simulation_start_time,
                'total_simulated_days': 7,  # Fixed to 7 days
                'daily_insights': self.daily_summary_cache
            }
            
            summary_filename = f"daily_insights_sim_{self.simulation_start_time}.json"
            summary_filepath = os.path.join(metrics_dir, summary_filename)
            
            with open(summary_filepath, 'w') as f:
                json.dump(all_summaries_data, f, indent=2)
            print(f"\nAll daily insight summaries saved to: {summary_filepath}")
            
        except Exception as e:
            print(f"Error saving all daily summaries: {e}")
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

    def get_location_satisfaction_stats(self, location: str, day: Optional[int] = None) -> Dict:
        """Get satisfaction statistics for a location"""
        try:
            if day is None:
                day = self.current_day
                
            if day not in self.daily_metrics_data:
                return {}
                
            ratings = self.daily_metrics_data[day]['satisfaction'][location]['ratings']
            
            if not ratings:
                return {
                    'total_ratings': 0,
                    'average_rating': 0,
                    'recommendation_rate': 0
                }
                
            return {
                'total_ratings': len(ratings),
                'ratings': ratings
            }
            
        except Exception as e:
            print(f"Error getting satisfaction stats: {str(e)}")
            return {}

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics data for saving state.
        
        Returns:
            Dict containing all metrics data including daily metrics, conversation topics,
            relationship trends, and satisfaction ratings.
        """
        return {
            'current_day': self.current_day,
            'daily_metrics_data': self.daily_metrics_data,
            'conversation_topics_by_day': self.conversation_topics_by_day,
            'relationship_trends_by_day': self.relationship_trends_by_day,
            'satisfaction_ratings_by_day': self.satisfaction_ratings_by_day,
            'simulation_start_time': self.simulation_start_time,
            'start_time': self.start_time.strftime('%Y%m%d_%H%M%S')
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics data for the simulation."""
        try:
            current_day = self.current_day
            self.ensure_day_initialized(current_day)
            
            return {
                'day': current_day,
                'store_visits': self.daily_metrics_data[current_day]['store_visits'],
                'purchases': self.daily_metrics_data[current_day]['purchases'],
                'conversations': self.daily_metrics_data[current_day]['conversations'],
                'satisfaction': self.daily_metrics_data[current_day]['satisfaction'],
                'energy': self.daily_metrics_data[current_day]['energy'],
                'travel': self.daily_metrics_data[current_day]['travel']
            }
        except Exception as e:
            print(f"Error getting current metrics: {str(e)}")
            traceback.print_exc()
            return {}

class MetricsManager:
    def __init__(self, simulation_dir: str):
        """Initialize metrics manager with simulation directory."""
        self.simulation_dir = simulation_dir
        self.metrics_dir = os.path.join(simulation_dir, 'simulation_metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics structure
        self.metrics = {
            "simulation_metadata": {
                "focus": "Town Performance Analysis",
                "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_simulated_days": 0,
                "discount_days": [3, 4],  # Example discount days
                "discount_percentage": 20
            },
            "overall_performance": {
                "total_visits": 0,
                "total_purchases": 0,
                "total_revenue": 0.0,
                "total_discount_usage": 0,
                "overall_conversion_rate": 0,
                "average_revenue_per_day": 0,
                "average_transaction_value": 0,
                "average_visits_per_day": 0
            },
            "temporal_analysis": {
                "hourly_patterns": {str(h): 0 for h in range(24)},
                "peak_hours": [],
                "daily_breakdown": {}
            },
            "shop_performance": {
                "fried_chicken_shop": {
                    "total_visits": 0,
                    "total_purchases": 0,
                    "total_revenue": 0.0,
                    "conversion_rate": 0,
                    "average_transaction_value": 0,
                    "popular_items": {},
                    "customer_satisfaction": 0
                },
                "grocery_store": {
                    "total_visits": 0,
                    "total_purchases": 0,
                    "total_revenue": 0.0,
                    "conversion_rate": 0,
                    "average_transaction_value": 0,
                    "popular_items": {},
                    "customer_satisfaction": 0
                },
                "coffee_shop": {
                    "total_visits": 0,
                    "total_purchases": 0,
                    "total_revenue": 0.0,
                    "conversion_rate": 0,
                    "average_transaction_value": 0,
                    "popular_items": {},
                    "customer_satisfaction": 0
                }
            },
            "social_analysis": {
                "total_interactions": 0,
                "interaction_types": {
                    "conversations": 0,
                    "meals_together": 0,
                    "shared_activities": 0
                },
                "relationship_strength": {},
                "popular_topics": {},
                "interaction_locations": {},
                "daily_social_activity": {}
            },
            "agent_behavior": {
                "daily_routines": {},
                "travel_patterns": {},
                "spending_habits": {},
                "social_preferences": {},
                "energy_levels": {},
                "satisfaction_scores": {}
            },
            "business_insights": {
                "best_performing_day": None,
                "worst_performing_day": None,
                "most_profitable_hour": None,
                "discount_roi": 0,
                "customer_behavior_notes": [],
                "social_impact_notes": []
            }
        }
    
    def record_daily_metrics(self, day: int, agents: List['Agent'], memory_mgr: 'MemoryManager') -> None:
        """Record metrics for a specific day."""
        try:
            # Initialize daily breakdown
            self.metrics['temporal_analysis']['daily_breakdown'][str(day)] = {
                "visits": 0,
                "purchases": 0,
                "revenue": 0.0,
                "conversion_rate": 0,
                "social_interactions": 0,
                "average_satisfaction": 0
            }
            
            # Reset hourly patterns for the day
            self.metrics['temporal_analysis']['hourly_patterns'] = {str(h): 0 for h in range(24)}
            
            # Collect metrics for the day
            day_start = (day - 1) * 24
            day_end = day * 24
            
            # Process each agent's activities for the day
            for agent in agents:
                # Get all location events (store visits)
                store_visits = memory_mgr.get_memories(
                    agent.name,
                    memory_type='LOCATION_EVENT',
                    start_time=day_start,
                    end_time=day_end
                )
                
                # Get all purchase events
                purchases = memory_mgr.get_memories(
                    agent.name,
                    memory_type='PURCHASE_EVENT',
                    start_time=day_start,
                    end_time=day_end
                )
                
                # Get all social interactions
                interactions = memory_mgr.get_memories(
                    agent.name,
                    memory_type='INTERACTION_EVENT',
                    start_time=day_start,
                    end_time=day_end
                )
                
                # Get all activities
                activities = memory_mgr.get_memories(
                    agent.name,
                    memory_type='ACTIVITY_EVENT',
                    start_time=day_start,
                    end_time=day_end
                )
                
                # Update shop performance metrics
                for visit in store_visits:
                    location = visit.memory_data.get('location')
                    if location in self.metrics['shop_performance']:
                        shop = self.metrics['shop_performance'][location]
                        shop['total_visits'] += 1
                        self.metrics['temporal_analysis']['daily_breakdown'][str(day)]['visits'] += 1
                        hour = visit.memory_data['time'] % 24
                        self.metrics['temporal_analysis']['hourly_patterns'][str(hour)] += 1
                
                for purchase in purchases:
                    location = purchase.memory_data.get('location')
                    if location in self.metrics['shop_performance']:
                        shop = self.metrics['shop_performance'][location]
                        shop['total_purchases'] += 1
                        shop['total_revenue'] += purchase.memory_data.get('amount', 0)
                        self.metrics['temporal_analysis']['daily_breakdown'][str(day)]['purchases'] += 1
                        self.metrics['temporal_analysis']['daily_breakdown'][str(day)]['revenue'] += purchase.memory_data.get('amount', 0)
                        
                        # Track popular items
                        items = purchase.memory_data.get('items', [])
                        for item in items:
                            shop['popular_items'][item] = shop['popular_items'].get(item, 0) + 1
                
                # Update social analysis metrics
                for interaction in interactions:
                    self.metrics['social_analysis']['total_interactions'] += 1
                    self.metrics['temporal_analysis']['daily_breakdown'][str(day)]['social_interactions'] += 1
                    
                    # Track interaction types
                    interaction_type = interaction.memory_data.get('interaction_type', 'unknown')
                    self.metrics['social_analysis']['interaction_types'][interaction_type] = (
                        self.metrics['social_analysis']['interaction_types'].get(interaction_type, 0) + 1
                    )
                    
                    # Track interaction locations
                    location = interaction.memory_data.get('location', 'unknown')
                    self.metrics['social_analysis']['interaction_locations'][location] = (
                        self.metrics['social_analysis']['interaction_locations'].get(location, 0) + 1
                    )
                    
                    # Track relationship strength
                    participants = interaction.memory_data.get('participants', [])
                    for participant in participants:
                        if participant != agent.name:
                            key = f"{agent.name}-{participant}"
                            self.metrics['social_analysis']['relationship_strength'][key] = (
                                self.metrics['social_analysis']['relationship_strength'].get(key, 0) + 1
                            )
                
                # Update agent behavior metrics
                self.metrics['agent_behavior']['daily_routines'][agent.name] = {
                    'locations_visited': [],
                    'activities_performed': [],
                    'interactions_had': []
                }
                
                for activity in activities:
                    activity_type = activity.memory_data.get('activity_type', 'unknown')
                    self.metrics['agent_behavior']['daily_routines'][agent.name]['activities_performed'].append(activity_type)
                    
                    # Track energy levels
                    energy_level = activity.memory_data.get('energy_level', 0)
                    self.metrics['agent_behavior']['energy_levels'][agent.name] = energy_level
                    
                    # Track satisfaction
                    satisfaction = activity.memory_data.get('satisfaction', 0)
                    self.metrics['agent_behavior']['satisfaction_scores'][agent.name] = satisfaction
            
            # Calculate conversion rates and averages
            for shop_name, shop_data in self.metrics['shop_performance'].items():
                if shop_data['total_visits'] > 0:
                    shop_data['conversion_rate'] = (
                        shop_data['total_purchases'] / shop_data['total_visits']
                    )
                if shop_data['total_purchases'] > 0:
                    shop_data['average_transaction_value'] = (
                        shop_data['total_revenue'] / shop_data['total_purchases']
                    )
            
            # Calculate overall averages
            if day > 0:
                self.metrics['overall_performance']['average_revenue_per_day'] = (
                    self.metrics['overall_performance']['total_revenue'] / day
                )
                self.metrics['overall_performance']['average_visits_per_day'] = (
                    self.metrics['overall_performance']['total_visits'] / day
                )
            
            # Find peak hours
            hourly_patterns = self.metrics['temporal_analysis']['hourly_patterns']
            max_visits = max(hourly_patterns.values())
            self.metrics['temporal_analysis']['peak_hours'] = [
                int(hour) for hour, visits in hourly_patterns.items()
                if visits == max_visits
            ]
            
            # Update business insights
            self._update_business_insights(day)
            
            # Save daily metrics
            self._save_metrics(day)
            
        except Exception as e:
            print(f"Error recording daily metrics: {str(e)}")
            traceback.print_exc()
    
    def _update_business_insights(self, day: int) -> None:
        """Update business insights based on current metrics."""
        try:
            daily_breakdown = self.metrics['temporal_analysis']['daily_breakdown']
            
            # Find best and worst performing days
            if daily_breakdown:
                best_day = max(daily_breakdown.items(), key=lambda x: x[1]['revenue'])
                worst_day = min(daily_breakdown.items(), key=lambda x: x[1]['revenue'])
                
                self.metrics['business_insights']['best_performing_day'] = {
                    'day': int(best_day[0]),
                    'revenue': best_day[1]['revenue'],
                    'visits': best_day[1]['visits'],
                    'conversion_rate': best_day[1]['conversion_rate'],
                    'social_interactions': best_day[1]['social_interactions']
                }
                
                self.metrics['business_insights']['worst_performing_day'] = {
                    'day': int(worst_day[0]),
                    'revenue': worst_day[1]['revenue'],
                    'visits': worst_day[1]['visits'],
                    'conversion_rate': worst_day[1]['conversion_rate'],
                    'social_interactions': worst_day[1]['social_interactions']
                }
            
            # Find most profitable hour
            hourly_patterns = self.metrics['temporal_analysis']['hourly_patterns']
            if hourly_patterns:
                most_profitable_hour = max(hourly_patterns.items(), key=lambda x: x[1])
                self.metrics['business_insights']['most_profitable_hour'] = int(most_profitable_hour[0])
            
            # Calculate discount ROI
            total_discount_given = self.metrics['discount_impact_analysis']['discount_effectiveness']['total_discount_given']
            if total_discount_given > 0:
                self.metrics['business_insights']['discount_roi'] = (
                    self.metrics['overall_performance']['total_revenue'] / total_discount_given
                )
            
            # Generate customer behavior notes
            self._generate_customer_behavior_notes()
            
            # Generate social impact notes
            self._generate_social_impact_notes()
            
        except Exception as e:
            print(f"Error updating business insights: {str(e)}")
            traceback.print_exc()
    
    def _generate_customer_behavior_notes(self) -> None:
        """Generate insights about customer behavior."""
        try:
            notes = []
            
            # Analyze shop performance
            for shop_name, shop_data in self.metrics['shop_performance'].items():
                if shop_data['total_visits'] > 0:
                    notes.append(f"{shop_name}: {shop_data['total_visits']} visits, "
                               f"{shop_data['conversion_rate']:.2%} conversion rate")
                    
                    # Popular items
                    if shop_data['popular_items']:
                        top_items = sorted(shop_data['popular_items'].items(), 
                                        key=lambda x: x[1], reverse=True)[:3]
                        notes.append(f"Top items at {shop_name}: {', '.join(item for item, _ in top_items)}")
            
            # Analyze spending patterns
            total_revenue = sum(shop['total_revenue'] for shop in self.metrics['shop_performance'].values())
            if total_revenue > 0:
                for shop_name, shop_data in self.metrics['shop_performance'].items():
                    revenue_share = shop_data['total_revenue'] / total_revenue
                    notes.append(f"{shop_name} accounts for {revenue_share:.1%} of total revenue")
            
            self.metrics['business_insights']['customer_behavior_notes'] = notes
            
        except Exception as e:
            print(f"Error generating customer behavior notes: {str(e)}")
            traceback.print_exc()
    
    def _generate_social_impact_notes(self) -> None:
        """Generate insights about social interactions and their impact."""
        try:
            notes = []
            
            # Analyze interaction patterns
            total_interactions = self.metrics['social_analysis']['total_interactions']
            if total_interactions > 0:
                for interaction_type, count in self.metrics['social_analysis']['interaction_types'].items():
                    percentage = count / total_interactions
                    notes.append(f"{interaction_type}: {percentage:.1%} of all interactions")
                
                # Popular locations
                if self.metrics['social_analysis']['interaction_locations']:
                    top_locations = sorted(
                        self.metrics['social_analysis']['interaction_locations'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    notes.append(f"Most social locations: {', '.join(loc for loc, _ in top_locations)}")
            
            # Analyze relationship strength
            if self.metrics['social_analysis']['relationship_strength']:
                strongest_relationships = sorted(
                    self.metrics['social_analysis']['relationship_strength'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                notes.append("Strongest relationships: " + 
                           ", ".join(f"{pair} ({count} interactions)" 
                                   for pair, count in strongest_relationships))
            
            self.metrics['business_insights']['social_impact_notes'] = notes
            
        except Exception as e:
            print(f"Error generating social impact notes: {str(e)}")
            traceback.print_exc()
    
    def _save_metrics(self, day: int) -> None:
        """Save metrics to file."""
        try:
            metrics_file = os.path.join(self.metrics_dir, f'metrics_day_{day}.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            print(f"\n=== Daily Metrics for Day {day} ===")
            print(json.dumps(self.metrics, indent=2))
            
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
            traceback.print_exc()