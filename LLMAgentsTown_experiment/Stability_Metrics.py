from datetime import datetime
import os
import json

# Define sales-related keywords for tracking
SALES_KEYWORDS = [
    "discount",
    "% off",
    "dollars off",
    "sale",
    "promotion",
    "deal",
    "special offer",
    "cheaper",
    "savings",
    "reduced price",
    "original price",
    "regular price",
    "save money"
]

class FriedChickenMetrics:
    def __init__(self, settings=None):
        """
        Initialize metrics with settings from experiment_settings.json
        
        Args:
            settings: Dict containing experiment settings
        """
        self.daily_metrics = {}
        self.current_day = 1
        self.start_time = datetime.now()
        
        # Use default settings if none provided
        if settings is None:
            settings = {
                'type': 'percentage',
                'value': 20,
                'days': [3, 4]
            }
        
        # Get discount settings using the correct keys
        self.discount_type = settings.get('type', 'percentage')
        self.discount_value = settings.get('value', 20)
        self.discount_days = settings.get('days', [3, 4])
        
        # Initialize first day
        self.daily_metrics[self.current_day] = {
            'store_visits': {
                'total_count': 0,
                'visitors': [],  # List of {agent, time, repeat_visit}
                'peak_hours': {}  # Track busy hours
            },
            'sales': {
                'total_amount': 0.0,
                'transactions': [],  # List of {customer, amount, time, used_discount}
                'meals_sold': 0,
                'average_transaction': 0.0,
                'repeat_customers': {}  # Track customer loyalty
            },
            'discount_usage': {
                'total_count': 0,
                'total_savings': 0.0,
                'transactions': [],  # List of {customer, original_price, discount_amount, final_price}
                'conversion_rate': 0.0  # % of visitors who used discount
            },
            'word_of_mouth': {
                'total_mentions': 0,
                'positive': [],  # Happy about discounts/savings
                'neutral': [],   # Just sharing discount info
                'negative': [],  # Complaints about discount terms
                'influencers': {},
                'spread_chains': [],
                'sale_mentions': {
                    'discount_awareness': [],  # People who know about discounts
                    'discount_sharing': [],    # People who told others
                    'price_discussion': []     # Price comparison discussions
                }
            },
            'customer_segments': {
                'new_customers': [],
                'returning_customers': [],
                'influenced_customers': []  # Customers who came after hearing about it
            },
            'social_network': {
                'information_flow': [],  # Track how discount info spreads
                'key_influencers': {},   # Track most effective message spreaders
                'community_impact': {}    # Track impact by community/location
            },
            'other_purchases': []  # Track other purchases
        }
        
        self.all_time_customers = set()  # Track all customers ever
        self.daily_customers = set()     # Reset each day
        self.total_revenue = 0.0
        self.total_meals_sold = 0
        self.total_visits = 0
        self.unique_visitors = set()

    def record_interaction(self, agent_name, location, event_type, details=None):
        """Record interactions with proper tracking"""
        if details is None:
            details = {}
            
        metrics = self.daily_metrics[self.current_day]
        
        if event_type == "store_visit":
            # Update visit counts
            metrics['store_visits']['total_count'] += 1
            self.total_visits += 1  # Update global counter
            
            hour = int(details.get('time', 0)) % 24
            metrics['store_visits']['peak_hours'][hour] = metrics['store_visits']['peak_hours'].get(hour, 0) + 1
            
            # Track unique visitors
            if agent_name not in [v['agent'] for v in metrics['store_visits']['visitors']]:
                metrics['store_visits']['visitors'].append({
                    'agent': agent_name,
                    'time': details.get('time'),
                    'repeat_visit': agent_name in self.all_time_customers
                })
                self.unique_visitors.add(agent_name)
            
        elif event_type == "purchase" and location == "Fried Chicken Shop":
            # Update revenue and sales counts
            price = details.get('price', 20.0)  # Default price if not specified
            metrics['sales']['total_amount'] += price
            metrics['sales']['meals_sold'] += 1
            self.total_revenue += price  # Update global revenue
            
            # Record transaction
            metrics['sales']['transactions'].append({
                'customer': agent_name,
                'amount': price,
                'time': details.get('time'),
                'used_discount': details.get('used_discount', False),
                'influenced_by': details.get('influenced_by')
            })
            
            # Update customer segmentation
            if agent_name not in self.all_time_customers:
                self.all_time_customers.add(agent_name)
                metrics['customer_segments']['new_customers'].append(agent_name)
            else:
                metrics['customer_segments']['returning_customers'].append(agent_name)

            # Update discount usage if applicable
            if details.get('used_discount'):
                metrics['discount_usage']['total_count'] += 1
                metrics['discount_usage']['total_savings'] += details.get('discount_amount', 0)
                metrics['discount_usage']['transactions'].append({
                    'customer': agent_name,
                    'original_price': details.get('original_price', price),
                    'discount_amount': details.get('discount_amount', 0),
                    'final_price': price
                })
        
        # Update conversion rates
        if metrics['store_visits']['total_count'] > 0:
            metrics['discount_usage']['conversion_rate'] = \
                (metrics['discount_usage']['total_count'] / metrics['store_visits']['total_count']) * 100
                
        # Update average transaction
        if metrics['sales']['meals_sold'] > 0:
            metrics['sales']['average_transaction'] = \
                metrics['sales']['total_amount'] / metrics['sales']['meals_sold']

    def get_key_influencers(self):
        """Identify key influencers based on their impact"""
        metrics = self.daily_metrics[self.current_day]
        influencers = metrics['word_of_mouth']['influencers']
        
        # Sort influencers by number of mentions
        sorted_influencers = sorted(
            influencers.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_influencers[:5]  # Return top 5 influencers

    def get_information_spread_analysis(self):
        """Analyze how information spreads through the social network"""
        metrics = self.daily_metrics[self.current_day]
        flow = metrics['social_network']['information_flow']
        
        # Create chains of information spread
        spread_chains = []
        for interaction in flow:
            # Find existing chains this interaction could connect to
            for chain in spread_chains:
                if chain[-1]['to'] == interaction['from']:
                    chain.append(interaction)
                    break
            else:
                # Start new chain
                spread_chains.append([interaction])
        
        return spread_chains

    def print_daily_summary(self, day=None):
        """Print daily summary with accurate metrics"""
        if day is None:
            day = self.current_day
            
        metrics = self.daily_metrics[day]
        
        print(f"\n=== Day {day} Fried Chicken Shop Activity ===")
        print(f"Discount Type: {self.discount_type}")
        print(f"Discount Value: {self.discount_value}")
        
        # Visit Statistics
        print("\nVisit Statistics:")
        print(f"- Total Visits: {metrics['store_visits']['total_count']}")
        print(f"- Unique Visitors: {len(set(v['agent'] for v in metrics['store_visits']['visitors']))}")
        
        # Peak Hours
        print("\nPeak Hours:")
        peak_hours = sorted(metrics['store_visits']['peak_hours'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        for hour, count in peak_hours:
            if count > 0:  # Only show hours with visits
                print(f"- {hour}:00 - {count} visits")
        
        # Sales Statistics
        print("\nSales Statistics:")
        print(f"- Total Revenue: ${metrics['sales']['total_amount']:.2f}")
        print(f"- Meals Sold: {metrics['sales']['meals_sold']}")
        if metrics['sales']['meals_sold'] > 0:
            avg_transaction = metrics['sales']['total_amount'] / metrics['sales']['meals_sold']
            print(f"- Average Transaction: ${avg_transaction:.2f}")
        print(f"- Repeat Customers: {len(metrics['customer_segments']['returning_customers'])}")
        
        # Word of Mouth Impact
        print("\nWord of Mouth Impact:")
        print(f"- Total Mentions: {metrics['word_of_mouth']['total_mentions']}")
        print(f"- Positive Mentions: {len(metrics['word_of_mouth']['positive'])}")
        print(f"- Neutral Mentions: {len(metrics['word_of_mouth']['neutral'])}")
        print(f"- Negative Mentions: {len(metrics['word_of_mouth']['negative'])}")
        
        # Customer Segments
        print("\nCustomer Segments:")
        print(f"- New Customers: {len(metrics['customer_segments']['new_customers'])}")
        print(f"- Returning Customers: {len(metrics['customer_segments']['returning_customers'])}")
        print(f"- Influenced Customers: {len(metrics['customer_segments']['influenced_customers'])}")
        
        # Community Impact
        print("\nCommunity Impact:")
        for location, count in metrics['social_network']['community_impact'].items():
            print(f"- {location}: {count} interactions")

    def new_day(self):
        """Reset daily metrics while preserving cumulative data"""
        self.current_day += 1
        
        # Print previous day summary
        if self.current_day > 1:
            self.print_daily_summary(self.current_day - 1)
        
        # Reset daily tracking while keeping cumulative data
        self.daily_customers = set()
        self.daily_metrics[self.current_day] = {
            'store_visits': {
                'total_count': 0,
                'visitors': [],
                'peak_hours': {}
            },
            'sales': {
                'total_amount': 0.0,
                'transactions': [],
                'meals_sold': 0,
                'average_transaction': 0.0,
                'repeat_customers': {}
            },
            'discount_usage': {
                'total_count': 0,
                'total_savings': 0.0,
                'transactions': [],
                'conversion_rate': 0.0
            },
            'word_of_mouth': {
                'total_mentions': 0,
                'positive': [],
                'neutral': [],
                'negative': [],
                'influencers': {},
                'spread_chains': [],
                'sale_mentions': {
                    'discount_awareness': [],
                    'discount_sharing': [],
                    'price_discussion': []
                }
            },
            'customer_segments': {
                'new_customers': [],
                'returning_customers': [],
                'influenced_customers': []
            },
            'social_network': {
                'information_flow': [],
                'key_influencers': {},
                'community_impact': {}
            },
            'visit_history': {},  # Track visit counts per agent
            'other_purchases': []  # Track other purchases
        }
        self.total_visits = 0
        self.unique_visitors = set()

    def is_discount_day(self):
        """Check if current day is a discount day based on settings"""
        # Convert current_day to day name
        day_index = (self.current_day - 1) % 7
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        current_day_name = day_names[day_index]
        
        # Check against configured discount days
        return current_day_name in self.discount_days

    def calculate_discount(self, original_price):
        """Calculate discount amount based on type and value"""
        if self.discount_type == "percentage":
            return original_price * (self.discount_value / 100)
        else:  # fixed amount
            return self.discount_value

    def save_metrics(self):
        """Save metrics to file"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        records_dir = os.path.join(base_dir, 'LLMAgentsTown_memory_records', 'simulation_metrics')
        os.makedirs(records_dir, exist_ok=True)
        
        filename = f"metrics_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(records_dir, filename)
        
        metrics_data = {
            'simulation_start': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'simulation_end': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'discount_type': self.discount_type,
            'discount_value': self.discount_value,
            'discount_days': self.discount_days,
            'daily_metrics': self.daily_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"\nMetrics saved to: {filepath}")
        return filepath

    def get_daily_summary(self, day):
        """Generate a formatted summary for a specific day"""
        if day not in self.daily_metrics:
            return f"No data available for day {day}"
        
        day_data = self.daily_metrics[day]
        summary = []
        
        summary.append(f"=== Day {day} Metrics ===")
        summary.append(f"Total Visits: {day_data.get('total_visits', 0)}")
        summary.append(f"Total Revenue: ${day_data.get('revenue', 0):.2f}")
        
        if 'discount_sales' in day_data:
            summary.append(f"Discount Sales: {day_data['discount_sales']}")
        
        if 'total_discount_amount' in day_data:
            summary.append(f"Total Discount Amount: ${day_data['total_discount_amount']:.2f}")
        
        summary.append("\nVisitor Breakdown:")
        for visitor_type, count in day_data.get('visitor_types', {}).items():
            summary.append(f"- {visitor_type}: {count}")
        
        summary.append("\nTop Customers:")
        for i, (customer, visits) in enumerate(day_data.get('top_customers', {}).items(), 1):
            if i <= 5:  # Top 5
                summary.append(f"- {customer}: {visits} visits")
        
        summary.append("\nVisit Statistics:")
        summary.append(f"- Total Visits: {day_data['store_visits']['total_count']}")
        summary.append(f"- Unique Visitors: {len(day_data['store_visits']['visitors'])}")
        
        return "\n".join(summary)

    def get_daily_summary_str(self, day):
        """Get a formatted string with the daily summary"""
        if day not in self.daily_metrics:
            return "No data available for this day."
        
        data = self.daily_metrics[day]
        
        summary = []
        summary.append(f"=== Day {day} Metrics ===")
        summary.append(f"Total Visits: {data.get('total_visits', 0)}")
        summary.append(f"Total Revenue: ${data.get('revenue', 0):.2f}")
        
        if 'discount_sales' in data:
            summary.append(f"Discount Sales: {data.get('discount_sales', 0)}")
        
        if 'word_of_mouth' in data:
            summary.append(f"Word of Mouth Mentions: {data.get('word_of_mouth', 0)}")
        
        if 'visitor_types' in data:
            summary.append("\nVisitor Types:")
            for visitor_type, count in data['visitor_types'].items():
                summary.append(f"- {visitor_type}: {count}")
        
        return "\n".join(summary)

    def analyze_social_impact(self):
        """Analyze how social interactions affect sales patterns"""
        metrics = self.daily_metrics[self.current_day]
        
        # Track purchase clusters (purchases happening close together)
        purchase_clusters = []
        current_cluster = []
        
        for transaction in sorted(metrics['sales']['transactions'], key=lambda x: x['time']):
            if current_cluster and (datetime.strptime(transaction['time'], '%H:%M') - 
                                  datetime.strptime(current_cluster[-1]['time'], '%H:%M')).minutes > 30:
                purchase_clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(transaction)
        
        if current_cluster:
            purchase_clusters.append(current_cluster)
        
        # Analyze social patterns
        social_impact_data = {
            'group_purchases': len([c for c in purchase_clusters if len(c) > 1]),
            'influenced_purchases': len([t for t in metrics['sales']['transactions'] 
                                       if t.get('influenced_by')]),
            'recommendation_conversions': len([t for t in metrics['sales']['transactions']
                                             if any(r in str(t.get('influenced_by', '')) 
                                                   for r in ['recommended', 'positive review'])]),
            'average_group_size': sum(len(c) for c in purchase_clusters) / len(purchase_clusters) 
                                 if purchase_clusters else 0
        }
        
        return social_impact_data 

    def update_customer_segments(self, agent_name, was_influenced=False):
        if agent_name not in self.all_time_customers:
            self.new_customers += 1
            self.all_time_customers.add(agent_name)
        else:
            self.returning_customers += 1
        
        if was_influenced:
            self.influenced_customers += 1

    def record_visit(self, agent_name):
        """Record a visit to the shop"""
        self.total_visits += 1
        self.unique_visitors.add(agent_name)

    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.daily_metrics[self.current_day] = {
            'store_visits': {
                'total_count': 0,
                'visitors': [],
                'peak_hours': {}
            },
            'sales': {
                'total_amount': 0.0,
                'transactions': [],
                'meals_sold': 0,
                'average_transaction': 0.0,
                'repeat_customers': {}
            },
            'discount_usage': {
                'total_count': 0,
                'total_savings': 0.0,
                'transactions': [],
                'conversion_rate': 0.0
            },
            'word_of_mouth': {
                'total_mentions': 0,
                'positive': [],
                'neutral': [],
                'negative': [],
                'influencers': {},
                'spread_chains': [],
                'sale_mentions': {
                    'discount_awareness': [],
                    'discount_sharing': [],
                    'price_discussion': []
                }
            },
            'customer_segments': {
                'new_customers': [],
                'returning_customers': [],
                'influenced_customers': []
            },
            'social_network': {
                'information_flow': [],
                'key_influencers': {},
                'community_impact': {}
            },
            'visit_history': {},  # Track visit counts per agent
            'other_purchases': []  # Track other purchases
        }
        self.total_visits = 0
        self.unique_visitors = set()