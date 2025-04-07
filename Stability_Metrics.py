from datetime import datetime
import os
import json

class FriedChickenMetrics:
    def __init__(self, experiment_type, discount_value):
        self.daily_metrics = {}
        self.current_day = 1
        self.start_time = datetime.now()
        self.experiment_type = experiment_type
        self.discount_value = discount_value
        
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
                'positive': [],  # List of {speaker, listener, location, time, content}
                'neutral': [],
                'negative': [],
                'influencers': {},  # Track who spreads the word most effectively
                'spread_chains': []  # Track how information spreads (A told B who told C...)
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
            }
        }

    def record_interaction(self, agent_name, location, event_type, details=None):
        """Record interactions with enhanced tracking"""
        current_time = datetime.now()
        metrics = self.daily_metrics[self.current_day]
        
        if event_type == "store_visit":
            # Record visit with timing
            visit_hour = current_time.hour
            metrics['store_visits']['total_count'] += 1
            metrics['store_visits']['peak_hours'][visit_hour] = metrics['store_visits']['peak_hours'].get(visit_hour, 0) + 1
            
            # Track if this is a repeat visit
            is_repeat = any(v['agent'] == agent_name for v in metrics['store_visits']['visitors'])
            metrics['store_visits']['visitors'].append({
                'agent': agent_name,
                'time': current_time.strftime('%H:%M'),
                'repeat_visit': is_repeat
            })
            
        elif event_type == "purchase":
            # Calculate pricing
            original_price = 20.0
            used_discount = self.is_discount_day()
            
            if used_discount:
                if self.experiment_type == "percentage":
                    discount_amount = original_price * (self.discount_value / 100)
                else:  # fixed amount
                    discount_amount = self.discount_value
                final_price = original_price - discount_amount
                
                # Record discount usage
                metrics['discount_usage']['total_count'] += 1
                metrics['discount_usage']['total_savings'] += discount_amount
                metrics['discount_usage']['transactions'].append({
                    'customer': agent_name,
                    'original_price': original_price,
                    'discount_amount': discount_amount,
                    'final_price': final_price
                })
            else:
                final_price = original_price
                discount_amount = 0
            
            # Record sale
            metrics['sales']['total_amount'] += final_price
            metrics['sales']['meals_sold'] += 1
            metrics['sales']['transactions'].append({
                'customer': agent_name,
                'amount': final_price,
                'time': current_time.strftime('%H:%M'),
                'used_discount': used_discount
            })
            
            # Update customer segments
            if agent_name not in metrics['customer_segments']['new_customers'] and \
               agent_name not in metrics['customer_segments']['returning_customers']:
                metrics['customer_segments']['new_customers'].append(agent_name)
            elif agent_name in metrics['customer_segments']['new_customers']:
                metrics['customer_segments']['new_customers'].remove(agent_name)
                metrics['customer_segments']['returning_customers'].append(agent_name)
            
            # Update repeat customer tracking
            metrics['sales']['repeat_customers'][agent_name] = \
                metrics['sales']['repeat_customers'].get(agent_name, 0) + 1
                
        elif event_type == "word_of_mouth":
            if not details or 'sentiment' not in details or 'listener' not in details:
                return
                
            sentiment = details['sentiment']
            listener = details['listener']
            content = details.get('content', '')
            
            if sentiment in ['positive', 'neutral', 'negative']:
                metrics['word_of_mouth']['total_mentions'] += 1
                mention_data = {
                    'speaker': agent_name,
                    'listener': listener,
                    'location': location,
                    'time': current_time.strftime('%H:%M'),
                    'content': content
                }
                metrics['word_of_mouth'][sentiment].append(mention_data)
                
                # Track influencer impact
                metrics['word_of_mouth']['influencers'][agent_name] = \
                    metrics['word_of_mouth']['influencers'].get(agent_name, 0) + 1
                    
                # Track information spread
                metrics['social_network']['information_flow'].append({
                    'from': agent_name,
                    'to': listener,
                    'time': current_time.strftime('%H:%M'),
                    'location': location,
                    'sentiment': sentiment
                })
                
                # Update community impact
                metrics['social_network']['community_impact'][location] = \
                    metrics['social_network']['community_impact'].get(location, 0) + 1
                    
                # If the listener later visits/purchases, they'll be marked as influenced
                if listener not in metrics['customer_segments']['influenced_customers']:
                    metrics['customer_segments']['influenced_customers'].append(listener)
        
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
        """Print enhanced daily summary"""
        if day is None:
            day = self.current_day
        
        metrics = self.daily_metrics[day]
        print(f"\n=== Day {day} Fried Chicken Shop Activity ===")
        print(f"Experiment Type: {self.experiment_type}")
        print(f"Discount Value: {self.discount_value}")
        
        # Visit Statistics
        print("\nVisit Statistics:")
        print(f"- Total Visits: {metrics['store_visits']['total_count']}")
        print(f"- Unique Visitors: {len(metrics['store_visits']['visitors'])}")
        print("\nPeak Hours:")
        peak_hours = sorted(metrics['store_visits']['peak_hours'].items(), key=lambda x: x[1], reverse=True)
        for hour, count in peak_hours[:3]:
            print(f"- {hour}:00 - {count} visits")
        
        # Sales Statistics
        print("\nSales Statistics:")
        print(f"- Total Revenue: ${metrics['sales']['total_amount']:.2f}")
        print(f"- Meals Sold: {metrics['sales']['meals_sold']}")
        print(f"- Average Transaction: ${metrics['sales']['average_transaction']:.2f}")
        print(f"- Repeat Customers: {len(metrics['sales']['repeat_customers'])}")
        
        # Discount Performance
        if self.is_discount_day():
            print("\nDiscount Performance:")
            print(f"- Usage Count: {metrics['discount_usage']['total_count']}")
            print(f"- Total Savings: ${metrics['discount_usage']['total_savings']:.2f}")
            print(f"- Conversion Rate: {metrics['discount_usage']['conversion_rate']:.1f}%")
        
        # Word of Mouth Impact
        print("\nWord of Mouth Impact:")
        print(f"- Total Mentions: {metrics['word_of_mouth']['total_mentions']}")
        for sentiment in ['positive', 'neutral', 'negative']:
            print(f"- {sentiment.title()} Mentions: {len(metrics['word_of_mouth'][sentiment])}")
        
        # Top Influencers
        print("\nTop Influencers:")
        top_influencers = self.get_key_influencers()
        for influencer, count in top_influencers:
            print(f"- {influencer}: {count} mentions")
        
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
        """Initialize a new day with the enhanced metrics structure"""
        self.current_day += 1
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
                'spread_chains': []
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
            'visit_history': {}  # Track visit counts per agent
        }
        
        # Print summary of the previous day
        self.print_daily_summary(self.current_day - 1)

    def is_discount_day(self):
        """Check if current day is a discount day (2 days per week)"""
        return self.current_day % 7 in [3, 5]  # Discount on Wednesday and Friday

    def save_metrics(self):
        """Save metrics in the memory_records directory"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        records_dir = os.path.join(base_dir, 'memory_records')
        os.makedirs(records_dir, exist_ok=True)

        filename = f"fried_chicken_metrics_{self.experiment_type}_{self.discount_value}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(records_dir, filename)

        metrics_data = {
            'simulation_start': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'simulation_end': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_days': self.current_day,
            'experiment_type': self.experiment_type,
            'discount_value': self.discount_value,
            'daily_metrics': self.daily_metrics
        }

        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Fried chicken metrics saved to {filepath}")
        return filepath 