                
            elif hour == 7:  # Preparation time
                if is_parent:
                    return self.prepare_children_for_school(current_time)
                elif is_young_child or is_supervised_teen:
                    parent = next((agent for agent in family_members_present 
                                 if hasattr(agent, 'family_role') and 
                                 agent.family_role == 'parent'), None)
                    if parent:
                        # Record the preparation interaction
                        self.memory_manager.add_memory(
                            agent_name=self.name,
                            memory_text=f"Getting ready for school with {parent.name}'s help",
                            memory_type="morning_preparation",
                            timestamp=current_time,
                            importance=0.5
                        )
                        return f"Getting ready for school with {parent.name}'s help"
                    return "Waiting for parent's help to get ready"
                elif is_independent_teen:
                    return self.prepare_for_school(hour, all_agents)
                elif hasattr(self, 'workplaces') and self.workplaces:
                    return "Preparing for work"
                return "Starting morning routine"
                    
            elif hour == 8:  # Departure time
                if is_young_child or is_supervised_teen:
                    parent = next((agent for agent in family_members_present 
                                 if hasattr(agent, 'family_role') and 
                                 agent.family_role == 'parent'), None)
                    if parent and hasattr(self, 'school_location'):
                        self.location = self.school_location
                        # Record the school trip
                        self.memory_manager.add_memory(
                            agent_name=self.name,
                            memory_text=f"Went to school with {parent.name}",
                            memory_type="transportation",
                            timestamp=current_time,
                            importance=0.5
                        )
                        return f"Going to school with {parent.name}"
                    return "Waiting for parent to go to school"
                elif is_independent_teen or (hasattr(self, 'is_student') and self.is_student):
                    self.location = self.school_location
                    return "Going to school independently"
                elif hasattr(self, 'workplaces') and self.workplaces:
                    self.location = self.workplaces[0]
                    return "Going to work"
                
                return None

        except Exception as e:
            print(f"Error in morning routine for {self.name}: {str(e)}")
            return "Continuing with default morning activities"

    def find_parent(self, all_agents):
        """Helper method to find parent of a child"""
        try:
            if hasattr(self, 'family_unit') and self.family_unit:
                for agent in all_agents:
                    if (hasattr(agent, 'family_unit') and 
                        agent.family_unit == self.family_unit and 
                        hasattr(agent, 'family_role') and 
                        agent.family_role == 'parent'):
                        return agent
            return None
        except Exception as e:
            print(f"Error finding parent for {self.name}: {str(e)}")
            return None

    def build_action_context(self, current_location, hour, nearby_agents):
        """Build context for action decision"""
        context = []
        
        # Location context
        if current_location == self.residence:
            context.append("This is your residence where you can rest.")
        
        # School context
        if self.is_student and hour in range(8, 15) and current_location == self.school_location:
            context.append("You are at school during school hours.")
        
        # Fried Chicken Shop context
        if self.location == "Fried Chicken Shop":
            current_day = (hour // 24) + 1
            location = self.locations.get("Fried Chicken Shop")
            
            if location and hasattr(location, 'discount') and current_day in location.discount['days']:
                discount_value = location.discount['value']
                context.append(f"There is currently a {discount_value}% discount on all meals!")
            
            if hasattr(self, 'location_last_purchase') and self.location in self.location_last_purchase:
                time_since_last_purchase = hour - self.location_last_purchase[self.location]
                if time_since_last_purchase < 4:
                    context.append(f"You've purchased at {self.location} recently and might want to wait.")
            else:
                context.append("You can purchase a meal if you'd like.")
        
        # Social context
        if nearby_agents:
            context.append(f"There are {len(nearby_agents)} other people here.")
        
        return context

    def generate_contextual_action(self, context, current_time):
        """Generate action based on context"""
        hour = int(current_time) % 24
        current_plan = self.plans[-1] if hasattr(self, 'plans') and self.plans else "No plan yet"
        
        prompt = f"You are {self.name} at {self.location}. Current time is {hour}:00. "
        prompt += f"Your current plan: {current_plan}. "
        if context:
            prompt += f"Context: {' '.join(context)}. "
        prompt += "What would you like to do?"
        
        action = generate(prompt)
        return action if action else "Stay at current location"

    def process_location_specific_actions(self, action, current_time):
        """Process location-specific actions"""
        # Handle Fried Chicken Shop actions
        if any(phrase in action.lower() for phrase in ["go to the fried chicken shop", "visit fried chicken shop", "get fried chicken"]):
            if self.location != "Fried Chicken Shop":
                self.location = "Fried Chicken Shop"
                return f"Moved to Fried Chicken Shop"
        
        # Handle purchase actions
        if "buy" in action.lower() and self.location == "Fried Chicken Shop":
            return self.buy_food(current_time)
        
        return None

    def handle_morning_family_planning(self, current_time):
        """Handle early morning family planning (5:00 AM)"""
        try:
            if not hasattr(self, 'family_unit') or not self.family_unit:
                return "No family to plan with"
                
            # Generate the family schedule for the day
            schedule = self.generate_family_daily_schedule(current_time)
            
            # Create a meaningful family conversation about the day ahead
            context = {
                'time': current_time,
                'schedule': schedule,
                'location': self.residence
            }
            conversation = self.generate_family_conversation(context)
            
            # Broadcast the schedule to family members
            self.broadcast_family_schedule(schedule, current_time, conversation)
            
            return f"Shared today's family schedule and had morning conversation"
            
        except Exception as e:
            print(f"Error in morning family planning for {self.name}: {str(e)}")
            return "Created basic family schedule"

    def handle_family_breakfast(self, current_time, all_agents):
        """Handle family breakfast and schedule coordination (6:00 AM)"""
        try:
            if not hasattr(self, 'family_unit') or not self.family_unit:
                return self.handle_breakfast(current_time)
                
            # Find family members present
            family_members_present = [
                agent for agent in all_agents 
                if hasattr(agent, 'family_unit') and 
                agent.family_unit == self.family_unit and
                agent.location == self.location
            ]
            
            # Check groceries for breakfast
            if self.grocery_level >= 10 * len(family_members_present):
                self.grocery_level -= 10 * len(family_members_present)
                
                # First, gather everyone's plans
                family_plans = {}
                for member in family_members_present:
                    if hasattr(member, 'plans') and member.plans:
                        family_plans[member.name] = member.plans[-1]
                
                # Coordinate schedules
                coordinated_schedule = self.coordinate_family_schedule(current_time)
                
                # Generate discussion about key points
                discussion_points = {
                    'school_transport': self.plan_school_transport(family_members_present),
                    'dinner_plans': self.plan_family_dinner(current_time),
                    'evening_activities': self.plan_evening_activities(family_members_present),
                    'special_needs': self.check_special_requirements(family_plans)
                }
                
                # Create breakfast conversation including schedule discussion
                context = {
                    'time': current_time,
                    'activity': 'family_breakfast',
                    'location': self.residence,
                    'participants': [member.name for member in family_members_present],
                    'schedule_discussion': discussion_points
                }
                conversation = self.generate_family_conversation(context)
                
                # Update everyone's memories and plans
                for member in family_members_present:
                    member.memory_manager.add_memory(
                        member.name,
                        "family_meal",
                        {
                            'activity': 'Family breakfast and schedule coordination',
                            'conversation': conversation,
                            'participants': [m.name for m in family_members_present],
                            'time': current_time,
                            'coordinated_schedule': coordinated_schedule,
                            'discussion_points': discussion_points,
                            'meal_type': 'breakfast',
                            'location': self.residence,
                            'grocery_used': 10
                        }
                    )
                    
                    # Update member's plan with coordinated schedule
                    if hasattr(member, 'plans') and member.plans:
                        updated_plan = self.merge_schedules(member.plans[-1], coordinated_schedule)
                        member.plans[-1] = updated_plan
                
                return "Had family breakfast and finalized daily schedule"
            else:
                return "Need to get breakfast supplies for the family"
            
        except Exception as e:
            print(f"Error in family breakfast for {self.name}: {str(e)}")
            return "Had simple family breakfast"

    def plan_school_transport(self, family_members):
        """Plan school transportation arrangements"""
        try:
            transport_plan = {}
            young_children = [
                member for member in family_members 
                if hasattr(member, 'age') and member.age < 15
            ]
            
            if young_children:
                # Find available parents
                parents = [
                    member for member in family_members
                    if hasattr(member, 'family_role') and member.family_role == 'parent'
                ]
                
                for parent in parents:
                    if hasattr(parent, 'workplaces') and parent.workplaces:
                        # Check if workplace is compatible with school transport
                        transport_plan[parent.name] = {
                            'can_dropoff': True,  # Add logic based on work schedule
                            'can_pickup': True    # Add logic based on work schedule
                        }
            
            return transport_plan
            
        except Exception as e:
            print(f"Error planning school transport: {str(e)}")
            return {}

    def plan_family_dinner(self, current_time):
        """Plan family dinner arrangements"""
        try:
            dinner_plan = {
                'time': '18:00',
                'location': self.residence,
                'grocery_status': self.grocery_level >= 10,
                'need_shopping': self.grocery_level < 50,
                'special_occasion': False  # Add logic for special occasions
            }
            return dinner_plan
            
        except Exception as e:
            print(f"Error planning family dinner: {str(e)}")
            return {'time': '18:00', 'location': self.residence}

    def plan_evening_activities(self, family_members):
        """Plan evening activities for family"""
        try:
            activities = {}
            for member in family_members:
                if hasattr(member, 'plans') and member.plans:
                    evening_plans = {
                        str(h): activity 
                        for h, activity in member.plans[-1].items() 
                        if 17 <= int(h) <= 21
                    }
                    activities[member.name] = evening_plans
            return activities
            
        except Exception as e:
            print(f"Error planning evening activities: {str(e)}")
            return {}

    def check_special_requirements(self, family_plans):
        """Check for any special requirements or conflicts"""
        try:
            special_needs = {
                'schedule_conflicts': [],
                'transport_needs': [],
                'meal_requirements': [],
                'shopping_needs': []
            }
            
            # Add logic to identify conflicts and special needs
            return special_needs
            
        except Exception as e:
            print(f"Error checking special requirements: {str(e)}")
            return {}

    def prepare_children_for_school(self, current_time):
        """Handle school preparation time (7:00 AM)"""
        try:
            if not hasattr(self, 'family_members') or 'children' not in self.family_members:
                return "No children to prepare for school"
                
            young_children = [
                child for child in self.family_members['children']
                if isinstance(child, dict) and child.get('age', 0) < 15
            ]
            
            if not young_children:
                return "No young children to prepare for school"
                
            # Pack lunches for children
            for child in young_children:
                if self.grocery_level >= 10:  # Need more groceries for lunch than breakfast
                    self.grocery_level -= 10
                    child_name = child.get('name')
                    if child_name in all_agents:
                        child_agent = all_agents[child_name]
                        child_agent.has_packed_lunch = True
                        child_agent.memory_manager.add_memory(
                            child_agent.name,
                            "preparation",
                            {
                                'activity': 'Parent packed school lunch',
                                'time': current_time
                            }
                        )
                else:
                    return "Need to get more groceries for children's lunches"
                    
            # Help children prepare
            preparation_conversation = self.generate_family_conversation({
                'activity': 'morning preparation',
                'time': current_time
            })
            
            for child in young_children:
                child_name = child.get('name')
                if child_name in all_agents:
                    child_agent = all_agents[child_name]
                    child_agent.memory_manager.add_memory(
                        child_name,
                        "preparation",
                        {
                            'activity': 'Getting ready for school with parent',
                            'conversation': preparation_conversation,
                            'time': current_time
                        }
                    )
                    
            return "Helped children prepare for school"
            
        except Exception as e:
            print(f"Error in prepare_children_for_school for {self.name}: {str(e)}")
            return "Basic school preparation completed"
        
    def handle_morning_departure(self, current_time, all_agents):
        """Handle morning departure (8:00-9:00)"""
        try:
            # Parents handle school dropoff first
            if self.family_role == 'parent':
                young_children = [
                    child for child in self.family_members.get('children', [])
                    if isinstance(child, dict) and child.get('age', 0) < 15
                ]
                if young_children:
                    return "Dropping children at school"
            
            # Students go to school
            if self.is_student:
                if self.needs_supervision:
                    return "Going to school with parent"
                else:
                    return f"Heading to {self.school_location}"
            
            # Workers go to work
            if self.workplaces and self.is_working(current_time):
                return f"Heading to {self.workplaces[0]}"
            
            return "Starting daily activities"
            
        except Exception as e:
            print(f"Error in morning departure: {e}")
            return "Continuing morning routine"

    def record_food_interaction(self, action, current_time):
        """Unified method for recording food-related memories and metrics"""
        location = locations.get(self.location)
        if not location:
            return
        
        # Specific handling for Fried Chicken Shop
        if self.location == "Fried Chicken Shop":
            # Get price from location object
            price = location.base_price
            current_day = (current_time // 24) + 1
            
            # Calculate final price with discount
            if hasattr(location, 'discount') and current_day in location.discount['days']:
                discount_amount = price * (location.discount['value'] / 100)
                final_price = price - discount_amount
            else:
                final_price = price
                discount_amount = 0
                
            purchase_details = {
                'content': action,
                'time': current_time,
                'location': location,
                'price': final_price,  # Use the actual final price
                'original_price': price,
                'used_discount': current_day in location.discount['days'],
                'discount_amount': discount_amount,
                'influenced_by': None,
                'nearby_agents': [a.name for a in locations[location].agents if a.name != self.name]
            }
            
            # Check for social influence
            recent_memories = self.memory_manager.get_recent_memories(self.name, current_time, 24)
            for memory in recent_memories:
                if memory['type'] == 'received_recommendation' and 'Fried Chicken Shop' in memory.get('content', ''):
                    purchase_details['influenced_by'] = memory['source']
                    break

            # Record in metrics
            self.metrics.record_interaction(
                self.name,
                location,
                "purchase",
                purchase_details
            )
            
    def generate_satisfaction_rating(self):
        """Generate satisfaction based on agent's actual experience"""
        # Get recent memories about Fried Chicken Shop
        recent_memories = self.memory_manager.get_recent_memories(
            self.name, 
            context="Fried Chicken Shop",
            limit=5
        )
        
        # Generate experience-based prompt
        prompt = f"""You are {self.name} evaluating your experience at the Fried Chicken Shop.
        Your recent experiences: {[m.get('content', '') for m in recent_memories]}
        
        Based on these experiences, provide ratings for:
        1. Overall satisfaction (1-5)
        2. Food quality (1-5)
        3. Price satisfaction (1-5)
        4. Service quality (1-5)
        5. Approximate wait time (in minutes)
        6. Would you recommend to others? (yes/no)
        7. How likely are you to return? (1-5)
        
        Respond in JSON format only."""
        
        try:
            response = generate(prompt)
            ratings = json.loads(response)
            
            return {
                'overall_rating': int(ratings.get('1', 4)),
                'food_quality': int(ratings.get('2', 4)),
                'price_satisfaction': int(ratings.get('3', 3)),
                'service': int(ratings.get('4', 4)),
                'wait_time': int(ratings.get('5', 10)),
                'would_recommend': ratings.get('6', 'yes').lower() == 'yes',
                'return_intention': int(ratings.get('7', 4))
            }
        except Exception as e:
            print(f"Error generating satisfaction: {e}")
            # Fallback ratings based on basic factors
            return self.generate_fallback_rating()

    def generate_fallback_rating(self):
        """Generate ratings based on basic factors if AI generation fails"""
        # Check if price was discounted
        current_day = (self.current_time // 24) + 1
        got_discount = current_day in [3, 4]
        
        # Check recent purchase history
        recent_purchases = [m for m in self.memory_manager.get_recent_memories(self.name, limit=10)
                          if m['type'] == 'purchase' and m['location'] == 'Fried Chicken Shop']
        is_repeat_customer = len(recent_purchases) > 1
        
        # Base ratings slightly higher for repeat customers and during discounts
        base_rating = 3
        if is_repeat_customer:
            base_rating += 1
        if got_discount:
            base_rating = min(5, base_rating + 1)
            
        return {
            'overall_rating': base_rating,
            'food_quality': base_rating,
            'price_satisfaction': base_rating + 1 if got_discount else base_rating,
            'service': base_rating,
            'wait_time': 10 if base_rating >= 4 else 15,
            'would_recommend': base_rating >= 4,
            'return_intention': base_rating
        }

    def generate_recommendation(self, target_agent, experiences, current_time):
        """Generate personalized recommendation based on experiences"""
        try:
            # Get recent experiences (including negative ones)
            recent_experiences = self.memory_manager.get_recent_memories(
                self.name,
                limit=5,
                location_filter="Fried Chicken Shop"
            )
            
            # Generate recommendation based on ALL experiences
            prompt = self.create_recommendation_prompt(context)
            recommendation = generate(prompt)
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation for {self.name}: {str(e)}")
            return None

    def record_word_of_mouth(self, message, listener, current_time):
        """Record word of mouth with proper metrics integration"""
        # Analyze sentiment
        sentiment_prompt = f"""Analyze the sentiment about the Fried Chicken Shop in this recommendation:
        '{message}'
        Respond with only one word: positive, negative, or neutral."""
        
        sentiment = generate(sentiment_prompt).strip().lower()
        
        # Record in metrics with proper format matching
        self.metrics.record_interaction(
                self.name,
            self.location,
            "word_of_mouth",
                {
                'sentiment': sentiment,
                'listener': listener,
                'content': message,
                    'time': current_time,
                    'location': self.location,
                # Add fields that match metrics tracking
                'discount_mentioned': any(keyword in message.lower() for keyword in SALES_KEYWORDS),
                'type': 'recommendation'
            }
        )

    def social_interaction(self, all_agents, action, current_time):
        """Enhanced social interaction with proper metrics tracking"""
        if "Fried Chicken Shop" in action.lower():
            listeners = [agent for agent in all_agents 
                       if agent.location == self.location 
                       and agent.name != self.name]
            
            for listener in listeners:
                recommendation = self.generate_recommendation(listener)
                
                # Record in social network metrics
                self.metrics.record_interaction(
                    self.name,
                    self.location,
                    "word_of_mouth",
                    {
                        'content': recommendation,
                        'listener': listener.name,
                        'time': current_time,
                        'location': self.location,
                        'type': 'social_interaction'
                    }
                )
                
                # Add to listener's memory
                listener.memory_manager.add_memory(
                    listener.name,
                    "received_recommendation",
                    {
                        'from_agent': self.name,
                        'message': recommendation,
                        'timestamp': current_time,
                        'location': self.location
                    }
                )
                
                # Update social network tracking
                self.metrics.daily_metrics[self.metrics.current_day]['social_network']['information_flow'].append({
                    'from': self.name,
                    'to': listener.name,
                    'time': current_time,
                    'content': recommendation,
                    'location': self.location
                })
                
                # Update community impact
                self.metrics.daily_metrics[self.metrics.current_day]['social_network']['community_impact'][self.location] = \
                    self.metrics.daily_metrics[self.metrics.current_day]['social_network']['community_impact'].get(self.location, 0) + 1

    def update_memory(self, action, time, all_agents=None):
        """Update agent's memory with proper word-of-mouth tracking"""
        if action and "Fried Chicken Shop" in action:
            # Determine sentiment
            sentiment_prompt = f"Analyze the sentiment towards the Fried Chicken Shop in this text. Only respond with one word: 'positive', 'negative', or 'neutral'. Text: {action}"
            sentiment = generate(sentiment_prompt).strip().lower()
            
            # Find potential listeners
            listeners = [agent.name for agent in all_agents 
                        if agent.location == self.location 
                        and agent.name != self.name]
            
            for listener in listeners:
                # Record word of mouth in metrics
                self.metrics.record_interaction(
                    self.name,
                    self.location,
                    "word_of_mouth",
                    {
                        'sentiment': sentiment,
                        'listener': listener,
                        'content': action,
                        'time': time
                    }
                )
                
                # Record in memory manager
                self.memory_manager.add_memory(
                    self.name,
                    "word_of_mouth",
                    {
                        'content': action,
                        'sentiment': sentiment,
                        'listener': listener,
                        'location': self.location,
                        'time': time
                    }
                )

    def get_purchase_price(self, location_name, purchase_type):
        """Get price for purchase based on location"""
        location = locations.get(location_name)
        if location and location.type == purchase_type:
            return location.base_price
        return None

    def buy_food(self, current_time):
        # Use self.locations instead of global locations
        location = self.locations.get(self.location)
        if not location or location.type not in ['local_shop', 'grocery']:
            return False, "Not at a food establishment"
        
        # Check time since last purchase at this specific location
        if self.location in self.location_last_purchase:
            time_since_last_purchase = current_time - self.location_last_purchase[self.location]
            if time_since_last_purchase < 4:
                return False, f"Too soon to purchase again at {self.location}"
        
        # Get social influence from recent recommendations
        recent_recommendations = self.memory_manager.retrieve_memories(
            self.name,
            current_time,
            memory_type="received_recommendation",
            limit=5
        )
        
        # Calculate social influence
        social_influence = 0
        if recent_recommendations:
            positive_count = sum(1 for rec in recent_recommendations 
                               if rec.get('sentiment') == 'positive' and 
                               rec.get('location') == self.location)
            social_influence = positive_count / len(recent_recommendations)
        
        # Get the fixed base price
        price = self.get_purchase_price(self.location, location.type)
        if not self.can_afford(price):
            return False, "Cannot afford meal"
        
        # Social influence affects decision to buy, not the price
        should_buy = (
            self.energy_level <= 40 or  # Very hungry
            (self.is_meal_time(current_time)[0]) or  # During meal time
            social_influence > 0.5  # Strong positive recommendations
        )
        
        if should_buy:
            success = self.process_purchase(price, current_time)
            if success:
                # Record the purchase in memory
                self.memory_manager.add_memory(
                    self.name,
                    "meal",
                    {
                        'type': 'purchased_meal',
                        'location': self.location,
                        'time': current_time,
                        'price': price,
                        'influenced_by_recommendations': social_influence > 0,
                        'energy_restored': True
                    }
                )
                
                # Update the last purchase time for this specific location
                self.location_last_purchase[self.location] = current_time
                return True, f"Bought meal at {self.location} for ${price:.2f}"
        
        return False, "Decided not to purchase"

    def buy_groceries(self, current_time):
        """Buy groceries at grocery stores"""
        if self.location not in experiment_settings['location_prices']['retail']:
            return False, "Not at a grocery store"
        
        needed = 100 - self.grocery_level
        price_per_unit = self.get_purchase_price(self.location, "retail")
        total_price = needed * price_per_unit
        
        if not self.can_afford(total_price):
            return False, "Cannot afford groceries"
            
        success, message = self.make_purchase(self.location, current_time)
        if success:
            self.grocery_level = 100
            return True, f"Bought groceries at {self.location} for ${total_price:.2f}"
        return False, message

    def can_afford(self, amount):
        """Check if purchase is affordable based on family income"""
        if self.shared_income:
            return self.get_household_income() >= amount
        return self.money >= amount

    def get_household_income(self):
        """Calculate total household income"""
        if not self.family_unit:
            return self.money
            
        family_info = town_data['family_units'][self.family_unit]
        total_income = 0
        
        for parent, income_info in family_info['household_income'].items():
            if income_info['type'] == 'monthly':
                total_income += income_info['amount']
            elif income_info['type'] == 'annual':
                total_income += income_info['amount'] / 12
            elif income_info['type'] == 'hourly':
                total_income += income_info['amount'] * income_info.get('hours_per_week', 40) * 4
                
        return total_income

    def coordinate_family_schedule(self, current_time):
        if not self.family_unit:
            return None
            
        hour = int(current_time) % 24
        family_data = town_data['family_units'][self.family_unit]
        members = family_data['members']
        
        # Morning coordination
        if hour == 7:
            if self.family_role == 'parent':
                young_children = [
                    child for child in members['children']
                    if child['age'] < 15  # Use age-based supervision
                ]
                if young_children:
                    return "Help children prepare for school"

        # After school/work coordination
        elif hour == 17:  # End of work/school day
            if self.family_role == 'parent':
                young_children = [c for c in self.family_members['children'] 
                                if c.startswith('child_') and int(c.split('_')[1]) <= 11]
                if young_children:
                    return "Pick up children from school"
                    
        # Evening family time
        elif 18 <= hour <= 20:
            if self.location == self.residence:
                return "Spend time with family"
                
        return None

    def handle_food_needs(self, hour):
        try:
            # Use self.locations instead of global locations
            current_location = self.location
            location = self.locations.get(current_location)
            
            if self.needs_food(hour):
                if current_location == "Fried Chicken Shop":
                    return self.buy_food(hour)
                
                # Pass locations to check_nearby_food_locations
                nearby_options = self.check_nearby_food_locations(hour)

        except Exception as e:
            print(f"Error in handle_food_needs for {self.name}: {str(e)}")
            return False, str(e)

    def check_family_meal_schedule(self, hour):
        """Check if it's family meal time"""
        if not self.family_unit:
            return False
        
        # Morning family breakfast (6-8)
        if 6 <= hour % 24 <= 8:
            return True
        
        # Evening family dinner (17-20)
        if 17 <= hour % 24 <= 20:
            # Check if parents are home
            parents_home = any(
                agents[parent].location == self.residence 
                for parent in self.family_members.get('parents', [])
                if isinstance(parent, str)
            )
            return parents_home

        return False

    def handle_school_meal(self, hour):
        """Handle meals during school hours"""
        if not self.is_during_school_hours(hour):
            return False, "Not during school hours"

        if 11 <= hour % 24 <= 12:  # Lunch time
            if self.needs_supervision:  # Under 15
                if hasattr(self, 'has_packed_lunch') and self.has_packed_lunch:
                    self.has_packed_lunch = False  # Consume lunch
                    self.energy_level = 100
                    self.last_meal_time = hour
                    if hasattr(self, 'bought_lunch') and self.bought_lunch:
                        self.bought_lunch = False
                        return True, "Ate takeout lunch at school"
                    return True, "Ate packed lunch at school"
                return False, "No lunch available"
            else:  # Independent student (15+)
                # Regular cafeteria purchase logic
                if self.can_afford(10):
                    self.money -= 10
                    self.energy_level = 100
                    self.last_meal_time = hour
                    return True, "Bought lunch at school cafeteria"
                return False, "Cannot afford school lunch"
    
        return False, "Not lunch time"

    def check_household_budget(self):
        """Check if household can afford groceries or dining out"""
        if not self.family_unit:
            return self.can_afford(50)  # Basic threshold for individual
        
        household = households.get(self.residence)
        if not household:
            return False
        
        # Check if household can afford either groceries or dining out
        min_budget_needed = 50 * len(household['members'])  # Estimate per person
        return household['money'] >= min_budget_needed

    def check_can_purchase_at_location(self, location, current_time):
        """Check if enough time has passed since last purchase at this location"""
        if not hasattr(self, 'location_last_purchase'):
            self.location_last_purchase = {}
        
        if location not in self.location_last_purchase:
            return True
        
        time_since_last_purchase = current_time - self.location_last_purchase[location]
        return time_since_last_purchase >= 4  # 4 hour cooldown

    def handle_location_interaction(self, current_time):
        """Handle interactions based on current location type"""
        try:
            location = locations.get(self.location)
            if not location:
                return False, "Invalid location"
            
            # Check if location is open
            if not location.is_open(current_time):
                return False, f"{self.location} is closed"
            
            # Handle different location types
            if location.type == 'local_shop':
                return self.handle_dining_interaction(current_time)
            elif location.type == 'grocery':
                return self.handle_retail_interaction(current_time)
            elif location.type == 'education':
                return self.handle_education_interaction(current_time)
            elif location.type == 'work_office':
                return self.handle_work_interaction(current_time)
            elif location.type == 'residence':
                return True, f"At home in {self.location}"
            elif location.type == 'community':
                return True, f"Spending time at {self.location}"
            
            return True, f"Spent time at {self.location}"
            
        except Exception as e:
            print(f"Error in location interaction for {self.name}: {str(e)}")
            return False, str(e)

    def handle_dining_interaction(self, current_time):
        """Handle dining-specific interactions"""
        try:
            location = locations.get(self.location)
            if not location or location.type != 'local_shop':
                return False, "Not at a dining establishment"
            
            base_price = location.base_price
            
            # Check for discounts
            current_day = (current_time // 24) + 1
            if hasattr(location, 'discount') and current_day in location.discount['days']:
                discount = location.discount['value']
                final_price = base_price * (1 - discount/100)
            else:
                final_price = base_price
            
            if not self.can_afford(final_price):
                return False, f"Cannot afford meal at {self.location}"
            
            success = self.process_purchase(final_price, current_time)
            if success:
                self.record_dining_experience(current_time, final_price)
                return True, f"Enjoyed a meal at {self.location}"
            
            return False, "Purchase failed"
            
        except Exception as e:
            print(f"Error in dining interaction for {self.name}: {str(e)}")
            return False, str(e)

    def handle_social_interaction(self, current_time, nearby_agents):
        """Handle social interactions and word-of-mouth"""
        try:
            # Skip if no nearby agents
            if not nearby_agents:
                return
            
            # Get recent experiences
            recent_experiences = self.memory_manager.get_recent_memories(
                self.name,
                limit=5,
                location_filter="Fried Chicken Shop"
            )
            
            # Share experiences with nearby agents
            for nearby_agent in nearby_agents:
                # Skip if already shared recently
                if self.has_recent_interaction_with(nearby_agent.name, current_time):
                    continue
                
                # Generate and share recommendation
                if recent_experiences:
                    recommendation = self.generate_recommendation(
                        nearby_agent,
                        recent_experiences,
                        current_time
                    )
                    
                    # Record the interaction
                    self.record_social_interaction(
                        nearby_agent.name,
                        recommendation,
                        current_time
                    )
                    
                    # Update nearby agent's memory
                    nearby_agent.receive_recommendation(
                        self.name,
                        recommendation,
                        current_time
                    )
                
        except Exception as e:
            print(f"Error in social interaction for {self.name}: {str(e)}")

    def handle_education_interaction(self, current_time):
        """Handle education-related activities"""
        try:
            hour = current_time % 24
            
            # Skip if outside school hours
            if not (8 <= hour < 15):
                return False, "Outside school hours"
            
            # Handle different student types
            if self.student_type == 'high_school':
                return self.handle_high_school_activity(hour)
            elif self.student_type == 'college':
                return self.handle_college_activity(hour)
            elif self.student_type == 'part_time_student':
                return self.handle_part_time_study(hour)
            
            return False, "Not a student"
            
        except Exception as e:
            print(f"Error in education interaction for {self.name}: {str(e)}")
            return False, str(e)

    def handle_high_school_activity(self, hour):
        """Handle high school student activities"""
        if 8 <= hour < 15:
            self.memory_manager.add_memory(
                self.name,
                "education",
                {
                    'activity': 'Attended classes at Town Public High School',
                    'location': 'Town Public High School',
                    'time': hour
                }
            )
            return True, "Attended high school classes"
        return False, "Not during school hours"

    def handle_college_activity(self, hour):
        """Handle college student activities"""
        if 8 <= hour < 16:
            self.memory_manager.add_memory(
                self.name,
                "education",
                {
                    'activity': 'Attended college classes',
                    'location': 'Town Community College',
                    'time': hour
                }
            )
            return True, "Attended college classes"
        return False, "Not during class hours"

    def handle_part_time_study(self, hour):
        """Handle part-time student activities"""
        if 18 <= hour < 21:  # Evening classes
            self.memory_manager.add_memory(
                self.name,
                "education",
                {
                    'activity': 'Attended evening classes',
                    'location': 'Town Community College',
                    'time': hour
                }
            )
            return True, "Attended evening classes"
        return False, "Not during evening class hours"

    def is_meal_time(self, current_time):
        """Check if it's a regular meal time"""
        hour = current_time % 24
        
        for meal, (start, end) in self.meal_schedule.items():
            if start <= hour < end:
                return True, meal
        return False, None

    def update_energy(self, current_time):
        """Update energy level, decreasing by 20 every hour"""
        if self.last_meal_time is None:
            self.last_meal_time = current_time - 1  # Initialize with 1 hour ago
        
        hours_passed = current_time - self.last_meal_time
        energy_decrease = hours_passed * 20
        self.energy_level = max(0, self.energy_level - energy_decrease)

    def needs_food(self, current_time):
        """Determine if agent needs to eat based on energy and meal times"""
        self.update_energy(current_time)
        is_meal_time, meal_type = self.is_meal_time(current_time)
        
        return (
            self.energy_level <= 30 or  # Very low energy
            (is_meal_time and self.energy_level <= 60)  # Moderately low energy during meal time
        )

    def prepare_meal(self, current_time):
        """Handle meal preparation with energy system"""
        try:
            # First check if we need food
            if not self.needs_food(current_time):
                return False, "Energy level still good, don't need to eat yet"
            
            # Verify we're at residence
            if self.location != self.residence:
                return False, "Can only prepare meals at residence"
            
            # Check if we have enough groceries
            if not hasattr(self, 'grocery_level') or self.grocery_level < 10:
                return False, "Not enough groceries to prepare a meal"
            
            # Get current meal type
            is_meal_time, meal_type = self.is_meal_time(current_time)
            if not is_meal_time:
                return False, "Not a proper meal time"
            
            # Set grocery consumption for different meal types
            if meal_type == 'breakfast':
                grocery_cost = 10
            elif meal_type == 'lunch' and self.needs_supervision:
                grocery_cost = 10  # Packed lunch
            elif meal_type == 'dinner':
                grocery_cost = 15  # Dinner
                if self.family_unit:  # Family dinner needs more groceries
                    present_family = len([m for m in self.family_members.get('parents', []) + 
                                       self.family_members.get('children', [])
                                       if isinstance(m, str) and 
                                       agents[m].location == self.residence])
                    grocery_cost *= present_family
            
            if self.grocery_level < grocery_cost:
                return False, f"Not enough groceries for {meal_type}"
                
            # Consume groceries and update energy
            self.grocery_level -= grocery_cost
            self.energy_level = 100
            self.last_meal_time = current_time
            
            # Record the meal preparation in memory
            self.memory_manager.add_memory(
                self.name,
                "meal",
                {
                    'type': 'prepared_meal',
                    'meal_type': meal_type,
                    'location': self.residence,
                    'time': current_time,
                    'with_family': bool(self.family_unit),
                    'grocery_used': grocery_cost,
                    'energy_restored': True
                }
            )
            
            return True, f"Prepared {meal_type} at {self.residence}"
            
        except Exception as e:
            print(f"Error in prepare_meal for {self.name}: {str(e)}")
            return False, str(e)

    def check_grocery_needs(self):
        """Check if agent needs to buy groceries"""
        if not hasattr(self, 'grocery_level'):
            self.grocery_level = 100  # Initialize if not exists
            return False
        
        return self.grocery_level < 30  # Return True if groceries are low

    def process_purchase(self, price, current_time):
        """Process a purchase and update relevant tracking"""
        try:
            # Check if can afford
            if not self.can_afford(price):
                return False
            
            # Process payment based on whether agent has personal income
            if self.has_income():
                self.money -= price
            else:
                households[self.residence]['money'] -= price
            
            # Update purchase tracking for this specific location
            self.location_last_purchase[self.location] = current_time
            
            return True
            
            # Calculate price with possible discount
            base_price = location.base_price
            final_price = self.calculate_price_with_discount(base_price, location)
            
            # Process purchase
            if self.process_purchase(final_price, hour):
                # Record purchase time for THIS location
                self.location_last_purchase[location.name] = hour
                
                # Generate satisfaction score (higher if socially influenced)
                recent_recommendations = self.memory_manager.retrieve_memories(
                    self.name, hour, memory_type="received_recommendation", limit=5
                )
                has_positive_recommendations = any(
                    rec.get('sentiment') == 'positive' and 
                    rec.get('location') == location.name 
                    for rec in recent_recommendations
                )
                satisfaction = random.uniform(0.8, 1.0) if has_positive_recommendations else random.uniform(0.6, 0.9)
                
                # Record in memory
                memory = {
                    'type': 'purchase',
                    'location': location.name,
                    'amount': final_price,
                    'timestamp': datetime.now(),
                    'satisfaction': satisfaction,
                    'was_recommended': has_positive_recommendations
                }
                self.memory_manager.add_memory(self.name, memory)
                
                # Record in metrics
                if hasattr(self.metrics, 'record_purchase'):
                    self.metrics.record_purchase(
                        agent_name=self.name,
                        location_name=location.name,
                        amount=final_price,
                        hour=hour,
                        is_discount=final_price < base_price,
                        was_recommended=has_positive_recommendations
                    )
                
                return f"Purchased food at {location.name} for ${final_price:.2f}"
            
            return f"Cannot afford food at {location.name}"
        except Exception as e:
            print(f"Error processing purchase for {self.name}: {str(e)}")
            return False

    def record_dining_experience(self, current_time, final_price):
        # All experiences are recorded, regardless of sentiment
        self.memory_manager.add_memory(
            self.name,
            "meal",
            {
                'activity': f'Purchased meal at {self.location}',
                'location': self.location,
                'time': current_time,
                'price': final_price,
                'energy_restored': True
            }
        )

    def has_income(self):
        """Check if agent has personal income"""
        if not self.description:
            return False
        return 'income' in self.description

    def calculate_price_with_discount(self, base_price, location):
        """Calculate final price with possible discount"""
        current_day = (self.current_time // 24) + 1
        if hasattr(location, 'discount') and current_day in location.discount['days']:
            discount = location.discount['value']
            return base_price * (1 - discount/100)
        return base_price

    def has_recent_interaction_with(self, other_agent_name, current_time):
        """Check if had recent interaction with another agent"""
        recent_interactions = self.memory_manager.retrieve_memories(
            self.name,
            current_time,
            memory_type="social_interaction",
            limit=1,
            filter_func=lambda m: m.get('target_agent') == other_agent_name
        )
        if not recent_interactions:
            return False
        last_interaction_time = recent_interactions[0].get('time', 0)
        return (current_time - last_interaction_time) < 4  # 4-hour cooldown

    def is_during_school_hours(self, hour):
        """Check if current time is during school hours"""
        hour = hour % 24
        return 8 <= hour < 15  # School hours are 8 AM to 3 PM

    def handle_supervised_meal(self, hour):
        """Handle meals for students needing supervision"""
        if not self.family_unit:
            return False, "No family unit to coordinate with"
            
        # Check if parent is available
        parent_available = any(
            agents[parent].location == self.location 
            for parent in self.family_members.get('parents', [])
            if isinstance(parent, str)
        )
        
        if parent_available:
            return self.handle_food_needs(hour)
        return False, "Need parent supervision for meal"

    def handle_independent_meal(self, hour):
        """Handle meals for independent students"""
        return self.handle_food_needs(hour)

    def generate_family_daily_schedule(self, current_time):
        """Generate comprehensive family schedule for the day"""
        if not self.family_unit or self.family_role != 'parent':
            return None
        
        daily_schedule = {
            'required_meals': {
                'breakfast': {'time': (6, 8), 'location': self.residence, 'required': True},
                'dinner': {'time': (18, 20), 'location': self.residence, 'required': True}
            },
            'school_activities': [],
            'work_schedules': [],
            'free_time': []
        }
        
        # Add children's school schedules
        for child in self.family_members.get('children', []):
            if isinstance(child, str) and child in agents:
                child_agent = agents[child]
                if child_agent.needs_supervision:
                    daily_schedule['school_activities'].append({
                        'type': 'school',
                        'member': child,
                        'dropoff': {'time': 8, 'location': child_agent.school_location},
                        'pickup': {'time': 15, 'location': child_agent.school_location}
                    })
        
        return daily_schedule

    def broadcast_family_schedule(self, schedule, current_time, conversation):
        """Share daily schedule with family members through conversation"""
        if self.family_role != 'parent':
            return
        
        # Generate morning family conversation
        conversation_context = {
            'time': current_time,
            'location': self.residence,
            'schedule': schedule,
            'present_family': [
                member for member_type in ['parents', 'children']
                for member in self.family_members.get(member_type, [])
                if isinstance(member, str) and 
                agents[member].location == self.residence
            ]
        }
        
        # Generate family conversation about the day's schedule
        morning_conversation = self.generate_family_conversation(conversation_context)
        
        # Record the conversation in everyone's memory
        for member in conversation_context['present_family']:
            agents[member].memory_manager.add_memory(
                agents[member].name,
                "family_conversation",
                {
                    'type': 'morning_planning',
                    'conversation': morning_conversation,
                    'participants': conversation_context['present_family'],
                    'time': current_time,
                    'location': self.residence,
                    'schedule_discussed': schedule
                }
            )
            # Each member receives and acknowledges the schedule
            agents[member].receive_family_schedule(
                schedule, 
                current_time, 
                morning_conversation
            )

    def generate_family_conversation(self, context):
        """Generate contextual family conversation about daily schedule"""
        try:
            hour = context['time'] % 24
            schedule = context['schedule']
            present_family = context['present_family']
            
            # Build conversation prompt based on family context
            prompt = f"""You are {self.name}, having a morning family conversation at {hour:02d}:00.
            Present family members: {', '.join(present_family)}
            Today's schedule includes:
            - School dropoffs: {[act['member'] for act in schedule.get('school_activities', [])]}
            - Family meals: {list(schedule.get('required_meals', {}).keys())}
            - Work schedules: {[work['member'] for work in schedule.get('work_schedules', [])]}
            
            Generate a natural family conversation about coordinating today's schedule, including:
            1. Morning routine reminders
            2. School/work schedules
            3. Pickup arrangements
            4. Meal planning
            5. Evening family time
            
            Format as a brief dialogue between family members."""
            
            conversation = generate(prompt)
            return conversation
            
        except Exception as e:
            print(f"Error generating family conversation: {e}")
            return "Basic schedule coordination discussion"

    def receive_family_schedule(self, schedule, current_time, conversation):
        """Process received family schedule with response"""
        self.family_schedule = schedule
        
        # Generate appropriate response based on role
        if self.family_role == 'child':
            if self.needs_supervision:  # Under 15
                response = "Acknowledges schedule and asks about lunch preparation"
            else:  # 15 and older
                response = "Confirms understanding and mentions any after-school activities"
        else:  # Parent
            response = "Confirms schedule and coordinates responsibilities"
        
        # Record both schedule and conversation
        self.memory_manager.add_memory(
            self.name,
            "family_coordination",
            {
                'schedule': schedule,
                'conversation': conversation,
                'my_response': response,
                'time': current_time,
                'type': 'morning_planning',
                'location': self.residence
            }
        )

    def check_family_obligations(self, current_time):
        """Check and handle family-related responsibilities"""
        try:
            # 1. Basic validation
            if not hasattr(self, 'family_role') or self.family_role != 'parent':
                return None
                
            hour = current_time % 24
            
            # 2. Get family members that need supervision
            young_children = [
                child for child in self.family_members.get('children', [])
                if isinstance(child, dict) and child.get('age', 0) < 15
            ]
            
            if not young_children:
                return None
                
            # 3. Morning routine (6:00-8:00)
            if 6 <= hour < 8:
                if self.location == self.residence:
                    return "Help children prepare for the day"
                    
            # 4. School dropoff (8:00-9:00)
            elif 8 <= hour < 9:
                children_schools = [
                    child.get('school_location') 
                    for child in young_children 
                    if child.get('school_location')
                ]
                if children_schools:
                    return f"Drop off children at {', '.join(set(children_schools))}"
                    
            # 5. School pickup (15:00)
            elif hour == 15:
                children_schools = [
                    child.get('school_location') 
                    for child in young_children 
                    if child.get('school_location')
                ]
                if children_schools:
                    return f"Pick up children from {', '.join(set(children_schools))}"
                    
            # 6. Evening routine (17:00-20:00)
            elif 17 <= hour < 20:
                if self.location == self.residence:
                    return "Spend time with family"
                    
            return None
            
        except Exception as e:
            print(f"Error in family obligations check for {self.name}: {str(e)}")
            return None

    def generate_family_reminder(self, context):
        """Generate reminder conversation about upcoming schedule items"""
        try:
            hour = context['time'] % 24
            schedule = context['schedule']
            present_family = context['present_family']
            
            # Find next scheduled activity
            next_activities = []
            for activity_type, activities in schedule.items():
                if activity_type == 'school_activities':
                    for act in activities:
                        if act['pickup']['time'] > hour:
                            next_activities.append(f"School pickup for {act['member']} at {act['pickup']['time']}:00")
                elif activity_type == 'required_meals':
                    for meal, details in activities.items():
                        start_time, _ = details['time']
                        if start_time > hour:
                            next_activities.append(f"{meal} at {start_time}:00")
            
            prompt = f"""You are {self.name}, reminding family members about upcoming activities.
            Current time: {hour:02d}:00
            Present family: {', '.join(present_family)}
            Next activities: {', '.join(next_activities) if next_activities else 'No more scheduled activities today'}
            
            Generate a brief, natural reminder conversation about upcoming activities."""
            
            reminder = generate(prompt)
            return reminder
        
        except Exception as e:
            print(f"Error generating reminder: {e}")
            return "Brief schedule reminder"

    def decide_action(self, current_time):
        """Hierarchical decision making"""
        # 1. Check family obligations first
        family_obligation = self.check_family_obligations(current_time)
        if family_obligation:
            return family_obligation
        
        # 2. If no family obligations, check individual needs
        if self.needs_food(current_time):
            # Check if during family meal time
            is_family_meal = self.check_family_meal_schedule(current_time)
            if is_family_meal:
                return self.handle_family_meal(current_time)
            else:
                # Individual food decision - use the existing handle_food_needs
                return self.handle_food_needs(current_time)
        
        # 3. If no food needs, continue with other activities
        return None

    def update_family_status(self, current_time, status_type, message):
        """Update family about important status changes"""
        if not self.family_unit:
            return
        
        for member_type in ['parents', 'children']:
            for member in self.family_members.get(member_type, []):
                if isinstance(member, str) and member in agents:
                    agents[member].receive_status_update({
                        'from': self.name,
                        'type': status_type,
                        'message': message,
                        'time': current_time
                    })

    def handle_family_meal(self, current_time):
        """Handle family meal coordination and execution"""
        # Must be at residence for family meal
        if self.location != self.residence:
            return False, "Must be at residence for family meal"
        
        # Count present family members
        present_family_members = [
            member for member_type in ['parents', 'children']
            for member in self.family_members.get(member_type, [])
            if isinstance(member, str) and 
               member != self.name and 
               agents[member].location == self.residence
        ]
        
        total_members = len(present_family_members) + 1  # Include self
        groceries_needed = 10 * total_members  # 10 groceries per person
        
        if present_family_members:
            # Check if enough groceries for everyone
            if self.grocery_level >= groceries_needed:
                # Prepare family meal
                self.grocery_level -= groceries_needed
                self.energy_level = 100
                self.last_meal_time = current_time
                
                # Record family meal for all present members
                for member in present_family_members:
                    agents[member].energy_level = 100
                    agents[member].last_meal_time = current_time
                    agents[member].memory_manager.add_memory(
                        member,
                        "family_meal",
                        {
                            'activity': 'Shared family meal at home',
                            'location': self.residence,
                            'time': current_time,
                            'type': 'family_coordination',
                            'members_present': total_members
                        }
                    )
                
                return True, f"Prepared family meal for {total_members} people"
            else:
                # Not enough groceries - decide alternative
                if self.check_household_budget():
                    # Try to get groceries first if can afford
                    if any(loc.type == 'grocery' and loc.is_open(current_time) 
                          for loc in locations.values()):
                        return False, "Need to buy groceries for family meal"
                    # If grocery stores closed, try restaurants
                    elif any(loc.type == 'local_shop' and loc.is_open(current_time) 
                            for loc in locations.values()):
                        return False, "Consider family dinner at restaurant"
                return False, "Not enough groceries and limited options available"
        
        return False, "Family members not present for meal"

    def prepare_for_school(self, hour, all_agents):
        try:
            if self.family_role == 'parent':
                if 7 <= hour < 8:
                    needs_takeout = self.grocery_level < 10
                    
                    # Use self.locations instead of global locations
                    current_open_shops = [loc for loc in self.locations.values() 
                                        if loc.type in ['local_shop'] and 
                                        loc.is_open(hour)]
                    
                    next_hour_shops = [loc for loc in self.locations.values() 
                                     if loc.type in ['local_shop'] and 
                                     loc.is_open(hour + 1)]
                    
                    for child in self.family_members.get('children', []):
                        if isinstance(child, dict) and child.get('age', 0) < 15:
                            child_name = child['name']
                            if child_name in all_agents:
                                child_agent = all_agents[child_name]
                                
                                # If we have enough groceries, pack lunch
                                if self.grocery_level >= 10:
                                    self.grocery_level -= 10
                                    child_agent.has_packed_lunch = True
                                    child_agent.memory_manager.add_memory(
                                        child_agent.name,
                                        "meal_prep",
                                        {'activity': 'Parent packed school lunch'}
                                    )
                                    continue
                                
                                # Handle takeout scenario
                                if needs_takeout:
                                    # Try currently open shops first
                                    available_shops = current_open_shops or next_hour_shops
                                    if available_shops:
                                        # Find most affordable option
                                        affordable_shops = [shop for shop in available_shops 
                                                         if self.can_afford(shop.base_price)]
                                        if affordable_shops:
                                            chosen_shop = min(affordable_shops, 
                                                            key=lambda x: x.base_price)
                                            
                                            # If shops aren't open yet, wait
                                            if not current_open_shops:
                                                self.memory_manager.add_memory(
                                                    self.name,
                                                    "planning",
                                                    {
                                                        'activity': f'Waiting for {chosen_shop.name} to open',
                                                        'child': child_name
                                                    }
                                                )
                                                return False, f"Need to wait for {chosen_shop.name} to open"
                                            
                                            # Buy takeout lunch
                                            if self.process_purchase(chosen_shop.base_price, hour):
                                                child_agent.has_packed_lunch = True
                                                child_agent.bought_lunch = True
                                                child_agent.memory_manager.add_memory(
                                                    child_agent.name,
                                                    "meal_prep",
                                                    {
                                                        'activity': f'Parent bought takeout lunch from {chosen_shop.name}',
                                                        'price': chosen_shop.base_price
                                                    }
                                                )
                                                self.memory_manager.add_memory(
                                                    self.name,
                                                    "purchase",
                                                    {
                                                        'activity': f'Bought takeout lunch for {child_name}',
                                                        'location': chosen_shop.name,
                                                        'price': chosen_shop.base_price
                                                    }
                                                )
                                                return True, f"Bought takeout lunch for {child_name} from {chosen_shop.name}"
                                            return False, "Purchase failed"
                                        return False, "Cannot afford takeout lunch"
                                    return False, "No food locations open or opening soon"
                    
                    # After handling all children
                    return True, "Finished preparing children's lunches"
                
                # If it's not lunch prep time yet
                elif hour < 7:
                    return False, "Too early to prepare lunches"
                else:
                    return False, "Lunch preparation time has passed"
            
            return False, "Not a parent"
            
        except Exception as e:
            print(f"Error in prepare_for_school for {self.name}: {str(e)}")
            return False, f"Error preparing for school: {str(e)}"

    def check_nearby_food_locations(self, current_time):
        """Check nearby food locations and their availability"""
        try:
            # Get locations based on agent's preferences and circumstances
            preferred_locations = []
            
            # Family members prefer family-friendly locations during meal times
            if self.family_unit:
                preferred_locations.extend([
                    loc for loc in self.locations.values()
                    if loc.type == 'local_shop' and 'family' in loc.description.lower()
                ])
                
            # Students prefer affordable options
            elif self.is_student:
                preferred_locations.extend([
                    loc for loc in self.locations.values()
                    if loc.type == 'local_shop' and loc.base_price <= 15
                ])
                
            # Working adults consider convenience and location
            elif self.workplaces:
                workplace = self.workplaces[0]
                preferred_locations.extend([
                    loc for loc in self.locations.values()
                    if loc.type == 'local_shop' and 
                    self.calculate_distance(workplace, loc.name) <= 2  # Nearby locations
                ])
                
            # If no specific preferences, consider all options
            if not preferred_locations:
                preferred_locations = [
                    loc for loc in self.locations.values()
                    if loc.type in ['local_shop', 'grocery']
                ]
                
            # Filter for open locations
            open_locations = [loc for loc in preferred_locations if loc.is_open(current_time)]
            
            if not open_locations:
                return False, "No suitable food locations open nearby"
                
            # Choose based on affordability
            affordable_locations = [loc for loc in open_locations if self.can_afford(loc.base_price)]
            
            if not affordable_locations:
                return False, "Cannot afford nearby food locations"
                
            # Choose the most suitable option
            chosen_location = min(affordable_locations, key=lambda x: x.base_price)
            
            return True, {
                'location': chosen_location.name,
                'price': chosen_location.base_price,
                'is_open': True
            }
            
        except Exception as e:
            print(f"Error checking food locations for {self.name}: {str(e)}")
            return False, str(e)

    def handle_work_and_pickup(self, hour):
        """Handle work schedule around school pickup times considering travel time"""
        try:
            young_children = [c for c in self.family_members.get('children', [])
                           if isinstance(c, dict) and c.get('age', 0) < 15]
            
            if not young_children:
                return None

            children_schools = [c.get('workplace') for c in young_children if c.get('workplace')]
            workplace = self.workplaces[0] if self.workplaces else None

            # Calculate travel times (in hours)
            NORTH_SOUTH_TRAVEL_TIME = 0.5  # 30 minutes between north and south
            CENTRAL_TRAVEL_TIME = 0.25     # 15 minutes to/from central areas

            # Get workplace position
            workplace_position = self.get_location_position(workplace)
            
            # Calculate total commute time needed
            if workplace_position == "south":
                commute_time = NORTH_SOUTH_TRAVEL_TIME
            elif workplace_position == "central":
                commute_time = CENTRAL_TRAVEL_TIME
            else:
                commute_time = 0.25  # Default travel time

            # Leave work early enough to reach school by 15:00
            pickup_prep_time = 15 - commute_time
            
            # Pre-pickup preparation (leave early based on location)
            if pickup_prep_time <= hour < 15:
                return f"Leave work early for school pickup (traveling from {workplace_position})"

            # Pickup time (15:00)
            elif hour == 15:
                if children_schools:
                    return f"Pick up children from {', '.join(set(children_schools))}"

            # Post-pickup decision (15:00-16:00)
            elif 15 < hour < 16:
                # Don't return to work due to distance
                return "Take children home after pickup"

            # Late afternoon (16:00-18:00)
            elif 16 <= hour < 18:
                if self.location == self.residence:
                    occupation = str(self.occupation).lower()
                    if any(job in occupation for job in ['manager', 'analyst', 'engineer']):
                        return "Work from home while supervising children"
                    else:
                        return "Supervise children at home"

            return None

        except Exception as e:
            print(f"Error in handle_work_and_pickup for {self.name}: {str(e)}")
            return None

    def get_location_position(self, location_name):
        """Determine the position (north/south/central) of a location"""
        if not location_name:
            return None
            
        for area, info in town_data['world_layout']['major_areas'].items():
            if location_name in info['locations']:
                return info['position']
        return "central"  # Default to central if not found

    def is_working(self, hour):
        """Enhanced work check considering child pickup and travel times"""
        if not any(work in str(self.occupation).lower() 
                  for work in ['manager', 'worker', 'crew', 'supervisor']):
            return False

        workplace = self.workplaces[0] if self.workplaces else None
        if not workplace or workplace not in locations:
            return False

        # Check if it's pickup time for parents with young children
        if self.family_role == 'parent':
            young_children = [c for c in self.family_members.get('children', [])
                           if isinstance(c, dict) and c.get('age', 0) < 15]
            if young_children:
                workplace_position = self.get_location_position(workplace)
                # Calculate when parent needs to leave work
                if workplace_position == "south":
                    leave_time = 14.5  # Leave at 14:30 if working in south
                elif workplace_position == "central":
                    leave_time = 14.75  # Leave at 14:45 if working in central
                else:
                    leave_time = 14.75  # Default leave time
                    
                # Not working during pickup window
                if leave_time <= hour < 16:
                    return False

        # Regular work check
        if 'part-time' in str(self.occupation).lower():
            today_shift = next((shift for shift in self.part_time_shifts 
                              if shift['day'] == (hour // 24) % 7), None)
            return today_shift and today_shift['start'] <= hour % 24 < today_shift['end']
        
        return locations[workplace].is_open(hour)

    def generate_conversation(self, context):
        """Generate contextual conversation based on location and time"""
        try:
            location = context['location']
            hour = context['time'] % 24
            nearby_agents = context['nearby_agents']
            
            # Build conversation prompt based on context
            prompt = f"""You are {self.name} at {location}. It's {hour:02d}:00.
            You are with: {', '.join([agent.name for agent in nearby_agents])}
            Recent experiences: {context['recent_experiences']}
            
            Generate a brief, natural conversation considering:
            1. Time of day and location
            2. Your relationship with others present
            3. Recent experiences and shared activities
            4. Current activities or plans
            
            What do you say or discuss?"""
            
            conversation = generate(prompt_meta.format(prompt))
            return conversation
            
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return "Makes small talk"

    def evening_reflection(self, current_time):
        """Handle evening reflection and next day planning"""
        try:
            if current_time.hour == 22:  # 10 PM
            # Get today's experiences
            today_memories = self.memory_manager.get_today_memories(self.name)
            
            # Categorize memories
            dining_experiences = [m for m in today_memories if m['type'] in ['dining_experience', 'meal']]
            received_recommendations = [m for m in today_memories if m['type'] == 'received_recommendation']
            family_interactions = [m for m in today_memories if m['type'] == 'family_interaction']
                daily_activities = [m for m in today_memories if m['type'] in ['work', 'school', 'leisure', 'social']]
            
            # Generate reflection with specific focus on next day planning
            reflection_prompt = f"""You are {self.name}, reflecting on today's experiences:
            Dining experiences: {[exp['content'] for exp in dining_experiences]}
            Recommendations received: {[rec['content'] for rec in received_recommendations]}
            Family interactions: {[int['content'] for int in family_interactions]}
                Daily activities: {[act['content'] for act in daily_activities]}
            
            Consider:
            1. What worked well in today's schedule?
            2. What dining choices were good/bad?
            3. Which recommendations seem worth trying?
            4. What should be changed for tomorrow?
            5. What family coordination could be improved?
                6. Were there any missed opportunities for social interaction?
                7. How well were meals coordinated with family members?
            
            Generate a thoughtful reflection that will help plan tomorrow."""
            
            reflection = generate(reflection_prompt)
            
                # Analyze experiences and generate planning insights
                planning_insights = {
                        'dining_preferences': self.analyze_dining_experiences(dining_experiences),
                        'recommended_places': self.analyze_recommendations(received_recommendations),
                        'schedule_adjustments': self.analyze_schedule_effectiveness(family_interactions),
                    'family_coordination': self.evaluate_family_coordination(family_interactions),
                    'activity_satisfaction': self.analyze_daily_activities(daily_activities)
                }
                
                # Store reflection with planning implications
                self.memory_manager.add_memory(
                    agent_name=self.name,
                    memory_text=reflection,
                memory_type="evening_reflection",
                    timestamp=current_time,
                    importance=0.8,
                    details={
                        'planning_insights': planning_insights,
                        'next_day_considerations': True
                    }
                )
                
                # Use reflection to plan next day
                if hasattr(self, 'family_role') and self.family_role == 'parent':
                    # Parents create initial family plan based on reflection
                    next_day_plan = self.plan_next_day_parent(reflection, current_time)
                    self.next_day_family_schedule = next_day_plan
                    
                    # Store specific meal and activity preferences for morning discussion
                    self.next_day_preferences = {
                        'meals': self.plan_meals(planning_insights),
                        'activities': self.plan_leisure_activities(planning_insights),
                        'work_schedule': self.plan_work_schedule(current_time),
                        'family_coordination': planning_insights['family_coordination']
                    }
                    
                elif hasattr(self, 'age') and self.age < 18:
                    # Children/teens create their wishlist/preferences based on reflection
                    next_day_plan = self.plan_next_day_child(reflection, current_time)
                    self.next_day_preferences = {
                        'desired_activities': self.plan_leisure_activities(planning_insights),
                        'meal_preferences': planning_insights['dining_preferences'],
                        'social_wishes': self.plan_social_activities(planning_insights)
                    }
                    
    else:
                    # Independent adults plan based on reflection
                    next_day_plan = self.plan_next_day_individual(reflection, current_time)
                    self.next_day_preferences = {
                        'work_schedule': self.plan_work_schedule(current_time),
                        'meals': self.plan_meals(planning_insights),
                        'activities': self.plan_leisure_activities(planning_insights)
                    }
                
                # Store the complete planning package for morning discussion
                self.memory_manager.add_memory(
                    agent_name=self.name,
                    memory_text="Completed planning for next day",
                    memory_type="next_day_plan",
                    timestamp=current_time,
                    importance=0.9,
                    details={
                        'plan': next_day_plan,
                        'preferences': self.next_day_preferences,
                        'based_on_reflection': reflection
                    }
                )
                
                return "Completed evening reflection and planning"
                
        except Exception as e:
            print(f"Error in evening reflection for {self.name}: {str(e)}")
            return "Basic evening routine"

    def plan_next_day(self, reflection, current_time):
        """Generate next day's plan based on evening reflection"""
        try:
            # Get recent reflections and recommendations
            recent_reflection = self.memory_manager.get_recent_memories(
                self.name,
                memory_type="evening_reflection",
                limit=1
            )[0]
            
            planning_insights = recent_reflection.get('planning_insights', {})
            
            # Create base schedule
            schedule = self.generate_family_daily_schedule(current_time + 2)  # +2 to plan for next day
            
            # Modify schedule based on insights
            if planning_insights:
                # Adjust dining plans
                preferred_places = planning_insights.get('dining_preferences', {})
                recommended_places = planning_insights.get('recommended_places', [])
                
                # Modify meal plans based on preferences and recommendations
                schedule['meals'] = self.adjust_meal_plans(
                    schedule.get('meals', {}),
                    preferred_places,
                    recommended_places
                )
                
                # Adjust timing based on schedule effectiveness
                schedule_adjustments = planning_insights.get('schedule_adjustments', {})
                if schedule_adjustments:
                    schedule = self.apply_schedule_adjustments(schedule, schedule_adjustments)
            
            return schedule
            
        except Exception as e:
            print(f"Error in plan_next_day for {self.name}: {str(e)}")
            return None

    def adjust_meal_plans(self, meal_schedule, preferred_places, recommended_places):
        """Adjust meal plans based on preferences and recommendations"""
        adjusted_meals = meal_schedule.copy()
        
        # Prioritize highly-rated places for dining out
        good_options = [place for place, rating in preferred_places.items() if rating >= 4]
        good_options.extend(recommended_places)
        
        # Modify dinner plans if good options available
        if good_options and self.grocery_level < 30:
            adjusted_meals['dinner'] = {
                'type': 'dine_out',
                'location': random.choice(good_options),
                'reason': 'positive_experience' if good_options[0] in preferred_places else 'recommendation'
            }
        
        return adjusted_meals

    def share_daily_plan(self, current_time):
        """Share next day's plan with family in the morning"""
        if not hasattr(self, 'next_day_plan'):
            self.next_day_plan = self.generate_family_daily_schedule(current_time)
        
        # Generate conversation including insights from evening reflection
        recent_reflection = self.memory_manager.get_recent_memories(
            self.name,
            memory_type="evening_reflection",
            limit=1
        )
        
        conversation_context = {
            'time': current_time,
            'location': self.residence,
            'schedule': self.next_day_plan,
            'reflection_insights': recent_reflection[0] if recent_reflection else None,
            'present_family': [
                member for member_type in ['parents', 'children']
                for member in self.family_members.get(member_type, [])
                if isinstance(member, str) and 
                agents[member].location == self.residence
            ]
        }
        
        # Generate and share morning conversation
        morning_conversation = self.generate_family_conversation(conversation_context)
        
        # Record the conversation and schedule for all present family members
        for member in conversation_context['present_family']:
            agents[member].receive_family_schedule(
                self.next_day_plan,
                current_time,
                morning_conversation
            )

    def analyze_dining_experiences(self, dining_experiences):
        """Analyze dining experiences to inform future choices"""
        preferences = {}
        for exp in dining_experiences:
            location = exp.get('location')
            if location:
                satisfaction = exp.get('satisfaction', 0)
                energy_restored = exp.get('energy_restored', False)
                cost = exp.get('price', 0)
                
                # Calculate overall rating
                rating = 0
                if satisfaction > 0:
                    rating += satisfaction * 0.4  # 40% weight
                if energy_restored:
                    rating += 3  # Base points for fulfilling need
                if cost > 0:
                    value_rating = min(5, (50 / cost) * 5)  # Value for money
                    rating += value_rating * 0.2  # 20% weight
                
                preferences[location] = round(rating, 2)
        
        return preferences

    def analyze_recommendations(self, recommendations):
        """Analyze received recommendations for planning"""
        recommended_places = []
        for rec in recommendations:
            if rec.get('sentiment') == 'positive':
                place = rec.get('location')
                if place and place not in recommended_places:
                    recommended_places.append(place)
        return recommended_places

    def analyze_schedule_effectiveness(self, family_interactions):
        """Analyze schedule effectiveness from family interactions"""
        adjustments = {}
        for interaction in family_interactions:
            if 'delayed' in str(interaction.get('content', '')).lower():
                time = interaction.get('time')
                if time:
                    hour = time % 24
                    adjustments[hour] = 'needs_more_time'
        return adjustments

    def plan_work_schedule(self, current_time):
        """Plan work schedule based on occupation and hours"""
        try:
            if 'part-time' in str(self.occupation).lower():
                return self.plan_part_time_work(current_time)
            elif self.workplaces:
                workplace = self.workplaces[0]
                if workplace in locations:
                    return {
                        'location': workplace,
                        'hours': locations[workplace].get_current_hours(current_time)
                    }
            return None
        except Exception as e:
            print(f"Error planning work schedule for {self.name}: {str(e)}")
            return None

    def plan_meals(self, planning_insights):
        """Plan meals based on preferences and recommendations"""
        meals = {}
        dining_preferences = planning_insights.get('dining_preferences', {})
        recommended_places = planning_insights.get('recommended_places', [])
        
        # Plan each meal
        for meal_type, (start, end) in self.meal_schedule.items():
            if self.grocery_level >= 30 and meal_type in ['breakfast', 'dinner']:
                meals[meal_type] = {
                    'type': 'home_meal',
                    'time': (start, end),
                    'location': self.residence
                }
            else:
                # Choose dining location based on preferences
                preferred_places = [place for place, rating in dining_preferences.items() 
                                 if rating >= 4 and place in locations]
                if preferred_places or recommended_places:
                    chosen_place = random.choice(preferred_places or recommended_places)
                    meals[meal_type] = {
                        'type': 'dine_out',
                        'time': (start, end),
                        'location': chosen_place
                    }
                else:
                    meals[meal_type] = {
                        'type': 'find_food',
                        'time': (start, end)
                    }
        
        return meals

    def plan_social_activities(self, planning_insights):
        """Plan social activities based on recent interactions"""
        social_activities = []
        recent_positive_interactions = planning_insights.get('positive_interactions', [])
        
        if recent_positive_interactions:
            # Plan to meet with people from positive interactions
            for interaction in recent_positive_interactions[:2]:  # Limit to 2 social activities
                social_activities.append({
                    'type': 'social_meeting',
                    'with': interaction.get('person'),
                    'location': interaction.get('location'),
                    'time': self.find_free_time_slot()
                })
        
        return social_activities

    def plan_leisure_activities(self, planning_insights):
        """Plan leisure activities based on preferences"""
        leisure_activities = []
        preferred_activities = planning_insights.get('enjoyed_activities', [])
        
        if preferred_activities:
            activity = random.choice(preferred_activities)
            leisure_activities.append({
                'type': 'leisure',
                'activity': activity,
                'time': self.find_free_time_slot(evening=True)
            })
        
        return leisure_activities

    def find_free_time_slot(self, evening=False):
        """Find available time slot for activities"""
        if evening:
            return (18, 21)  # Evening slot
        return (16, 18)  # Afternoon slot

    def review_and_adjust_plan(self, current_time):
        """Review and adjust plan based on family schedule and personal preferences"""
        try:
            if not hasattr(self, 'next_day_plan'):
                return
            
            # Get family schedule if part of family
            if self.family_unit:
                family_schedule = self.family_schedule if hasattr(self, 'family_schedule') else None
                if family_schedule:
                    # Adjust personal activities around family obligations
                    self.next_day_plan = self.merge_schedules(
                        self.next_day_plan,
                        family_schedule
                    )
            
            # Record final adjusted plan
            self.memory_manager.add_memory(
                self.name,
                "daily_planning",
                {
                    'type': 'adjusted_plan',
                    'plan': self.next_day_plan,
                    'time': current_time
                }
            )
            
        except Exception as e:
            print(f"Error adjusting plan for {self.name}: {str(e)}")

    def merge_schedules(self, personal_plan, family_plan):
        """Merge personal and family schedules, prioritizing family obligations"""
        merged = family_plan.copy()
        
        # Add personal activities that don't conflict with family obligations
        for activity_type, activities in personal_plan.items():
            if activity_type not in merged:
                merged[activity_type] = activities
            elif isinstance(activities, list):
                # Add non-conflicting activities
                for activity in activities:
                    if not self.time_conflicts(activity, merged):
                        merged[activity_type].append(activity)
        
        return merged

    def time_conflicts(self, activity, schedule):
        """Check if activity conflicts with existing schedule"""
        activity_time = activity.get('time')
        if not activity_time:
            return False
        
        for scheduled_items in schedule.values():
            if isinstance(scheduled_items, list):
                for item in scheduled_items:
                    if item.get('time') == activity_time:
                        return True
            elif isinstance(scheduled_items, dict):
                if scheduled_items.get('time') == activity_time:
                    return True
        
        return False

    def plan_next_day_individual(self, reflection, current_time):
        """Plan next day for individual agents"""
        try:
            # Get insights from reflection
            if not reflection:
                return None
                
            # Create basic schedule
            next_day_plan = {
                'meals': self.plan_meals(reflection),
                'work': self.plan_work_schedule(current_time),
                'social': self.plan_social_activities(reflection),
                'leisure': self.plan_leisure_activities(reflection)
            }
            
            return next_day_plan
            
        except Exception as e:
            print(f"Error in individual planning for {self.name}: {str(e)}")
            return None

    def plan_next_day_parent(self, reflection, current_time):
        """Plan next day for parent agents"""
        try:
            # Start with family schedule
            next_day_plan = self.generate_family_daily_schedule(current_time)
            
            # Add personal activities around family obligations
            if reflection:
                next_day_plan.update({
                    'personal': {
                        'meals': self.plan_meals(reflection),
                        'work': self.plan_work_schedule(current_time),
                        'leisure': self.plan_leisure_activities(reflection)
                    }
                })
            
            return next_day_plan
            
        except Exception as e:
            print(f"Error in parent planning for {self.name}: {str(e)}")
            return None

    def plan_next_day_child(self, reflection, current_time):
        """Plan next day for child agents"""
        try:
            # Start with school/study schedule
            next_day_plan = {
                'education': {
                    'location': self.school_location,
                    'hours': (8, 15) if self.needs_supervision else (9, 16)
                }
            }
            
            # Add activities around school
            if reflection:
                next_day_plan.update({
                    'meals': self.plan_meals(reflection),
                    'social': self.plan_social_activities(reflection),
                    'leisure': self.plan_leisure_activities(reflection)
                })
            
            return next_day_plan
            
        except Exception as e:
            print(f"Error in child planning for {self.name}: {str(e)}")
            return None

    def handle_breakfast(self, current_time):
        """Handle individual breakfast routine"""
        try:
            # First check if we've already had breakfast today
            today_memories = self.memory_manager.get_memories_for_day(current_time)
            already_had_breakfast = any(
                "breakfast" in memory.lower() 
                for memory in today_memories
            )
            
            if already_had_breakfast:
                return "Already had breakfast today"

            # Check if agent has enough groceries (10 per meal)
            if hasattr(self, 'grocery_level') and self.grocery_level >= 10:
                self.grocery_level -= 10  # Use groceries for breakfast
                self.energy = min(100, self.energy + 30)  # Breakfast provides energy
                
            self.memory_manager.add_memory(
                agent_name=self.name,
                    memory_text="Had breakfast at home",
                memory_type="meal",
                timestamp=current_time,
                    importance=0.6,
                    details={
                        'location': self.residence,
                        'energy_gained': 30,
                        'groceries_used': 10
                    }
                )
                return "Having breakfast at home"
    else:
                # Look for breakfast options nearby
                nearby_food = self.check_nearby_food_locations(current_time)
                if nearby_food:
                    # Record the out-of-home breakfast
                    self.memory_manager.add_memory(
                        agent_name=self.name,
                        memory_text=f"Had breakfast at {nearby_food}",
                        memory_type="meal",
                        timestamp=current_time,
                        importance=0.6,
                        details={
                            'location': nearby_food,
                            'reason': 'insufficient_groceries'
                        }
                    )
                    return nearby_food
                
                # If no options available, record the missed breakfast
                self.memory_manager.add_memory(
                    agent_name=self.name,
                    memory_text="Missed breakfast due to no food available",
                    memory_type="missed_meal",
                    timestamp=current_time,
                    importance=0.7,
                    details={
                        'reason': 'no_food_available',
                        'grocery_level': getattr(self, 'grocery_level', 0)
        return 1000  # Default if unknown income type

# Then the initialize_agents function that uses it
def initialize_agents(town_data, experiment_settings, metrics, memory_mgr, locations):
    agents = {}
    
    # Process family units
    for family_name, family_data in town_data['family_units'].items():
        # Initialize parents
        for parent in family_data['members']['parents']:
            agent = Agent(
                name=parent['name'],
                location=family_data['residence'],
                money=calculate_initial_money(parent['income']),
                occupation=parent['occupation'],
                experiment_settings=experiment_settings,
                metrics=metrics,
                memory_manager=memory_mgr,
                locations=locations
            )
            agents[parent['name']] = agent
            
        # Initialize children
        for child in family_data['members'].get('children', []):
            agent = Agent(
                name=child['name'],
                location=family_data['residence'],
                money=calculate_initial_money(child.get('income', None)),  # Some older children might have income
                occupation=child['occupation'],
                experiment_settings=experiment_settings,
                metrics=metrics,
                memory_manager=memory_mgr,
                locations=locations
            )
            agents[child['name']] = agent

    # Initialize individual town people
    for person_name, person_data in town_data.get('town_people', {}).items():
        basics = person_data['basics']
        agent = Agent(
            name=person_name,
            location=basics['residence'],
            money=calculate_initial_money(basics['income']),  # Also fix this to use calculate_initial_money
            occupation=basics['occupation'],
            experiment_settings=experiment_settings,
            metrics=metrics,
            memory_manager=memory_mgr,
            locations=locations  # Add this line
        )
        agent.description = basics
        agent.residence = basics['residence']
        agent.workplace = basics['workplace']
        agent.income = basics['income']
        agents[person_name] = agent

    return agents

# 2. Load configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, 'Stability_Agents_Config.json')

print(f"Loading config from: {config_path}")
with open(config_path, 'r') as f:
    town_data = json.load(f)

# 3. Initialize locations
locations = {}
for category, category_locations in town_data['town_areas'].items():
    for location_name, info in category_locations.items():
        if isinstance(info, str):
            info = {
                "description": info,
                "type": category,
                "hours": {"open": 0, "close": 24}
            }
        locations[location_name] = Location(location_name, info)

print(f"Initialized locations: {list(locations.keys())}")
print(f"Loaded locations: {list(town_data['town_areas'].keys())}")

# 4. Create metrics object
fried_chicken_location = locations.get("Fried Chicken Shop")
discount_settings = getattr(fried_chicken_location, 'discount', None)
metrics = FriedChickenMetrics(settings=discount_settings)

# 5. Create memory manager
memory_mgr = MemoryManager()

# 6. NOW initialize agents (after everything is defined)
agents = initialize_agents(town_data, experiment_settings, metrics, memory_mgr, locations)

# Initialize households after agents are created
households = {}
for agent_name, agent in agents.items():  # Change to iterate over items() since agents is a dictionary
    if not isinstance(agent, Agent):  # Skip if not an Agent object
        continue
        
    residence = agent.residence
    if residence not in households:
        households[residence] = {'members': [], 'money': 0}
    households[residence]['members'].append(agent)
    
    # Add income if agent has one
    if hasattr(agent, 'description') and agent.description:
        income = agent.description.get('income', {})
        if income:
            if income.get('type') == 'monthly':
                households[residence]['money'] += income.get('amount', 0)
            elif income.get('type') == 'annual':
                households[residence]['money'] += income.get('amount', 0) // 12
            elif income.get('type') == 'hourly':
                households[residence]['money'] += income.get('amount', 0) * 160  # Assuming 160 hours per month
            elif income.get('type') == 'pension':
                households[residence]['money'] += income.get('amount', 0)

# Main simulation loop
def run_simulation():
    try:
        global global_time
        whole_simulation_output = ""
        
        print("\n=== Starting Simulation ===")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        
        total_hours = 163  # Total simulation hours
        first_day = True
        
        while global_time < total_hours:
            current_day = (global_time // 24) + 1
            current_hour = global_time % 24
            
            # Print current timestamp
            print(f"\nDay {current_day}, Hour {current_hour:02d}:00")
            
            # Skip early hours only on first day
            if first_day and current_hour < 5:
                global_time += 1
                continue
            
            if current_hour == 0:
                first_day = False
            
            # Evening reflections and planning for ALL agents at 22:00
            if current_hour == 22:
                print(f"Day {current_day} - Evening reflection time")
                for agent in agents.values():
                    # Everyone does evening reflection
                    reflection = agent.evening_reflection(global_time)
                    
                    # Different planning based on agent type
                    if agent.family_role == 'parent':
                        next_day_plan = agent.plan_next_day_parent(reflection, global_time)
                    elif agent.family_role == 'child':
                        next_day_plan = agent.plan_next_day_child(reflection, global_time)
                    else:
                        next_day_plan = agent.plan_next_day_individual(reflection, global_time)
                    
                    agent.store_next_day_plan(next_day_plan, global_time)

            # Morning coordination at 5:00
            elif current_hour == 5:
                print(f"Day {current_day} - Morning coordination time")
                for agent in agents.values():
                    if agent.family_role == 'parent':
                        agent.share_daily_plan(global_time)
                    else:
                        agent.review_and_adjust_plan(global_time)
            
            # Regular hour processing...
            for agent in agents.values():
                try:
                    plan = agent.plan(global_time)
                    action = agent.execute_action(agents.values(), agent.location, global_time)
                    # Log action if enabled
                    if log_actions:
                        print(f"{agent.name} at {current_hour:02d}:00: {action}")
                except Exception as e:
                    print(f"Error processing agent {agent.name} at Day {current_day}, Hour {current_hour:02d}:00: {str(e)}")
                    continue  # Continue with next agent even if one fails
            
            global_time += 1
            
        return whole_simulation_output
        
    except Exception as e:
        print(f"Critical simulation error at Day {global_time//24 + 1}, Hour {global_time%24:02d}:00: {str(e)}")
        return str(e)

# Add validation for experiment settings
def validate_experiment_settings():
    required_fields = ['simulation', 'fried_chicken_shop', 'discount_settings']
    with open('experiment_settings.json', 'r') as f:
        settings = json.load(f)
        
    for field in required_fields:
        if field not in settings:
            raise ValueError(f"Missing required field in experiment_settings.json: {field}")
    
    return settings

# Add to main simulation
if __name__ == "__main__":
    try:
        print("\n=== Starting Simulation ===")
        print(f"Simulation started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Initialize agents as a dictionary
        agents = initialize_agents(town_data, experiment_settings, metrics, memory_mgr, locations)
        
        # Initialize households using the agents dictionary
        households = {}
        for agent_name, agent in agents.items():
            residence = agent.residence
            if residence not in households:
                households[residence] = {'members': [], 'money': 0}
            households[residence]['members'].append(agent)
            
            # Add income if agent has income
            if hasattr(agent, 'income'):
                income_data = agent.income
                if income_data['type'] == 'monthly':
                    households[residence]['money'] += income_data['amount']
                elif income_data['type'] == 'annual':
                    households[residence]['money'] += income_data['amount'] / 12
                elif income_data['type'] == 'hourly':
                    hours = income_data.get('hours_per_week', 40)
                    households[residence]['money'] += income_data['amount'] * hours * 4
        
        run_simulation()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        if 'memory_mgr' in locals():
            memory_file = memory_mgr.save_to_file()
        if 'metrics' in locals():
            metrics_file = metrics.save_metrics()

# At the start of the simulation, after imports
def load_configuration():
    """Single source for loading and validating configuration"""
    try:
        # Load experiment settings
        with open('experiment_settings.json', 'r') as f:
            experiment_settings = json.load(f)
            
        # Load town configuration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'Stability_Agents_Config.json')
        with open(config_path, 'r') as f:
            town_data = json.load(f)
            
        # Validate configuration
        validate_configuration(town_data, experiment_settings)
        
        return experiment_settings, town_data
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

def generate_daily_summary(self, current_day):
    """Generate a comprehensive daily summary"""
    summary = f"\n=== Daily Summary for Day {current_day} ===\n"
    
    # Location activity summary
    summary += "\nLocation Activity:\n"
    for location_name, location in self.locations.items():
        if hasattr(location, 'daily_visitors'):
            unique_visitors = len(set(location.daily_visitors))
            total_visits = len(location.daily_visitors)
            summary += f"{location_name}: {unique_visitors} unique visitors, {total_visits} total visits\n"
    
    # Family activity summary
    summary += "\nFamily Activity:\n"
    for family_name, family in self.family_units.items():
        summary += f"\n{family_name}:"
        for member_type in ['parents', 'children']:
            if member_type in family['members']:
                for member in family['members'][member_type]:
                    if isinstance(member, dict):
                        member_name = member['name']
                    else:
                        member_name = member
                    if member_name in self.agents:
                        agent = self.agents[member_name]
                        summary += f"\n  - {member_name}: Last location: {agent.current_location}"

    # Reset daily tracking
    for location in self.locations.values():
        if hasattr(location, 'daily_visitors'):
            location.daily_visitors = []

    print(summary)
    return summary

def run_simulation(self):
    sim_data = initialize_simulation()
    locations = sim_data['locations']
    agents = sim_data['agents']
    metrics = sim_data['metrics']
    settings = sim_data['settings']
    
    global_time = 5  # Start at 5 AM
    
    while global_time < (settings['duration_days'] * 24):
        current_day = global_time // 24
        
        for agent in agents.values():
            # Pass required parameters to methods
            if global_time % 24 == 5:  # 5 AM
                agent.create_daily_plan(global_time)
            
            # Pass all required parameters to execute_action
            agent.execute_action(agents, agent.location, global_time)
            
            # Evening reflection needs all context
            if global_time % 24 == 22:  # 10 PM
                reflection = agent.evening_reflection(global_time)
                next_day_plan = agent.plan_next_day(reflection, global_time)

def initialize_simulation():
    # Load configurations first
    config = load_configuration()
    experiment_settings = config['simulation']
    location_settings = config['location_prices']
    
    # Initialize core components with settings
    memory_mgr = MemoryManager()
    metrics = FriedChickenMetrics(settings=location_settings['dining']['Fried Chicken Shop'])
    
    # Initialize locations with settings
    locations = {}
    with open('LLMAgentsTown_experiment/Stability_Agents_Config.json') as f:
        town_data = json.load(f)
        for category, category_locations in town_data['town_areas'].items():
            for location_name, info in category_locations.items():
                locations[location_name] = Location(location_name, info)
    
    # Initialize agents with required parameters
    agents = initialize_agents(town_data, experiment_settings, metrics, memory_mgr, locations)
    
    return {
        'locations': locations,
        'agents': agents,
        'metrics': metrics,
        'memory_mgr': memory_mgr,
        'settings': experiment_settings
    }

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
global_time = 5  # Start at 5:00 AM
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
    def __init__(self, name, info):
        self.name = name
        # Handle both object and string formats
        if isinstance(info, str):
            self.description = info
            self.type = "default"
            if name == "Fried Chicken Shop":
                self.hours = {"open": 10, "close": 22}  # 10 AM to 10 PM
            elif name == "The Coffee Shop":
                self.hours = {"open": 7, "close": 20}   # 7 AM to 8 PM
            else:
                self.hours = {"open": 0, "close": 24}
            self.base_price = 0.0
        else:
            self.description = info.get("description", "")
            self.type = info.get("type", "default")
            if name == "Fried Chicken Shop":
                self.hours = {"open": 10, "close": 22}  # 10 AM to 10 PM
            elif name == "The Coffee Shop":
                self.hours = {"open": 7, "close": 20}   # 7 AM to 8 PM
            else:
                self.hours = info.get("hours", {"open": 0, "close": 24})
            self.base_price = info.get("base_price", 0.0)
            if "discount" in info:
                self.discount = info["discount"]
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)

    def is_open(self, current_time):
        hour = int(current_time) % 24
        return self.hours["open"] <= hour < self.hours["close"]

    def get_current_hours(self, current_time):
        day = int(current_time) // 24
        is_weekend = (day % 7) in [5, 6]
        return self.hours.get("weekend" if is_weekend else "weekday", self.hours)

class Agent:
    def __init__(self, name, location, money, occupation, experiment_settings, metrics, memory_manager, locations):
        # Initialize basic attributes first
        self.name = name
        self.location = location
        self.money = money
        self.occupation = occupation
        self.experiment_settings = experiment_settings
        self.metrics = metrics
        self.memory_manager = memory_manager
        self.locations = locations  # Store locations dictionary
        self.plans = []  # Initialize empty plans list
        
        # Initialize optional attributes with defaults
        self.description = ""  # Add this line before checking self.description
        self.schedule = {}
        self.family = []
        self.children = []
        self.parents = []
        self.energy = 100
        self.last_meal_time = 0
        self.grocery_level = 100
        self.daily_plan = {}  # This is initialized as a dictionary
        
        # Now we can safely use locations
        self.residence = location if locations.get(location) and locations[location].type == 'residence' else None
        
        # Get settings directly from experiment_settings
        self.settings = experiment_settings
        self.duration_days = self.settings['simulation']['duration_days']
        first_day_hours = 24 - 5  # Starting at 5 AM
        remaining_days_hours = (self.duration_days - 1) * 24
        self.simulation_duration = first_day_hours + remaining_days_hours
        
        # Initialize basic attributes
        self.last_purchase_time = None
        self.daily_plan = []
        self.plans = []
        self.current_action = None
        self.location_last_purchase = {}  # Track last purchase time for each location
        
        # Enhanced education attributes
        self.is_student = False
        self.student_type = None  # 'high_school', 'college', 'part_time_student'
        self.school_location = None
        self.workplaces = []
        
        # Enhanced child/student handling
        self.needs_supervision = False
        
        # Now check description after it's initialized
        if hasattr(self, 'description') and self.description:
            age = self.description.get('age', 0)
            if age < 11:  # Elementary school (K-5)
                self.needs_supervision = True
                self.student_type = 'elementary'
                self.school_location = 'Town Elementary School'
            elif age < 14:  # Middle school (6-8)
                self.needs_supervision = True
                self.student_type = 'middle_school'
                self.school_location = 'Town Middle School'
            elif 'student' in self.occupation.lower():
                self.is_student = True
                if age <= 18:  # High school (9-12)
                    self.student_type = 'high_school'
                    self.school_location = 'Town Public High School'
                else:  # College
                    self.student_type = 'college'
                    self.school_location = 'Town Community College'
        
        # Parse person information
        if self.description:
            age = self.description.get('age', 0)
            occupation = self.description.get('occupation', '').lower()
            
            # Handle multiple workplaces
            workplace = self.description.get('workplace')
            if isinstance(workplace, list):
                self.workplaces = workplace
            else:
                self.workplaces = [workplace] if workplace else []
            
            # Determine student status
            if 'student' in occupation:
                self.is_student = True
                if 'high school' in occupation or age <= 18:
                    self.student_type = 'high_school'
                    self.school_location = 'Town Public High School'
                elif 'college' in occupation or 'evening student' in occupation:
                    self.student_type = 'college' if 'part-time' not in occupation else 'part_time_student'
                    self.school_location = 'Town Community College'
        
        # Debug info (only once)
        if not hasattr(Agent, '_debug_printed'):
            print("\n=== Simulation Configuration ===")
            print(f"Duration days: {self.duration_days}")
            print(f"First day hours: {first_day_hours}")
            print(f"Remaining days hours: {remaining_days_hours}")
            print(f"Total simulation duration: {self.simulation_duration}")
            print("===============================\n")
            Agent._debug_printed = True
        
        # Add family-related attributes
        self.family_unit = None
        self.family_role = None  # 'parent' or 'child'
        self.family_members = {}
        self.shared_income = False
        
        # Initialize family relationships if they exist
        if town_data.get('family_units'):
            for family_name, family_info in town_data['family_units'].items():
                if name in family_info['members']['parents']:
                    self.family_unit = family_name
                    self.family_role = 'parent'
                    self.family_members = family_info['members']
                    self.shared_income = True
                elif name in family_info['members']['children']:
                    self.family_unit = family_name
                    self.family_role = 'child'
                    self.family_members = family_info['members']
                    self.shared_income = True
        
        # Replace hunger tracking with energy tracking
        self.energy_level = 100  # Start with full energy
        self.last_meal_time = None
        self.meal_schedule = {
            'breakfast': (6, 9),    # 6:00-9:00
            'lunch': (11, 14),      # 11:00-14:00
            'dinner': (17, 20)      # 17:00-20:00
        }
        
        # Add part-time work schedule attributes
        self.part_time_shifts = []
        if 'part-time' in str(occupation).lower():
            self.initialize_part_time_schedule()
        
        # Validate school location exists in town_areas
        if self.school_location and self.school_location not in locations:
            print(f"Warning: Invalid school location '{self.school_location}' for {self.name}")
            # Try to find correct school based on age
            if hasattr(self, 'description') and self.description:
                age = self.description.get('age', 0)
                if age < 11:
                    self.school_location = 'Town Elementary School'
                elif age < 14:
                    self.school_location = 'Town Middle School'
                elif age <= 18:
                    self.school_location = 'Town Public High School'

    def initialize_part_time_schedule(self):
        """Initialize random part-time work schedule (5 shifts x 4 hours = 20 hours/week)"""
        possible_shifts = [
            (9, 13),   # Morning shift
            (13, 17),  # Afternoon shift
            (17, 21)   # Evening shift
        ]
        
        # Randomly assign 5 shifts for the week (max 20 hours)
        available_days = list(range(7))  # 0-6 for week days
        random.shuffle(available_days)
        work_days = available_days[:5]  # Pick 5 random days
        
        for day in work_days:
            shift = random.choice(possible_shifts)
            self.part_time_shifts.append({
                'day': day,
                'start': shift[0],
                'end': shift[1]
            })

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
        
        # Update memory importance based on reflection insights
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
        try:
            hour = int(current_time) % 24
            new_plan = self.create_daily_plan(hour)
            
            # Initialize daily_plan as a list if it doesn't exist
            if not hasattr(self, 'daily_plan'):
                self.daily_plan = []
            elif isinstance(self.daily_plan, dict):
                # Convert existing dict to list if needed
                self.daily_plan = [self.daily_plan] if self.daily_plan else []
                
            self.daily_plan.append(new_plan)
            return new_plan
            
        except Exception as e:
            print(f"Error in plan for {self.name}: {str(e)}")
            return "Stay at current location"

    def create_daily_plan(self, current_time):
        """Create a daily plan based on agent's attributes from config"""
        try:
            hour = int(current_time) % 24
            day = current_time // 24 + 1
            plan = {}

            # Basic schedule based on age and occupation
            if hasattr(self, 'age'):
                if self.age < 18:  # School-age
                    # School schedule (9:00-15:00)
                    if hasattr(self, 'school_location'):
                        for h in range(9, 15):
                            plan[str(h)] = f"Attend classes at {self.school_location}"
                        
                        # Different schedules based on age
                        if self.age <= 12:  # Young children
                            plan['7'] = "Get ready for school with parent's help"
                            plan['8'] = f"Go to {self.school_location} with parent"
                            plan['15'] = "Wait for parent pickup from school"
                        elif self.age < 15:  # Supervised teens
                            plan['7'] = "Prepare for school with some parent supervision"
                            plan['8'] = f"Go to {self.school_location} with parent"
                            plan['15'] = "Return home with parent or approved method"
                        else:  # Independent teens
                            plan['7'] = "Prepare for school independently"
                            plan['8'] = f"Go to {self.school_location} independently"
                            plan['15'] = "Return home from school"

            # Work schedule for adults
            if hasattr(self, 'workplaces') and self.workplaces:
                workplace = self.workplaces[0]  # Primary workplace
                plan['8'] = f"Go to work at {workplace}"
                for h in range(9, 17):  # Standard work hours
                    plan[str(h)] = f"Work at {workplace}"
                plan['17'] = "Return home from work"

            # Family role specific planning
            if hasattr(self, 'family_role') and self.family_role == 'parent':
                if day == 1:  # First day morning planning
                    plan['5'] = "Create and share family schedule"
                else:
                    plan['22'] = "Plan next day with family"
                
                plan['6'] = "Prepare and have family breakfast"
                plan['7'] = "Help children prepare for school"
                
                # Add school transport duties if have young children
                if hasattr(self, 'family_members') and 'children' in self.family_members:
                    young_children = [child for child in self.family_members['children'] 
                                    if isinstance(child, dict) and child.get('age', 18) < 15]
                if young_children:
                        plan['8'] = "Take children to school"
                        plan['15'] = "Pick up children from school"

            # Meal times
            if not '6' in plan:  # If breakfast not already planned
                plan['6'] = "Have breakfast"
            plan['12'] = "Lunch break"
            plan['18'] = "Dinner time"

            # Evening activities
            if day > 1:  # Regular days end with reflection and planning
                plan['21'] = "Evening free time"
                plan['22'] = "Reflect on day and plan tomorrow"

            self.plans.append(plan)
            return plan

        except Exception as e:
            print(f"Error creating daily plan for {self.name}: {str(e)}")
            return self.create_basic_plan()

    def create_basic_plan(self):
        """Create a basic fallback plan if detailed planning fails"""
        basic_plan = {
            '6': "Wake up and have breakfast",
            '12': "Lunch break",
            '18': "Dinner time",
            '22': "Prepare for next day"
        }
        self.plans.append(basic_plan)
        return basic_plan

    def review_daily_plan(self, current_time):
        """Review and adjust daily plan based on current circumstances"""
        try:
            if not hasattr(self, 'plans') or not self.plans:
                self.create_daily_plan(current_time)
                
            current_plan = self.plans[-1]
            
            # Record the planning activity
            self.memory_manager.add_memory(
                self.name,
                "daily_planning",
                {
                    'activity': "Daily planning",
                    'time': current_time,
                    'plan': current_plan,
                    'is_first_day': current_time < 24,
                    'coordinated_with_family': hasattr(self, 'family_unit') and self.family_unit
                }
            )

            if hasattr(self, 'family_unit') and self.family_unit:
                return "Reviewing daily plan with family schedule"
            return "Reviewing personal daily plan"

        except Exception as e:
            print(f"Error in review_daily_plan for {self.name}: {str(e)}")
            return "Created basic daily plan"

    def choose_leisure_activity(self, hour):
        """Choose a leisure activity based on available options and preferences"""
        try:
            # Get open community locations
            community_locations = [loc for loc in locations.values() 
                                 if loc.type == 'community' and 
                                 loc.is_open(hour) and 
                                 loc.name in locations]
            
            if not community_locations:
                return "Spend free time at home"

            # Check energy level - tired people prefer relaxing activities
            is_tired = self.energy_level < 40
            is_good_hour = 6 <= hour <= 21  # Reasonable hours for outdoor activities
            
            # Higher chance of going out if:
            # - It's good hours
            # - Not too tired
            # - Weather is good (could be expanded)
            should_go_out = (
                is_good_hour and 
                not is_tired and 
                random.random() < 0.7  # 70% chance if conditions are good
            )

            if should_go_out:
                chosen_location = random.choice(community_locations)
                
                if chosen_location.name == "City Park":
                    if is_tired:
                        activities = ["Relax in the park", "Read a book", "Meet friends"]
                    else:
                        activities = ["Go for a walk", "Have a picnic", "Play outdoor games"]
                    
                elif chosen_location.name == "Community Sports Complex Center":
                    if is_tired:
                        activities = ["Light exercise", "Use sports facilities"]
                    else:
                        activities = ["Exercise", "Play sports", "Join fitness class"]
                    
                activity = random.choice(activities)
                return f"{activity} at {chosen_location.name}"
                
            return "Spend free time at home"

        except Exception as e:
            print(f"Error choosing leisure activity for {self.name}: {str(e)}")
            return "Stay at current location"

    def is_school_time(self, hour):
        """Check if it's school hours"""
        return (
            self.is_student and 
            8 <= hour < 15 and 
            self.workplace in locations and 
            locations[self.workplace].is_open(hour)
        )

    def is_working(self, hour):
        """Check if agent should be working at this hour"""
        if not any(work in str(self.occupation).lower() 
                  for work in ['manager', 'worker', 'crew', 'supervisor']):
            return False

        if 'part-time' in str(self.occupation).lower():
            today_shift = next((shift for shift in self.part_time_shifts 
                              if shift['day'] == (hour // 24) % 7), None)
            return today_shift and today_shift['start'] <= hour % 24 < today_shift['end']
        
        workplace = self.workplaces[0] if self.workplaces else None
        return workplace and workplace in locations and locations[workplace].is_open(hour)

    def execute_action(self, all_agents, current_location, current_time):
        try:
            # 1. First validate location
            current_loc = self.locations.get(current_location)
            if current_loc and not current_loc.is_open(current_time):
                return f"Cannot act at {current_location} - location is closed"

            hour = int(current_time) % 24
            
            # 2. Find nearby agents for social context
            nearby_agents = [
                agent for agent in all_agents 
                if agent.location == current_location 
                and agent.name != self.name
            ]

            # 3. Morning Routine Priority (5:00-9:00)
            if 5 <= hour <= 8:
                morning_action = self.handle_morning_routine(hour, current_time, all_agents)
                if morning_action:
                    return morning_action

            # 4. Build context
            context = self.build_action_context(current_location, hour, nearby_agents)
            
            # 5. Generate action based on context
            action = self.generate_contextual_action(context, current_time)
            
            # 6. Process specific location actions
            location_action = self.process_location_specific_actions(action, current_time)
            if location_action:
                return location_action

            # 7. Handle food needs
            if self.needs_food(current_time):
                food_action = self.handle_food_needs(current_time)
                if food_action:
                    return food_action

            # 8. Process social interactions
            if nearby_agents:
                social_result = self.social_interaction(all_agents, action, current_time)
                if social_result:
                    return social_result

            return action

        except Exception as e:
            print(f"Error executing action for {self.name}: {str(e)}")
            return f"Stayed at {current_location}"

    def handle_morning_routine(self, hour, current_time, all_agents):
        try:
            # Find nearby family members
            family_members_present = [
                agent for agent in all_agents 
                if hasattr(agent, 'family_unit') and 
                agent.family_unit == self.family_unit and
                agent.location == self.location
            ]

            is_young_child = hasattr(self, 'age') and self.age <= 12
            is_supervised_teen = hasattr(self, 'age') and 12 < self.age < 15
            is_independent_teen = hasattr(self, 'age') and 15 <= self.age < 18
            is_parent = hasattr(self, 'family_role') and self.family_role == 'parent'

            if hour == 5:  # Early morning
                if current_time.day == 1:  # Only on first day
                    if is_parent:
                        return self.handle_morning_family_planning(current_time)
                    elif is_young_child:
                        return "Still sleeping"
                    else:
                        return self.review_daily_plan(current_time)
                else:
                    return "Getting ready for the day"
                
            elif hour == 6:  # Breakfast and family discussion time
                if hasattr(self, 'family_unit') and self.family_unit:
                    if is_parent:
                        # Parents initiate breakfast and discussion
                        result = self.handle_family_breakfast(current_time, all_agents)
                        # After breakfast, discuss and finalize today's plan
                        schedule = self.generate_family_daily_schedule(current_time)
                        conversation = self.generate_family_conversation({
                            'type': 'morning_planning',
                            'schedule': schedule,
                            'family_members': family_members_present
                        })
                        self.broadcast_family_schedule(schedule, current_time, conversation)
                        return result
                    else:
                        # Children and teens participate in discussion
                        parent = next((agent for agent in family_members_present 
                                     if hasattr(agent, 'family_role') and 
                                     agent.family_role == 'parent'), None)
                        if parent:
                            # Receive and respond to family schedule
                            schedule = parent.current_family_schedule if hasattr(parent, 'current_family_schedule') else None
                            if schedule:
                                response = self.receive_family_schedule(schedule, current_time, None)
                                # Generate child's input to the plan
                                child_input = self.generate_family_conversation({
                                    'type': 'schedule_response',
                                    'age': getattr(self, 'age', 18),
                                    'preferences': self.create_basic_plan()
                                })
                                # Record the interaction
                                self.memory_manager.add_memory(
                                    agent_name=self.name,
                                    memory_text=f"Discussed daily schedule with family during breakfast",
                                    memory_type="family_planning",
                                    timestamp=current_time,
                                    importance=0.8,
                                    details={'parent': parent.name, 'response': child_input}
                                )
                                return "Having breakfast and discussing daily plan with family"
                            return "Having breakfast with family"
                return self.handle_breakfast(current_time)
                
            elif hour == 7:  # Preparation time
                if is_parent:
                    return self.prepare_children_for_school(current_time)
                elif is_young_child or is_supervised_teen:
                    parent = next((agent for agent in family_members_present 
                                 if hasattr(agent, 'family_role') and 
                                 agent.family_role == 'parent'), None)
                    if parent:
                        # Record the preparation interaction
                        self.memory_manager.add_memory(
                            agent_name=self.name,
                            memory_text=f"Getting ready for school with {parent.name}'s help",
                            memory_type="morning_preparation",
                            timestamp=current_time,
                            importance=0.5
                        )
                        return f"Getting ready for school with {parent.name}'s help"
                    return "Waiting for parent's help to get ready"
                elif is_independent_teen:
                    return self.prepare_for_school(hour, all_agents)
                elif hasattr(self, 'workplaces') and self.workplaces:
                    return "Preparing for work"
                return "Starting morning routine"
                    
            elif hour == 8:  # Departure time
                if is_young_child or is_supervised_teen:
                    parent = next((agent for agent in family_members_present 
                                 if hasattr(agent, 'family_role') and 
                                 agent.family_role == 'parent'), None)
                    if parent and hasattr(self, 'school_location'):
                        self.location = self.school_location
                        # Record the school trip
                        self.memory_manager.add_memory(
                            agent_name=self.name,
                            memory_text=f"Went to school with {parent.name}",
                            memory_type="transportation",
                            timestamp=current_time,
                            importance=0.5
                        )
                        return f"Going to school with {parent.name}"
                    return "Waiting for parent to go to school"
                elif is_independent_teen or (hasattr(self, 'is_student') and self.is_student):
                    self.location = self.school_location
                    return "Going to school independently"
                elif hasattr(self, 'workplaces') and self.workplaces:
                    self.location = self.workplaces[0]
                    return "Going to work"
                
                return None

        except Exception as e:
            print(f"Error in morning routine for {self.name}: {str(e)}")
            return "Continuing with default morning activities"

    def find_parent(self, all_agents):
        """Helper method to find parent of a child"""
        try:
            if hasattr(self, 'family_unit') and self.family_unit:
                for agent in all_agents:
                    if (hasattr(agent, 'family_unit') and 
                        agent.family_unit == self.family_unit and 
                        hasattr(agent, 'family_role') and 
                        agent.family_role == 'parent'):
                        return agent
            return None
        except Exception as e:
            print(f"Error finding parent for {self.name}: {str(e)}")
            return None

    def build_action_context(self, current_location, hour, nearby_agents):
        """Build context for action decision"""
        context = []
        
        # Location context
        if current_location == self.residence:
            context.append("This is your residence where you can rest.")
        
        # School context
        if self.is_student and hour in range(8, 15) and current_location == self.school_location:
            context.append("You are at school during school hours.")
        
        # Fried Chicken Shop context
        if self.location == "Fried Chicken Shop":
            current_day = (hour // 24) + 1
            location = self.locations.get("Fried Chicken Shop")
            
            if location and hasattr(location, 'discount') and current_day in location.discount['days']:
                discount_value = location.discount['value']
                context.append(f"There is currently a {discount_value}% discount on all meals!")
            
            if hasattr(self, 'location_last_purchase') and self.location in self.location_last_purchase:
                time_since_last_purchase = hour - self.location_last_purchase[self.location]
                if time_since_last_purchase < 4:
                    context.append(f"You've purchased at {self.location} recently and might want to wait.")
            else:
                context.append("You can purchase a meal if you'd like.")
        
        # Social context
        if nearby_agents:
            context.append(f"There are {len(nearby_agents)} other people here.")
        
        return context

    def generate_contextual_action(self, context, current_time):
        """Generate action based on context"""
        hour = int(current_time) % 24
        current_plan = self.plans[-1] if hasattr(self, 'plans') and self.plans else "No plan yet"
        
        prompt = f"You are {self.name} at {self.location}. Current time is {hour}:00. "
        prompt += f"Your current plan: {current_plan}. "
        if context:
            prompt += f"Context: {' '.join(context)}. "
        prompt += "What would you like to do?"
        
        action = generate(prompt)
        return action if action else "Stay at current location"

    def process_location_specific_actions(self, action, current_time):
        """Process location-specific actions"""
        # Handle Fried Chicken Shop actions
        if any(phrase in action.lower() for phrase in ["go to the fried chicken shop", "visit fried chicken shop", "get fried chicken"]):
            if self.location != "Fried Chicken Shop":
                self.location = "Fried Chicken Shop"
                return f"Moved to Fried Chicken Shop"
        
        # Handle purchase actions
        if "buy" in action.lower() and self.location == "Fried Chicken Shop":
            return self.buy_food(current_time)
        
        return None

    def handle_morning_family_planning(self, current_time):
        """Handle early morning family planning (5:00 AM)"""
        try:
            if not hasattr(self, 'family_unit') or not self.family_unit:
                return "No family to plan with"
                
            # Generate the family schedule for the day
            schedule = self.generate_family_daily_schedule(current_time)
            
            # Create a meaningful family conversation about the day ahead
            context = {
                'time': current_time,
                'schedule': schedule,
                'location': self.residence
            }
            conversation = self.generate_family_conversation(context)
            
            # Broadcast the schedule to family members
            self.broadcast_family_schedule(schedule, current_time, conversation)
            
            return f"Shared today's family schedule and had morning conversation"
            
        except Exception as e:
            print(f"Error in morning family planning for {self.name}: {str(e)}")
            return "Created basic family schedule"

    def handle_family_breakfast(self, current_time, all_agents):
        """Handle family breakfast and schedule coordination (6:00 AM)"""
        try:
            if not hasattr(self, 'family_unit') or not self.family_unit:
                return self.handle_breakfast(current_time)
                
            # Find family members present
            family_members_present = [
                agent for agent in all_agents 
                if hasattr(agent, 'family_unit') and 
                agent.family_unit == self.family_unit and
                agent.location == self.location
            ]
            
            # Check groceries for breakfast
            if self.grocery_level >= 10 * len(family_members_present):
                self.grocery_level -= 10 * len(family_members_present)
                
                # First, gather everyone's plans
                family_plans = {}
                for member in family_members_present:
                    if hasattr(member, 'plans') and member.plans:
                        family_plans[member.name] = member.plans[-1]
                
                # Coordinate schedules
                coordinated_schedule = self.coordinate_family_schedule(current_time)
                
                # Generate discussion about key points
                discussion_points = {
                    'school_transport': self.plan_school_transport(family_members_present),
                    'dinner_plans': self.plan_family_dinner(current_time),
                    'evening_activities': self.plan_evening_activities(family_members_present),
                    'special_needs': self.check_special_requirements(family_plans)
                }
                
                # Create breakfast conversation including schedule discussion
                context = {
                    'time': current_time,
                    'activity': 'family_breakfast',
                    'location': self.residence,
                    'participants': [member.name for member in family_members_present],
                    'schedule_discussion': discussion_points
                }
                conversation = self.generate_family_conversation(context)
                
                # Update everyone's memories and plans
                for member in family_members_present:
                    member.memory_manager.add_memory(
                        member.name,
                        "family_meal",
                        {
                            'activity': 'Family breakfast and schedule coordination',
                            'conversation': conversation,
                            'participants': [m.name for m in family_members_present],
                            'time': current_time,
                            'coordinated_schedule': coordinated_schedule,
                            'discussion_points': discussion_points,
                            'meal_type': 'breakfast',
                            'location': self.residence,
                            'grocery_used': 10
                        }
                    )
                    
                    # Update member's plan with coordinated schedule
                    if hasattr(member, 'plans') and member.plans:
                        updated_plan = self.merge_schedules(member.plans[-1], coordinated_schedule)
                        member.plans[-1] = updated_plan
                
                return "Had family breakfast and finalized daily schedule"
            else:
                return "Need to get breakfast supplies for the family"
            
        except Exception as e:
            print(f"Error in family breakfast for {self.name}: {str(e)}")
            return "Had simple family breakfast"

    def plan_school_transport(self, family_members):
        """Plan school transportation arrangements"""
        try:
            transport_plan = {}
            young_children = [
                member for member in family_members 
                if hasattr(member, 'age') and member.age < 15
            ]
            
            if young_children:
                # Find available parents
                parents = [
                    member for member in family_members
                    if hasattr(member, 'family_role') and member.family_role == 'parent'
                ]
                
                for parent in parents:
                    if hasattr(parent, 'workplaces') and parent.workplaces:
                        # Check if workplace is compatible with school transport
                        transport_plan[parent.name] = {
                            'can_dropoff': True,  # Add logic based on work schedule
                            'can_pickup': True    # Add logic based on work schedule
                        }
            
            return transport_plan
            
        except Exception as e:
            print(f"Error planning school transport: {str(e)}")
            return {}

    def plan_family_dinner(self, current_time):
        """Plan family dinner arrangements"""
        try:
            dinner_plan = {
                'time': '18:00',
                'location': self.residence,
                'grocery_status': self.grocery_level >= 10,
                'need_shopping': self.grocery_level < 50,
                'special_occasion': False  # Add logic for special occasions
            }
            return dinner_plan
            
        except Exception as e:
            print(f"Error planning family dinner: {str(e)}")
            return {'time': '18:00', 'location': self.residence}

    def plan_evening_activities(self, family_members):
        """Plan evening activities for family"""
        try:
            activities = {}
            for member in family_members:
                if hasattr(member, 'plans') and member.plans:
                    evening_plans = {
                        str(h): activity 
                        for h, activity in member.plans[-1].items() 
                        if 17 <= int(h) <= 21
                    }
                    activities[member.name] = evening_plans
            return activities
            
        except Exception as e:
            print(f"Error planning evening activities: {str(e)}")
            return {}

    def check_special_requirements(self, family_plans):
        """Check for any special requirements or conflicts"""
        try:
            special_needs = {
                'schedule_conflicts': [],
                'transport_needs': [],
                'meal_requirements': [],
                'shopping_needs': []
            }
            
            # Add logic to identify conflicts and special needs
            return special_needs
            
        except Exception as e:
            print(f"Error checking special requirements: {str(e)}")
            return {}

    def prepare_children_for_school(self, current_time):
        """Handle school preparation time (7:00 AM)"""
        try:
            if not hasattr(self, 'family_members') or 'children' not in self.family_members:
                return "No children to prepare for school"
                
            young_children = [
                child for child in self.family_members['children']
                if isinstance(child, dict) and child.get('age', 0) < 15
            ]
            
            if not young_children:
                return "No young children to prepare for school"
                
            # Pack lunches for children
            for child in young_children:
                if self.grocery_level >= 10:  # Need more groceries for lunch than breakfast
                    self.grocery_level -= 10
                    child_name = child.get('name')
                    if child_name in all_agents:
                        child_agent = all_agents[child_name]
                        child_agent.has_packed_lunch = True
                        child_agent.memory_manager.add_memory(
                            child_agent.name,
                            "preparation",
                            {
                                'activity': 'Parent packed school lunch',
                                'time': current_time
                            }
                        )
                else:
                    return "Need to get more groceries for children's lunches"
                    
            # Help children prepare
            preparation_conversation = self.generate_family_conversation({
                'activity': 'morning preparation',
                'time': current_time
            })
            
            for child in young_children:
                child_name = child.get('name')
                if child_name in all_agents:
                    child_agent = all_agents[child_name]
                    child_agent.memory_manager.add_memory(
                        child_name,
                        "preparation",
                        {
                            'activity': 'Getting ready for school with parent',
                            'conversation': preparation_conversation,
                            'time': current_time
                        }
                    )
                    
            return "Helped children prepare for school"
            
        except Exception as e:
            print(f"Error in prepare_children_for_school for {self.name}: {str(e)}")
            return "Basic school preparation completed"
        
    def handle_morning_departure(self, current_time, all_agents):
        """Handle morning departure (8:00-9:00)"""
        try:
            # Parents handle school dropoff first
            if self.family_role == 'parent':
                young_children = [
                    child for child in self.family_members.get('children', [])
                    if isinstance(child, dict) and child.get('age', 0) < 15
                ]
                if young_children:
                    return "Dropping children at school"
            
            # Students go to school
            if self.is_student:
                if self.needs_supervision:
                    return "Going to school with parent"
                else:
                    return f"Heading to {self.school_location}"
            
            # Workers go to work
            if self.workplaces and self.is_working(current_time):
                return f"Heading to {self.workplaces[0]}"
            
            return "Starting daily activities"
            
        except Exception as e:
            print(f"Error in morning departure: {e}")
            return "Continuing morning routine"

    def record_food_interaction(self, action, current_time):
        """Unified method for recording food-related memories and metrics"""
        location = locations.get(self.location)
        if not location:
            return
        
        # Specific handling for Fried Chicken Shop
        if self.location == "Fried Chicken Shop":
            # Get price from location object
            price = location.base_price
            current_day = (current_time // 24) + 1
            
            # Calculate final price with discount
            if hasattr(location, 'discount') and current_day in location.discount['days']:
                discount_amount = price * (location.discount['value'] / 100)
                final_price = price - discount_amount
            else:
                final_price = price
                discount_amount = 0
                
            purchase_details = {
                'content': action,
                'time': current_time,
                'location': location,
                'price': final_price,  # Use the actual final price
                'original_price': price,
                'used_discount': current_day in location.discount['days'],
                'discount_amount': discount_amount,
                'influenced_by': None,
                'nearby_agents': [a.name for a in locations[location].agents if a.name != self.name]
            }
            
            # Check for social influence
            recent_memories = self.memory_manager.get_recent_memories(self.name, current_time, 24)
            for memory in recent_memories:
                if memory['type'] == 'received_recommendation' and 'Fried Chicken Shop' in memory.get('content', ''):
                    purchase_details['influenced_by'] = memory['source']
                    break

            # Record in metrics
            self.metrics.record_interaction(
                self.name,
                location,
                "purchase",
                purchase_details
            )
            
    def generate_satisfaction_rating(self):
        """Generate satisfaction based on agent's actual experience"""
        # Get recent memories about Fried Chicken Shop
        recent_memories = self.memory_manager.get_recent_memories(
            self.name, 
            context="Fried Chicken Shop",
            limit=5
        )
        
        # Generate experience-based prompt
        prompt = f"""You are {self.name} evaluating your experience at the Fried Chicken Shop.
        Your recent experiences: {[m.get('content', '') for m in recent_memories]}
        
        Based on these experiences, provide ratings for:
        1. Overall satisfaction (1-5)
        2. Food quality (1-5)
        3. Price satisfaction (1-5)
        4. Service quality (1-5)
        5. Approximate wait time (in minutes)
        6. Would you recommend to others? (yes/no)
        7. How likely are you to return? (1-5)
        
        Respond in JSON format only."""
        
        try:
            response = generate(prompt)
            ratings = json.loads(response)
            
            return {
                'overall_rating': int(ratings.get('1', 4)),
                'food_quality': int(ratings.get('2', 4)),
                'price_satisfaction': int(ratings.get('3', 3)),
                'service': int(ratings.get('4', 4)),
                'wait_time': int(ratings.get('5', 10)),
                'would_recommend': ratings.get('6', 'yes').lower() == 'yes',
                'return_intention': int(ratings.get('7', 4))
            }
        except Exception as e:
            print(f"Error generating satisfaction: {e}")
            # Fallback ratings based on basic factors
            return self.generate_fallback_rating()

    def generate_fallback_rating(self):
        """Generate ratings based on basic factors if AI generation fails"""
        # Check if price was discounted
        current_day = (self.current_time // 24) + 1
        got_discount = current_day in [3, 4]
        
        # Check recent purchase history
        recent_purchases = [m for m in self.memory_manager.get_recent_memories(self.name, limit=10)
                          if m['type'] == 'purchase' and m['location'] == 'Fried Chicken Shop']
        is_repeat_customer = len(recent_purchases) > 1
        
        # Base ratings slightly higher for repeat customers and during discounts
        base_rating = 3
        if is_repeat_customer:
            base_rating += 1
        if got_discount:
            base_rating = min(5, base_rating + 1)
            
        return {
            'overall_rating': base_rating,
            'food_quality': base_rating,
            'price_satisfaction': base_rating + 1 if got_discount else base_rating,
            'service': base_rating,
            'wait_time': 10 if base_rating >= 4 else 15,
            'would_recommend': base_rating >= 4,
            'return_intention': base_rating
        }

    def generate_recommendation(self, target_agent, experiences, current_time):
        """Generate personalized recommendation based on experiences"""
        try:
            # Get recent experiences (including negative ones)
            recent_experiences = self.memory_manager.get_recent_memories(
                self.name,
                limit=5,
                location_filter="Fried Chicken Shop"
            )
            
            # Generate recommendation based on ALL experiences
            prompt = self.create_recommendation_prompt(context)
            recommendation = generate(prompt)
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation for {self.name}: {str(e)}")
            return None

    def record_word_of_mouth(self, message, listener, current_time):
        """Record word of mouth with proper metrics integration"""
        # Analyze sentiment
        sentiment_prompt = f"""Analyze the sentiment about the Fried Chicken Shop in this recommendation:
        '{message}'
        Respond with only one word: positive, negative, or neutral."""
        
        sentiment = generate(sentiment_prompt).strip().lower()
        
        # Record in metrics with proper format matching
        self.metrics.record_interaction(
                self.name,
            self.location,
            "word_of_mouth",
                {
                'sentiment': sentiment,
                'listener': listener,
                'content': message,
                    'time': current_time,
                    'location': self.location,
                # Add fields that match metrics tracking
                'discount_mentioned': any(keyword in message.lower() for keyword in SALES_KEYWORDS),
                'type': 'recommendation'
            }
        )

    def social_interaction(self, all_agents, action, current_time):
        """Enhanced social interaction with proper metrics tracking"""
        if "Fried Chicken Shop" in action.lower():
            listeners = [agent for agent in all_agents 
                       if agent.location == self.location 
                       and agent.name != self.name]
            
            for listener in listeners:
                recommendation = self.generate_recommendation(listener)
                
                # Record in social network metrics
                self.metrics.record_interaction(
                    self.name,
                    self.location,
                    "word_of_mouth",
                    {
                        'content': recommendation,
                        'listener': listener.name,
                        'time': current_time,
                        'location': self.location,
                        'type': 'social_interaction'
                    }
                )
                
                # Add to listener's memory
                listener.memory_manager.add_memory(
                    listener.name,
                    "received_recommendation",
                    {
                        'from_agent': self.name,
                        'message': recommendation,
                        'timestamp': current_time,
                        'location': self.location
                    }
                )
                
                # Update social network tracking
                self.metrics.daily_metrics[self.metrics.current_day]['social_network']['information_flow'].append({
                    'from': self.name,
                    'to': listener.name,
                    'time': current_time,
                    'content': recommendation,
                    'location': self.location
                })
                
                # Update community impact
                self.metrics.daily_metrics[self.metrics.current_day]['social_network']['community_impact'][self.location] = \
                    self.metrics.daily_metrics[self.metrics.current_day]['social_network']['community_impact'].get(self.location, 0) + 1

    def update_memory(self, action, time, all_agents=None):
        """Update agent's memory with proper word-of-mouth tracking"""
        if action and "Fried Chicken Shop" in action:
            # Determine sentiment
            sentiment_prompt = f"Analyze the sentiment towards the Fried Chicken Shop in this text. Only respond with one word: 'positive', 'negative', or 'neutral'. Text: {action}"
            sentiment = generate(sentiment_prompt).strip().lower()
            
            # Find potential listeners
            listeners = [agent.name for agent in all_agents 
                        if agent.location == self.location 
                        and agent.name != self.name]
            
            for listener in listeners:
                # Record word of mouth in metrics
                self.metrics.record_interaction(
                    self.name,
                    self.location,
                    "word_of_mouth",
                    {
                        'sentiment': sentiment,
                        'listener': listener,
                        'content': action,
                        'time': time
                    }
                )
                
                # Record in memory manager
                self.memory_manager.add_memory(
                    self.name,
                    "word_of_mouth",
                    {
                        'content': action,
                        'sentiment': sentiment,
                        'listener': listener,
                        'location': self.location,
                        'time': time
                    }
                )

    def get_purchase_price(self, location_name, purchase_type):
        """Get price for purchase based on location"""
        location = locations.get(location_name)
        if location and location.type == purchase_type:
            return location.base_price
        return None

    def buy_food(self, current_time):
        # Use self.locations instead of global locations
        location = self.locations.get(self.location)
        if not location or location.type not in ['local_shop', 'grocery']:
            return False, "Not at a food establishment"
        
        # Check time since last purchase at this specific location
        if self.location in self.location_last_purchase:
            time_since_last_purchase = current_time - self.location_last_purchase[self.location]
            if time_since_last_purchase < 4:
                return False, f"Too soon to purchase again at {self.location}"
        
        # Get social influence from recent recommendations
        recent_recommendations = self.memory_manager.retrieve_memories(
            self.name,
            current_time,
            memory_type="received_recommendation",
            limit=5
        )
        
        # Calculate social influence
        social_influence = 0
        if recent_recommendations:
            positive_count = sum(1 for rec in recent_recommendations 
                               if rec.get('sentiment') == 'positive' and 
                               rec.get('location') == self.location)
            social_influence = positive_count / len(recent_recommendations)
        
        # Get the fixed base price
        price = self.get_purchase_price(self.location, location.type)
        if not self.can_afford(price):
            return False, "Cannot afford meal"
        
        # Social influence affects decision to buy, not the price
        should_buy = (
            self.energy_level <= 40 or  # Very hungry
            (self.is_meal_time(current_time)[0]) or  # During meal time
            social_influence > 0.5  # Strong positive recommendations
        )
        
        if should_buy:
            success = self.process_purchase(price, current_time)
            if success:
                # Record the purchase in memory
                self.memory_manager.add_memory(
                    self.name,
                    "meal",
                    {
                        'type': 'purchased_meal',
                        'location': self.location,
                        'time': current_time,
                        'price': price,
                        'influenced_by_recommendations': social_influence > 0,
                        'energy_restored': True
                    }
                )
                
                # Update the last purchase time for this specific location
                self.location_last_purchase[self.location] = current_time
                return True, f"Bought meal at {self.location} for ${price:.2f}"
        
        return False, "Decided not to purchase"

    def buy_groceries(self, current_time):
        """Buy groceries at grocery stores"""
        if self.location not in experiment_settings['location_prices']['retail']:
            return False, "Not at a grocery store"
        
        needed = 100 - self.grocery_level
        price_per_unit = self.get_purchase_price(self.location, "retail")
        total_price = needed * price_per_unit
        
        if not self.can_afford(total_price):
            return False, "Cannot afford groceries"
            
        success, message = self.make_purchase(self.location, current_time)
        if success:
            self.grocery_level = 100
            return True, f"Bought groceries at {self.location} for ${total_price:.2f}"
        return False, message

    def can_afford(self, amount):
        """Check if purchase is affordable based on family income"""
        if self.shared_income:
            return self.get_household_income() >= amount
        return self.money >= amount

    def get_household_income(self):
        """Calculate total household income"""
        if not self.family_unit:
            return self.money
            
        family_info = town_data['family_units'][self.family_unit]
        total_income = 0
        
        for parent, income_info in family_info['household_income'].items():
            if income_info['type'] == 'monthly':
                total_income += income_info['amount']
            elif income_info['type'] == 'annual':
                total_income += income_info['amount'] / 12
            elif income_info['type'] == 'hourly':
                total_income += income_info['amount'] * income_info.get('hours_per_week', 40) * 4
                
        return total_income

    def coordinate_family_schedule(self, current_time):
        if not self.family_unit:
            return None
            
        hour = int(current_time) % 24
        family_data = town_data['family_units'][self.family_unit]
        members = family_data['members']
        
        # Morning coordination
        if hour == 7:
            if self.family_role == 'parent':
                young_children = [
                    child for child in members['children']
                    if child['age'] < 15  # Use age-based supervision
                ]
                if young_children:
                    return "Help children prepare for school"

        # After school/work coordination
        elif hour == 17:  # End of work/school day
            if self.family_role == 'parent':
                young_children = [c for c in self.family_members['children'] 
                                if c.startswith('child_') and int(c.split('_')[1]) <= 11]
                if young_children:
                    return "Pick up children from school"
                    
        # Evening family time
        elif 18 <= hour <= 20:
            if self.location == self.residence:
                return "Spend time with family"
                
        return None

    def handle_food_needs(self, hour):
        try:
            # Use self.locations instead of global locations
            current_location = self.location
            location = self.locations.get(current_location)
            
            if self.needs_food(hour):
                if current_location == "Fried Chicken Shop":
                    return self.buy_food(hour)
                
                # Pass locations to check_nearby_food_locations
                nearby_options = self.check_nearby_food_locations(hour)

        except Exception as e:
            print(f"Error in handle_food_needs for {self.name}: {str(e)}")
            return False, str(e)

    def check_family_meal_schedule(self, hour):
        """Check if it's family meal time"""
        if not self.family_unit:
            return False
        
        # Morning family breakfast (6-8)
        if 6 <= hour % 24 <= 8:
            return True
        
        # Evening family dinner (17-20)
        if 17 <= hour % 24 <= 20:
            # Check if parents are home
            parents_home = any(
                agents[parent].location == self.residence 
                for parent in self.family_members.get('parents', [])
                if isinstance(parent, str)
            )
            return parents_home

        return False

    def handle_school_meal(self, hour):
        """Handle meals during school hours"""
        if not self.is_during_school_hours(hour):
            return False, "Not during school hours"

        if 11 <= hour % 24 <= 12:  # Lunch time
            if self.needs_supervision:  # Under 15
                if hasattr(self, 'has_packed_lunch') and self.has_packed_lunch:
                    self.has_packed_lunch = False  # Consume lunch
                    self.energy_level = 100
                    self.last_meal_time = hour
                    if hasattr(self, 'bought_lunch') and self.bought_lunch:
                        self.bought_lunch = False
                        return True, "Ate takeout lunch at school"
                    return True, "Ate packed lunch at school"
                return False, "No lunch available"
            else:  # Independent student (15+)
                # Regular cafeteria purchase logic
                if self.can_afford(10):
                    self.money -= 10
                    self.energy_level = 100
                    self.last_meal_time = hour
                    return True, "Bought lunch at school cafeteria"
                return False, "Cannot afford school lunch"
    
        return False, "Not lunch time"

    def check_household_budget(self):
        """Check if household can afford groceries or dining out"""
        if not self.family_unit:
            return self.can_afford(50)  # Basic threshold for individual
        
        household = households.get(self.residence)
        if not household:
            return False
        
        # Check if household can afford either groceries or dining out
        min_budget_needed = 50 * len(household['members'])  # Estimate per person
        return household['money'] >= min_budget_needed

    def check_can_purchase_at_location(self, location, current_time):
        """Check if enough time has passed since last purchase at this location"""
        if not hasattr(self, 'location_last_purchase'):
            self.location_last_purchase = {}
        
        if location not in self.location_last_purchase:
            return True
        
        time_since_last_purchase = current_time - self.location_last_purchase[location]
        return time_since_last_purchase >= 4  # 4 hour cooldown

    def handle_location_interaction(self, current_time):
        """Handle interactions based on current location type"""
        try:
            location = locations.get(self.location)
            if not location:
                return False, "Invalid location"
            
            # Check if location is open
            if not location.is_open(current_time):
                return False, f"{self.location} is closed"
            
            # Handle different location types
            if location.type == 'local_shop':
                return self.handle_dining_interaction(current_time)
            elif location.type == 'grocery':
                return self.handle_retail_interaction(current_time)
            elif location.type == 'education':
                return self.handle_education_interaction(current_time)
            elif location.type == 'work_office':
                return self.handle_work_interaction(current_time)
            elif location.type == 'residence':
                return True, f"At home in {self.location}"
            elif location.type == 'community':
                return True, f"Spending time at {self.location}"
            
            return True, f"Spent time at {self.location}"
            
        except Exception as e:
            print(f"Error in location interaction for {self.name}: {str(e)}")
            return False, str(e)

    def handle_dining_interaction(self, current_time):
        """Handle dining-specific interactions"""
        try:
            location = locations.get(self.location)
            if not location or location.type != 'local_shop':
                return False, "Not at a dining establishment"
            
            base_price = location.base_price
            
            # Check for discounts
            current_day = (current_time // 24) + 1
            if hasattr(location, 'discount') and current_day in location.discount['days']:
                discount = location.discount['value']
                final_price = base_price * (1 - discount/100)
            else:
                final_price = base_price
            
            if not self.can_afford(final_price):
                return False, f"Cannot afford meal at {self.location}"
            
            success = self.process_purchase(final_price, current_time)
            if success:
                self.record_dining_experience(current_time, final_price)
                return True, f"Enjoyed a meal at {self.location}"
            
            return False, "Purchase failed"
            
        except Exception as e:
            print(f"Error in dining interaction for {self.name}: {str(e)}")
            return False, str(e)

    def handle_social_interaction(self, current_time, nearby_agents):
        """Handle social interactions and word-of-mouth"""
        try:
            # Skip if no nearby agents
            if not nearby_agents:
                return
            
            # Get recent experiences
            recent_experiences = self.memory_manager.get_recent_memories(
                self.name,
                limit=5,
                location_filter="Fried Chicken Shop"
            )
            
            # Share experiences with nearby agents
            for nearby_agent in nearby_agents:
                # Skip if already shared recently
                if self.has_recent_interaction_with(nearby_agent.name, current_time):
                    continue
                
                # Generate and share recommendation
                if recent_experiences:
                    recommendation = self.generate_recommendation(
                        nearby_agent,
                        recent_experiences,
                        current_time
                    )
                    
                    # Record the interaction
                    self.record_social_interaction(
                        nearby_agent.name,
                        recommendation,
                        current_time
                    )
                    
                    # Update nearby agent's memory
                    nearby_agent.receive_recommendation(
                        self.name,
                        recommendation,
                        current_time
                    )
                
        except Exception as e:
            print(f"Error in social interaction for {self.name}: {str(e)}")

    def handle_education_interaction(self, current_time):
        """Handle education-related activities"""
        try:
            hour = current_time % 24
            
            # Skip if outside school hours
            if not (8 <= hour < 15):
                return False, "Outside school hours"
            
            # Handle different student types
            if self.student_type == 'high_school':
                return self.handle_high_school_activity(hour)
            elif self.student_type == 'college':
                return self.handle_college_activity(hour)
            elif self.student_type == 'part_time_student':
                return self.handle_part_time_study(hour)
            
            return False, "Not a student"
            
        except Exception as e:
            print(f"Error in education interaction for {self.name}: {str(e)}")
            return False, str(e)

    def handle_high_school_activity(self, hour):
        """Handle high school student activities"""
        if 8 <= hour < 15:
            self.memory_manager.add_memory(
                self.name,
                "education",
                {
                    'activity': 'Attended classes at Town Public High School',
                    'location': 'Town Public High School',
                    'time': hour
                }
            )
            return True, "Attended high school classes"
        return False, "Not during school hours"

    def handle_college_activity(self, hour):
        """Handle college student activities"""
        if 8 <= hour < 16:
            self.memory_manager.add_memory(
                self.name,
                "education",
                {
                    'activity': 'Attended college classes',
                    'location': 'Town Community College',
                    'time': hour
                }
            )
            return True, "Attended college classes"
        return False, "Not during class hours"

    def handle_part_time_study(self, hour):
        """Handle part-time student activities"""
        if 18 <= hour < 21:  # Evening classes
            self.memory_manager.add_memory(
                self.name,
                "education",
                {
                    'activity': 'Attended evening classes',
                    'location': 'Town Community College',
                    'time': hour
                }
            )
            return True, "Attended evening classes"
        return False, "Not during evening class hours"

    def is_meal_time(self, current_time):
        """Check if it's a regular meal time"""
        hour = current_time % 24
        
        for meal, (start, end) in self.meal_schedule.items():
            if start <= hour < end:
                return True, meal
        return False, None

    def update_energy(self, current_time):
        """Update energy level, decreasing by 20 every hour"""
        if self.last_meal_time is None:
            self.last_meal_time = current_time - 1  # Initialize with 1 hour ago
        
        hours_passed = current_time - self.last_meal_time
        energy_decrease = hours_passed * 20
        self.energy_level = max(0, self.energy_level - energy_decrease)

    def needs_food(self, current_time):
        """Determine if agent needs to eat based on energy and meal times"""
        self.update_energy(current_time)
        is_meal_time, meal_type = self.is_meal_time(current_time)
        
        return (
            self.energy_level <= 30 or  # Very low energy
            (is_meal_time and self.energy_level <= 60)  # Moderately low energy during meal time
        )

    def prepare_meal(self, current_time):
        """Handle meal preparation with energy system"""
        try:
            # First check if we need food
            if not self.needs_food(current_time):
                return False, "Energy level still good, don't need to eat yet"
            
            # Verify we're at residence
            if self.location != self.residence:
                return False, "Can only prepare meals at residence"
            
            # Check if we have enough groceries
            if not hasattr(self, 'grocery_level') or self.grocery_level < 10:
                return False, "Not enough groceries to prepare a meal"
            
            # Get current meal type
            is_meal_time, meal_type = self.is_meal_time(current_time)
            if not is_meal_time:
                return False, "Not a proper meal time"
            
            # Set grocery consumption for different meal types
            if meal_type == 'breakfast':
                grocery_cost = 10
            elif meal_type == 'lunch' and self.needs_supervision:
                grocery_cost = 10  # Packed lunch
            elif meal_type == 'dinner':
                grocery_cost = 15  # Dinner
                if self.family_unit:  # Family dinner needs more groceries
                    present_family = len([m for m in self.family_members.get('parents', []) + 
                                       self.family_members.get('children', [])
                                       if isinstance(m, str) and 
                                       agents[m].location == self.residence])
                    grocery_cost *= present_family
            
            if self.grocery_level < grocery_cost:
                return False, f"Not enough groceries for {meal_type}"
                
            # Consume groceries and update energy
            self.grocery_level -= grocery_cost
            self.energy_level = 100
            self.last_meal_time = current_time
            
            # Record the meal preparation in memory
            self.memory_manager.add_memory(
                self.name,
                "meal",
                {
                    'type': 'prepared_meal',
                    'meal_type': meal_type,
                    'location': self.residence,
                    'time': current_time,
                    'with_family': bool(self.family_unit),
                    'grocery_used': grocery_cost,
                    'energy_restored': True
                }
            )
            
            return True, f"Prepared {meal_type} at {self.residence}"
            
        except Exception as e:
            print(f"Error in prepare_meal for {self.name}: {str(e)}")
            return False, str(e)

    def check_grocery_needs(self):
        """Check if agent needs to buy groceries"""
        if not hasattr(self, 'grocery_level'):
            self.grocery_level = 100  # Initialize if not exists
            return False
        
        return self.grocery_level < 30  # Return True if groceries are low

    def process_purchase(self, price, current_time):
        """Process a purchase and update relevant tracking"""
        try:
            # Check if can afford
            if not self.can_afford(price):
                return False
            
            # Process payment based on whether agent has personal income
            if self.has_income():
                self.money -= price
            else:
                households[self.residence]['money'] -= price
            
            # Update purchase tracking for this specific location
            self.location_last_purchase[self.location] = current_time
            
            return True
            
            # Calculate price with possible discount
            base_price = location.base_price
            final_price = self.calculate_price_with_discount(base_price, location)
            
            # Process purchase
            if self.process_purchase(final_price, hour):
                # Record purchase time for THIS location
                self.location_last_purchase[location.name] = hour
                
                # Generate satisfaction score (higher if socially influenced)
                recent_recommendations = self.memory_manager.retrieve_memories(
                    self.name, hour, memory_type="received_recommendation", limit=5
                )
                has_positive_recommendations = any(
                    rec.get('sentiment') == 'positive' and 
                    rec.get('location') == location.name 
                    for rec in recent_recommendations
                )
                satisfaction = random.uniform(0.8, 1.0) if has_positive_recommendations else random.uniform(0.6, 0.9)
                
                # Record in memory
                memory = {
                    'type': 'purchase',
                    'location': location.name,
                    'amount': final_price,
                    'timestamp': datetime.now(),
                    'satisfaction': satisfaction,
                    'was_recommended': has_positive_recommendations
                }
                self.memory_manager.add_memory(self.name, memory)
                
                # Record in metrics
                if hasattr(self.metrics, 'record_purchase'):
                    self.metrics.record_purchase(
                        agent_name=self.name,
                        location_name=location.name,
                        amount=final_price,
                        hour=hour,
                        is_discount=final_price < base_price,
                        was_recommended=has_positive_recommendations
                    )
                
                return f"Purchased food at {location.name} for ${final_price:.2f}"
            
            return f"Cannot afford food at {location.name}"
        except Exception as e:
            print(f"Error processing purchase for {self.name}: {str(e)}")
            return False

    def record_dining_experience(self, current_time, final_price):
        # All experiences are recorded, regardless of sentiment
        self.memory_manager.add_memory(
            self.name,
            "meal",
            {
                'activity': f'Purchased meal at {self.location}',
                'location': self.location,
                'time': current_time,
                'price': final_price,
                'energy_restored': True
            }
        )

    def has_income(self):
        """Check if agent has personal income"""
        if not self.description:
            return False
        return 'income' in self.description

    def calculate_price_with_discount(self, base_price, location):
        """Calculate final price with possible discount"""
        current_day = (self.current_time // 24) + 1
        if hasattr(location, 'discount') and current_day in location.discount['days']:
            discount = location.discount['value']
            return base_price * (1 - discount/100)
        return base_price

    def has_recent_interaction_with(self, other_agent_name, current_time):
        """Check if had recent interaction with another agent"""
        recent_interactions = self.memory_manager.retrieve_memories(
            self.name,
            current_time,
            memory_type="social_interaction",
            limit=1,
            filter_func=lambda m: m.get('target_agent') == other_agent_name
        )
        if not recent_interactions:
            return False
        last_interaction_time = recent_interactions[0].get('time', 0)
        return (current_time - last_interaction_time) < 4  # 4-hour cooldown

    def is_during_school_hours(self, hour):
        """Check if current time is during school hours"""
        hour = hour % 24
        return 8 <= hour < 15  # School hours are 8 AM to 3 PM

    def handle_supervised_meal(self, hour):
        """Handle meals for students needing supervision"""
        if not self.family_unit:
            return False, "No family unit to coordinate with"
            
        # Check if parent is available
        parent_available = any(
            agents[parent].location == self.location 
            for parent in self.family_members.get('parents', [])
            if isinstance(parent, str)
        )
        
        if parent_available:
            return self.handle_food_needs(hour)
        return False, "Need parent supervision for meal"

    def handle_independent_meal(self, hour):
        """Handle meals for independent students"""
        return self.handle_food_needs(hour)

    def generate_family_daily_schedule(self, current_time):
        """Generate comprehensive family schedule for the day"""
        if not self.family_unit or self.family_role != 'parent':
            return None
        
        daily_schedule = {
            'required_meals': {
                'breakfast': {'time': (6, 8), 'location': self.residence, 'required': True},
                'dinner': {'time': (18, 20), 'location': self.residence, 'required': True}
            },
            'school_activities': [],
            'work_schedules': [],
            'free_time': []
        }
        
        # Add children's school schedules
        for child in self.family_members.get('children', []):
            if isinstance(child, str) and child in agents:
                child_agent = agents[child]
                if child_agent.needs_supervision:
                    daily_schedule['school_activities'].append({
                        'type': 'school',
                        'member': child,
                        'dropoff': {'time': 8, 'location': child_agent.school_location},
                        'pickup': {'time': 15, 'location': child_agent.school_location}
                    })
        
        return daily_schedule

    def broadcast_family_schedule(self, schedule, current_time, conversation):
        """Share daily schedule with family members through conversation"""
        if self.family_role != 'parent':
            return
        
        # Generate morning family conversation
        conversation_context = {
            'time': current_time,
            'location': self.residence,
            'schedule': schedule,
            'present_family': [
                member for member_type in ['parents', 'children']
                for member in self.family_members.get(member_type, [])
                if isinstance(member, str) and 
                agents[member].location == self.residence
            ]
        }
        
        # Generate family conversation about the day's schedule
        morning_conversation = self.generate_family_conversation(conversation_context)
        
        # Record the conversation in everyone's memory
        for member in conversation_context['present_family']:
            agents[member].memory_manager.add_memory(
                agents[member].name,
                "family_conversation",
                {
                    'type': 'morning_planning',
                    'conversation': morning_conversation,
                    'participants': conversation_context['present_family'],
                    'time': current_time,
                    'location': self.residence,
                    'schedule_discussed': schedule
                }
            )
            # Each member receives and acknowledges the schedule
            agents[member].receive_family_schedule(
                schedule, 
                current_time, 
                morning_conversation
            )

    def generate_family_conversation(self, context):
        """Generate contextual family conversation about daily schedule"""
        try:
            hour = context['time'] % 24
            schedule = context['schedule']
            present_family = context['present_family']
            
            # Build conversation prompt based on family context
            prompt = f"""You are {self.name}, having a morning family conversation at {hour:02d}:00.
            Present family members: {', '.join(present_family)}
            Today's schedule includes:
            - School dropoffs: {[act['member'] for act in schedule.get('school_activities', [])]}
            - Family meals: {list(schedule.get('required_meals', {}).keys())}
            - Work schedules: {[work['member'] for work in schedule.get('work_schedules', [])]}
            
            Generate a natural family conversation about coordinating today's schedule, including:
            1. Morning routine reminders
            2. School/work schedules
            3. Pickup arrangements
            4. Meal planning
            5. Evening family time
            
            Format as a brief dialogue between family members."""
            
            conversation = generate(prompt)
            return conversation
            
        except Exception as e:
            print(f"Error generating family conversation: {e}")
            return "Basic schedule coordination discussion"

    def receive_family_schedule(self, schedule, current_time, conversation):
        """Process received family schedule with response"""
        self.family_schedule = schedule
        
        # Generate appropriate response based on role
        if self.family_role == 'child':
            if self.needs_supervision:  # Under 15
                response = "Acknowledges schedule and asks about lunch preparation"
            else:  # 15 and older
                response = "Confirms understanding and mentions any after-school activities"
        else:  # Parent
            response = "Confirms schedule and coordinates responsibilities"
        
        # Record both schedule and conversation
        self.memory_manager.add_memory(
            self.name,
            "family_coordination",
            {
                'schedule': schedule,
                'conversation': conversation,
                'my_response': response,
                'time': current_time,
                'type': 'morning_planning',
                'location': self.residence
            }
        )

    def check_family_obligations(self, current_time):
        """Check and handle family-related responsibilities"""
        try:
            # 1. Basic validation
            if not hasattr(self, 'family_role') or self.family_role != 'parent':
                return None
                
            hour = current_time % 24
            
            # 2. Get family members that need supervision
            young_children = [
                child for child in self.family_members.get('children', [])
                if isinstance(child, dict) and child.get('age', 0) < 15
            ]
            
            if not young_children:
                return None
                
            # 3. Morning routine (6:00-8:00)
            if 6 <= hour < 8:
                if self.location == self.residence:
                    return "Help children prepare for the day"
                    
            # 4. School dropoff (8:00-9:00)
            elif 8 <= hour < 9:
                children_schools = [
                    child.get('school_location') 
                    for child in young_children 
                    if child.get('school_location')
                ]
                if children_schools:
                    return f"Drop off children at {', '.join(set(children_schools))}"
                    
            # 5. School pickup (15:00)
            elif hour == 15:
                children_schools = [
                    child.get('school_location') 
                    for child in young_children 
                    if child.get('school_location')
                ]
                if children_schools:
                    return f"Pick up children from {', '.join(set(children_schools))}"
                    
            # 6. Evening routine (17:00-20:00)
            elif 17 <= hour < 20:
                if self.location == self.residence:
                    return "Spend time with family"
                    
            return None
            
        except Exception as e:
            print(f"Error in family obligations check for {self.name}: {str(e)}")
            return None

    def generate_family_reminder(self, context):
        """Generate reminder conversation about upcoming schedule items"""
        try:
            hour = context['time'] % 24
            schedule = context['schedule']
            present_family = context['present_family']
            
            # Find next scheduled activity
            next_activities = []
            for activity_type, activities in schedule.items():
                if activity_type == 'school_activities':
                    for act in activities:
                        if act['pickup']['time'] > hour:
                            next_activities.append(f"School pickup for {act['member']} at {act['pickup']['time']}:00")
                elif activity_type == 'required_meals':
                    for meal, details in activities.items():
                        start_time, _ = details['time']
                        if start_time > hour:
                            next_activities.append(f"{meal} at {start_time}:00")
            
            prompt = f"""You are {self.name}, reminding family members about upcoming activities.
            Current time: {hour:02d}:00
            Present family: {', '.join(present_family)}
            Next activities: {', '.join(next_activities) if next_activities else 'No more scheduled activities today'}
            
            Generate a brief, natural reminder conversation about upcoming activities."""
            
            reminder = generate(prompt)
            return reminder
        
        except Exception as e:
            print(f"Error generating reminder: {e}")
            return "Brief schedule reminder"

    def decide_action(self, current_time):
        """Hierarchical decision making"""
        # 1. Check family obligations first
        family_obligation = self.check_family_obligations(current_time)
        if family_obligation:
            return family_obligation
        
        # 2. If no family obligations, check individual needs
        if self.needs_food(current_time):
            # Check if during family meal time
            is_family_meal = self.check_family_meal_schedule(current_time)
            if is_family_meal:
                return self.handle_family_meal(current_time)
            else:
                # Individual food decision - use the existing handle_food_needs
                return self.handle_food_needs(current_time)
        
        # 3. If no food needs, continue with other activities
        return None

    def update_family_status(self, current_time, status_type, message):
        """Update family about important status changes"""
        if not self.family_unit:
            return
        
        for member_type in ['parents', 'children']:
            for member in self.family_members.get(member_type, []):
                if isinstance(member, str) and member in agents:
                    agents[member].receive_status_update({
                        'from': self.name,
                        'type': status_type,
                        'message': message,
                        'time': current_time
                    })

    def handle_family_meal(self, current_time):
        """Handle family meal coordination and execution"""
        # Must be at residence for family meal
        if self.location != self.residence:
            return False, "Must be at residence for family meal"
        
        # Count present family members
        present_family_members = [
            member for member_type in ['parents', 'children']
            for member in self.family_members.get(member_type, [])
            if isinstance(member, str) and 
               member != self.name and 
               agents[member].location == self.residence
        ]
        
        total_members = len(present_family_members) + 1  # Include self
        groceries_needed = 10 * total_members  # 10 groceries per person
        
        if present_family_members:
            # Check if enough groceries for everyone
            if self.grocery_level >= groceries_needed:
                # Prepare family meal
                self.grocery_level -= groceries_needed
                self.energy_level = 100
                self.last_meal_time = current_time
                
                # Record family meal for all present members
                for member in present_family_members:
                    agents[member].energy_level = 100
                    agents[member].last_meal_time = current_time
                    agents[member].memory_manager.add_memory(
                        member,
                        "family_meal",
                        {
                            'activity': 'Shared family meal at home',
                            'location': self.residence,
                            'time': current_time,
                            'type': 'family_coordination',
                            'members_present': total_members
                        }
                    )
                
                return True, f"Prepared family meal for {total_members} people"
            else:
                # Not enough groceries - decide alternative
                if self.check_household_budget():
                    # Try to get groceries first if can afford
                    if any(loc.type == 'grocery' and loc.is_open(current_time) 
                          for loc in locations.values()):
                        return False, "Need to buy groceries for family meal"
                    # If grocery stores closed, try restaurants
                    elif any(loc.type == 'local_shop' and loc.is_open(current_time) 
                            for loc in locations.values()):
                        return False, "Consider family dinner at restaurant"
                return False, "Not enough groceries and limited options available"
        
        return False, "Family members not present for meal"

    def prepare_for_school(self, hour, all_agents):
        try:
            if self.family_role == 'parent':
                if 7 <= hour < 8:
                    needs_takeout = self.grocery_level < 10
                    
                    # Use self.locations instead of global locations
                    current_open_shops = [loc for loc in self.locations.values() 
                                        if loc.type in ['local_shop'] and 
                                        loc.is_open(hour)]
                    
                    next_hour_shops = [loc for loc in self.locations.values() 
                                     if loc.type in ['local_shop'] and 
                                     loc.is_open(hour + 1)]
                    
                    for child in self.family_members.get('children', []):
                        if isinstance(child, dict) and child.get('age', 0) < 15:
                            child_name = child['name']
                            if child_name in all_agents:
                                child_agent = all_agents[child_name]
                                
                                # If we have enough groceries, pack lunch
                                if self.grocery_level >= 10:
                                    self.grocery_level -= 10
                                    child_agent.has_packed_lunch = True
                                    child_agent.memory_manager.add_memory(
                                        child_agent.name,
                                        "meal_prep",
                                        {'activity': 'Parent packed school lunch'}
                                    )
                                    continue
                                
                                # Handle takeout scenario
                                if needs_takeout:
                                    # Try currently open shops first
                                    available_shops = current_open_shops or next_hour_shops
                                    if available_shops:
                                        # Find most affordable option
                                        affordable_shops = [shop for shop in available_shops 
                                                         if self.can_afford(shop.base_price)]
                                        if affordable_shops:
                                            chosen_shop = min(affordable_shops, 
                                                            key=lambda x: x.base_price)
                                            
                                            # If shops aren't open yet, wait
                                            if not current_open_shops:
                                                self.memory_manager.add_memory(
                                                    self.name,
                                                    "planning",
                                                    {
                                                        'activity': f'Waiting for {chosen_shop.name} to open',
                                                        'child': child_name
                                                    }
                                                )
                                                return False, f"Need to wait for {chosen_shop.name} to open"
                                            
                                            # Buy takeout lunch
                                            if self.process_purchase(chosen_shop.base_price, hour):
                                                child_agent.has_packed_lunch = True
                                                child_agent.bought_lunch = True
                                                child_agent.memory_manager.add_memory(
                                                    child_agent.name,
                                                    "meal_prep",
                                                    {
                                                        'activity': f'Parent bought takeout lunch from {chosen_shop.name}',
                                                        'price': chosen_shop.base_price
                                                    }
                                                )
                                                self.memory_manager.add_memory(
                                                    self.name,
                                                    "purchase",
                                                    {
                                                        'activity': f'Bought takeout lunch for {child_name}',
                                                        'location': chosen_shop.name,
                                                        'price': chosen_shop.base_price
                                                    }
                                                )
                                                return True, f"Bought takeout lunch for {child_name} from {chosen_shop.name}"
                                            return False, "Purchase failed"
                                        return False, "Cannot afford takeout lunch"
                                    return False, "No food locations open or opening soon"
                    
                    # After handling all children
                    return True, "Finished preparing children's lunches"
                
                # If it's not lunch prep time yet
                elif hour < 7:
                    return False, "Too early to prepare lunches"
                else:
                    return False, "Lunch preparation time has passed"
            
            return False, "Not a parent"
            
        except Exception as e:
            print(f"Error in prepare_for_school for {self.name}: {str(e)}")
            return False, f"Error preparing for school: {str(e)}"

    def check_nearby_food_locations(self, current_time):
        """Check nearby food locations and their availability"""
        try:
            # Get locations based on agent's preferences and circumstances
            preferred_locations = []
            
            # Family members prefer family-friendly locations during meal times
            if self.family_unit:
                preferred_locations.extend([
                    loc for loc in self.locations.values()
                    if loc.type == 'local_shop' and 'family' in loc.description.lower()
                ])
                
            # Students prefer affordable options
            elif self.is_student:
                preferred_locations.extend([
                    loc for loc in self.locations.values()
                    if loc.type == 'local_shop' and loc.base_price <= 15
                ])
                
            # Working adults consider convenience and location
            elif self.workplaces:
                workplace = self.workplaces[0]
                preferred_locations.extend([
                    loc for loc in self.locations.values()
                    if loc.type == 'local_shop' and 
                    self.calculate_distance(workplace, loc.name) <= 2  # Nearby locations
                ])
                
            # If no specific preferences, consider all options
            if not preferred_locations:
                preferred_locations = [
                    loc for loc in self.locations.values()
                    if loc.type in ['local_shop', 'grocery']
                ]
                
            # Filter for open locations
            open_locations = [loc for loc in preferred_locations if loc.is_open(current_time)]
            
            if not open_locations:
                return False, "No suitable food locations open nearby"
                
            # Choose based on affordability
            affordable_locations = [loc for loc in open_locations if self.can_afford(loc.base_price)]
            
            if not affordable_locations:
                return False, "Cannot afford nearby food locations"
                
            # Choose the most suitable option
            chosen_location = min(affordable_locations, key=lambda x: x.base_price)
            
            return True, {
                'location': chosen_location.name,
                'price': chosen_location.base_price,
                'is_open': True
            }
            
        except Exception as e:
            print(f"Error checking food locations for {self.name}: {str(e)}")
            return False, str(e)

    def handle_work_and_pickup(self, hour):
        """Handle work schedule around school pickup times considering travel time"""
        try:
            young_children = [c for c in self.family_members.get('children', [])
                           if isinstance(c, dict) and c.get('age', 0) < 15]
            
            if not young_children:
                return None

            children_schools = [c.get('workplace') for c in young_children if c.get('workplace')]
            workplace = self.workplaces[0] if self.workplaces else None

            # Calculate travel times (in hours)
            NORTH_SOUTH_TRAVEL_TIME = 0.5  # 30 minutes between north and south
            CENTRAL_TRAVEL_TIME = 0.25     # 15 minutes to/from central areas

            # Get workplace position
            workplace_position = self.get_location_position(workplace)
            
            # Calculate total commute time needed
            if workplace_position == "south":
                commute_time = NORTH_SOUTH_TRAVEL_TIME
            elif workplace_position == "central":
                commute_time = CENTRAL_TRAVEL_TIME
            else:
                commute_time = 0.25  # Default travel time

            # Leave work early enough to reach school by 15:00
            pickup_prep_time = 15 - commute_time
            
            # Pre-pickup preparation (leave early based on location)
            if pickup_prep_time <= hour < 15:
                return f"Leave work early for school pickup (traveling from {workplace_position})"

            # Pickup time (15:00)
            elif hour == 15:
                if children_schools:
                    return f"Pick up children from {', '.join(set(children_schools))}"

            # Post-pickup decision (15:00-16:00)
            elif 15 < hour < 16:
                # Don't return to work due to distance
                return "Take children home after pickup"

            # Late afternoon (16:00-18:00)
            elif 16 <= hour < 18:
                if self.location == self.residence:
                    occupation = str(self.occupation).lower()
                    if any(job in occupation for job in ['manager', 'analyst', 'engineer']):
                        return "Work from home while supervising children"
                    else:
                        return "Supervise children at home"

            return None

        except Exception as e:
            print(f"Error in handle_work_and_pickup for {self.name}: {str(e)}")
            return None

    def get_location_position(self, location_name):
        """Determine the position (north/south/central) of a location"""
        if not location_name:
            return None
            
        for area, info in town_data['world_layout']['major_areas'].items():
            if location_name in info['locations']:
                return info['position']
        return "central"  # Default to central if not found

    def is_working(self, hour):
        """Enhanced work check considering child pickup and travel times"""
        if not any(work in str(self.occupation).lower() 
                  for work in ['manager', 'worker', 'crew', 'supervisor']):
            return False

        workplace = self.workplaces[0] if self.workplaces else None
        if not workplace or workplace not in locations:
            return False

        # Check if it's pickup time for parents with young children
        if self.family_role == 'parent':
            young_children = [c for c in self.family_members.get('children', [])
                           if isinstance(c, dict) and c.get('age', 0) < 15]
            if young_children:
                workplace_position = self.get_location_position(workplace)
                # Calculate when parent needs to leave work
                if workplace_position == "south":
                    leave_time = 14.5  # Leave at 14:30 if working in south
                elif workplace_position == "central":
                    leave_time = 14.75  # Leave at 14:45 if working in central
                else:
                    leave_time = 14.75  # Default leave time
                    
                # Not working during pickup window
                if leave_time <= hour < 16:
                    return False

        # Regular work check
        if 'part-time' in str(self.occupation).lower():
            today_shift = next((shift for shift in self.part_time_shifts 
                              if shift['day'] == (hour // 24) % 7), None)
            return today_shift and today_shift['start'] <= hour % 24 < today_shift['end']
        
        return locations[workplace].is_open(hour)

    def generate_conversation(self, context):
        """Generate contextual conversation based on location and time"""
        try:
            location = context['location']
            hour = context['time'] % 24
            nearby_agents = context['nearby_agents']
            
            # Build conversation prompt based on context
            prompt = f"""You are {self.name} at {location}. It's {hour:02d}:00.
            You are with: {', '.join([agent.name for agent in nearby_agents])}
            Recent experiences: {context['recent_experiences']}
            
            Generate a brief, natural conversation considering:
            1. Time of day and location
            2. Your relationship with others present
            3. Recent experiences and shared activities
            4. Current activities or plans
            
            What do you say or discuss?"""
            
            conversation = generate(prompt_meta.format(prompt))
            return conversation
            
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return "Makes small talk"

    def evening_reflection(self, current_time):
        """Enhanced evening reflection with influence on next day's planning"""
        try:
            # Get today's experiences
            today_memories = self.memory_manager.get_today_memories(self.name)
            
            # Categorize memories
            dining_experiences = [m for m in today_memories if m['type'] in ['dining_experience', 'meal']]
            received_recommendations = [m for m in today_memories if m['type'] == 'received_recommendation']
            family_interactions = [m for m in today_memories if m['type'] == 'family_interaction']
            
            # Generate reflection with specific focus on next day planning
            reflection_prompt = f"""You are {self.name}, reflecting on today's experiences:
            Dining experiences: {[exp['content'] for exp in dining_experiences]}
            Recommendations received: {[rec['content'] for rec in received_recommendations]}
            Family interactions: {[int['content'] for int in family_interactions]}
            
            Consider:
            1. What worked well in today's schedule?
            2. What dining choices were good/bad?
            3. Which recommendations seem worth trying?
            4. What should be changed for tomorrow?
            5. What family coordination could be improved?
            
            Generate a thoughtful reflection that will help plan tomorrow."""
            
            reflection = generate(reflection_prompt)
            
            # Store reflection with planning implications
            self.memory_manager.add_memory(
                self.name,
                "evening_reflection",
                {
                    'reflection': reflection,
                    'time': current_time,
                    'planning_insights': {
                        'dining_preferences': self.analyze_dining_experiences(dining_experiences),
                        'recommended_places': self.analyze_recommendations(received_recommendations),
                        'schedule_adjustments': self.analyze_schedule_effectiveness(family_interactions),
                        'family_coordination': self.evaluate_family_coordination(family_interactions)
                    }
                }
            )
            
            return reflection
            
        except Exception as e:
            print(f"Error in evening reflection for {self.name}: {str(e)}")
            return None

    def plan_next_day(self, reflection, current_time):
        """Generate next day's plan based on evening reflection"""
        try:
            # Get recent reflections and recommendations
            recent_reflection = self.memory_manager.get_recent_memories(
                self.name,
                memory_type="evening_reflection",
                limit=1
            )[0]
            
            planning_insights = recent_reflection.get('planning_insights', {})
            
            # Create base schedule
            schedule = self.generate_family_daily_schedule(current_time + 2)  # +2 to plan for next day
            
            # Modify schedule based on insights
            if planning_insights:
                # Adjust dining plans
                preferred_places = planning_insights.get('dining_preferences', {})
                recommended_places = planning_insights.get('recommended_places', [])
                
                # Modify meal plans based on preferences and recommendations
                schedule['meals'] = self.adjust_meal_plans(
                    schedule.get('meals', {}),
                    preferred_places,
                    recommended_places
                )
                
                # Adjust timing based on schedule effectiveness
                schedule_adjustments = planning_insights.get('schedule_adjustments', {})
                if schedule_adjustments:
                    schedule = self.apply_schedule_adjustments(schedule, schedule_adjustments)
            
            return schedule
            
        except Exception as e:
            print(f"Error in plan_next_day for {self.name}: {str(e)}")
            return None

    def adjust_meal_plans(self, meal_schedule, preferred_places, recommended_places):
        """Adjust meal plans based on preferences and recommendations"""
        adjusted_meals = meal_schedule.copy()
        
        # Prioritize highly-rated places for dining out
        good_options = [place for place, rating in preferred_places.items() if rating >= 4]
        good_options.extend(recommended_places)
        
        # Modify dinner plans if good options available
        if good_options and self.grocery_level < 30:
            adjusted_meals['dinner'] = {
                'type': 'dine_out',
                'location': random.choice(good_options),
                'reason': 'positive_experience' if good_options[0] in preferred_places else 'recommendation'
            }
        
        return adjusted_meals

    def share_daily_plan(self, current_time):
        """Share next day's plan with family in the morning"""
        if not hasattr(self, 'next_day_plan'):
            self.next_day_plan = self.generate_family_daily_schedule(current_time)
        
        # Generate conversation including insights from evening reflection
        recent_reflection = self.memory_manager.get_recent_memories(
            self.name,
            memory_type="evening_reflection",
            limit=1
        )
        
        conversation_context = {
            'time': current_time,
            'location': self.residence,
            'schedule': self.next_day_plan,
            'reflection_insights': recent_reflection[0] if recent_reflection else None,
            'present_family': [
                member for member_type in ['parents', 'children']
                for member in self.family_members.get(member_type, [])
                if isinstance(member, str) and 
                agents[member].location == self.residence
            ]
        }
        
        # Generate and share morning conversation
        morning_conversation = self.generate_family_conversation(conversation_context)
        
        # Record the conversation and schedule for all present family members
        for member in conversation_context['present_family']:
            agents[member].receive_family_schedule(
                self.next_day_plan,
                current_time,
                morning_conversation
            )

    def analyze_dining_experiences(self, dining_experiences):
        """Analyze dining experiences to inform future choices"""
        preferences = {}
        for exp in dining_experiences:
            location = exp.get('location')
            if location:
                satisfaction = exp.get('satisfaction', 0)
                energy_restored = exp.get('energy_restored', False)
                cost = exp.get('price', 0)
                
                # Calculate overall rating
                rating = 0
                if satisfaction > 0:
                    rating += satisfaction * 0.4  # 40% weight
                if energy_restored:
                    rating += 3  # Base points for fulfilling need
                if cost > 0:
                    value_rating = min(5, (50 / cost) * 5)  # Value for money
                    rating += value_rating * 0.2  # 20% weight
                
                preferences[location] = round(rating, 2)
        
        return preferences

    def analyze_recommendations(self, recommendations):
        """Analyze received recommendations for planning"""
        recommended_places = []
        for rec in recommendations:
            if rec.get('sentiment') == 'positive':
                place = rec.get('location')
                if place and place not in recommended_places:
                    recommended_places.append(place)
        return recommended_places

    def analyze_schedule_effectiveness(self, family_interactions):
        """Analyze schedule effectiveness from family interactions"""
        adjustments = {}
        for interaction in family_interactions:
            if 'delayed' in str(interaction.get('content', '')).lower():
                time = interaction.get('time')
                if time:
                    hour = time % 24
                    adjustments[hour] = 'needs_more_time'
        return adjustments

    def plan_work_schedule(self, current_time):
        """Plan work schedule based on occupation and hours"""
        try:
            if 'part-time' in str(self.occupation).lower():
                return self.plan_part_time_work(current_time)
            elif self.workplaces:
                workplace = self.workplaces[0]
                if workplace in locations:
                    return {
                        'location': workplace,
                        'hours': locations[workplace].get_current_hours(current_time)
                    }
            return None
        except Exception as e:
            print(f"Error planning work schedule for {self.name}: {str(e)}")
            return None

    def plan_meals(self, planning_insights):
        """Plan meals based on preferences and recommendations"""
        meals = {}
        dining_preferences = planning_insights.get('dining_preferences', {})
        recommended_places = planning_insights.get('recommended_places', [])
        
        # Plan each meal
        for meal_type, (start, end) in self.meal_schedule.items():
            if self.grocery_level >= 30 and meal_type in ['breakfast', 'dinner']:
                meals[meal_type] = {
                    'type': 'home_meal',
                    'time': (start, end),
                    'location': self.residence
                }
            else:
                # Choose dining location based on preferences
                preferred_places = [place for place, rating in dining_preferences.items() 
                                 if rating >= 4 and place in locations]
                if preferred_places or recommended_places:
                    chosen_place = random.choice(preferred_places or recommended_places)
                    meals[meal_type] = {
                        'type': 'dine_out',
                        'time': (start, end),
                        'location': chosen_place
                    }
                else:
                    meals[meal_type] = {
                        'type': 'find_food',
                        'time': (start, end)
                    }
        
        return meals

    def plan_social_activities(self, planning_insights):
        """Plan social activities based on recent interactions"""
        social_activities = []
        recent_positive_interactions = planning_insights.get('positive_interactions', [])
        
        if recent_positive_interactions:
            # Plan to meet with people from positive interactions
            for interaction in recent_positive_interactions[:2]:  # Limit to 2 social activities
                social_activities.append({
                    'type': 'social_meeting',
                    'with': interaction.get('person'),
                    'location': interaction.get('location'),
                    'time': self.find_free_time_slot()
                })
        
        return social_activities

    def plan_leisure_activities(self, planning_insights):
        """Plan leisure activities based on preferences"""
        leisure_activities = []
        preferred_activities = planning_insights.get('enjoyed_activities', [])
        
        if preferred_activities:
            activity = random.choice(preferred_activities)
            leisure_activities.append({
                'type': 'leisure',
                'activity': activity,
                'time': self.find_free_time_slot(evening=True)
            })
        
        return leisure_activities

    def find_free_time_slot(self, evening=False):
        """Find available time slot for activities"""
        if evening:
            return (18, 21)  # Evening slot
        return (16, 18)  # Afternoon slot

    def review_and_adjust_plan(self, current_time):
        """Review and adjust plan based on family schedule and personal preferences"""
        try:
            if not hasattr(self, 'next_day_plan'):
                return
            
            # Get family schedule if part of family
            if self.family_unit:
                family_schedule = self.family_schedule if hasattr(self, 'family_schedule') else None
                if family_schedule:
                    # Adjust personal activities around family obligations
                    self.next_day_plan = self.merge_schedules(
                        self.next_day_plan,
                        family_schedule
                    )
            
            # Record final adjusted plan
            self.memory_manager.add_memory(
                self.name,
                "daily_planning",
                {
                    'type': 'adjusted_plan',
                    'plan': self.next_day_plan,
                    'time': current_time
                }
            )
            
        except Exception as e:
            print(f"Error adjusting plan for {self.name}: {str(e)}")

    def merge_schedules(self, personal_plan, family_plan):
        """Merge personal and family schedules, prioritizing family obligations"""
        merged = family_plan.copy()
        
        # Add personal activities that don't conflict with family obligations
        for activity_type, activities in personal_plan.items():
            if activity_type not in merged:
                merged[activity_type] = activities
            elif isinstance(activities, list):
                # Add non-conflicting activities
                for activity in activities:
                    if not self.time_conflicts(activity, merged):
                        merged[activity_type].append(activity)
        
        return merged

    def time_conflicts(self, activity, schedule):
        """Check if activity conflicts with existing schedule"""
        activity_time = activity.get('time')
        if not activity_time:
            return False
        
        for scheduled_items in schedule.values():
            if isinstance(scheduled_items, list):
                for item in scheduled_items:
                    if item.get('time') == activity_time:
                        return True
            elif isinstance(scheduled_items, dict):
                if scheduled_items.get('time') == activity_time:
                    return True
        
        return False

    def plan_next_day_individual(self, reflection, current_time):
        """Plan next day for individual agents"""
        try:
            # Get insights from reflection
            if not reflection:
                return None
                
            # Create basic schedule
            next_day_plan = {
                'meals': self.plan_meals(reflection),
                'work': self.plan_work_schedule(current_time),
                'social': self.plan_social_activities(reflection),
                'leisure': self.plan_leisure_activities(reflection)
            }
            
            return next_day_plan
            
        except Exception as e:
            print(f"Error in individual planning for {self.name}: {str(e)}")
            return None

    def plan_next_day_parent(self, reflection, current_time):
        """Plan next day for parent agents"""
        try:
            # Start with family schedule
            next_day_plan = self.generate_family_daily_schedule(current_time)
            
            # Add personal activities around family obligations
            if reflection:
                next_day_plan.update({
                    'personal': {
                        'meals': self.plan_meals(reflection),
                        'work': self.plan_work_schedule(current_time),
                        'leisure': self.plan_leisure_activities(reflection)
                    }
                })
            
            return next_day_plan
            
        except Exception as e:
            print(f"Error in parent planning for {self.name}: {str(e)}")
            return None

    def plan_next_day_child(self, reflection, current_time):
        """Plan next day for child agents"""
        try:
            # Start with school/study schedule
            next_day_plan = {
                'education': {
                    'location': self.school_location,
                    'hours': (8, 15) if self.needs_supervision else (9, 16)
                }
            }
            
            # Add activities around school
            if reflection:
                next_day_plan.update({
                    'meals': self.plan_meals(reflection),
                    'social': self.plan_social_activities(reflection),
                    'leisure': self.plan_leisure_activities(reflection)
                })
            
            return next_day_plan
            
        except Exception as e:
            print(f"Error in child planning for {self.name}: {str(e)}")
            return None

    def handle_breakfast(self, current_time):
        """Handle individual breakfast routine"""
        try:
            # First check if we've already had breakfast today
            today_memories = self.memory_manager.get_memories_for_day(current_time)
            already_had_breakfast = any(
                "breakfast" in memory.lower() 
                for memory in today_memories
            )
            
            if already_had_breakfast:
                return "Already had breakfast today"

            # Check if agent has enough groceries (10 per meal)
            if hasattr(self, 'grocery_level') and self.grocery_level >= 10:
                self.grocery_level -= 10  # Use groceries for breakfast
                self.energy = min(100, self.energy + 30)  # Breakfast provides energy
                
            self.memory_manager.add_memory(
                agent_name=self.name,
                    memory_text="Had breakfast at home",
                memory_type="meal",
                timestamp=current_time,
                    importance=0.6,
                    details={
                        'location': self.residence,
                        'energy_gained': 30,
                        'groceries_used': 10
                    }
                )
                return "Having breakfast at home"
    else:
                # Look for breakfast options nearby
                nearby_food = self.check_nearby_food_locations(current_time)
                if nearby_food:
                    # Record the out-of-home breakfast
                    self.memory_manager.add_memory(
                        agent_name=self.name,
                        memory_text=f"Had breakfast at {nearby_food}",
                        memory_type="meal",
                        timestamp=current_time,
                        importance=0.6,
                        details={
                            'location': nearby_food,
                            'reason': 'insufficient_groceries'
                        }
                    )
                    return nearby_food
                
                # If no options available, record the missed breakfast
                self.memory_manager.add_memory(
                    agent_name=self.name,
                    memory_text="Missed breakfast due to no food available",
                    memory_type="missed_meal",
                    timestamp=current_time,
                    importance=0.7,
                    details={
                        'reason': 'no_food_available',
                        'grocery_level': getattr(self, 'grocery_level', 0)
        return 1000  # Default if unknown income type

    def coordinate_family_discussion(self, current_time, family_members, initial_schedule):
        """Coordinate morning family discussion about daily plans"""
        try:
            # Collect all family members' preferences
            family_preferences = {}
            for member in family_members:
                if hasattr(member, 'next_day_preferences'):
                    family_preferences[member.name] = member.next_day_preferences
            
            # Generate discussion points
            discussion_points = {
                'meals': self.coordinate_meal_preferences(family_preferences),
                'activities': self.coordinate_activity_preferences(family_preferences),
                'schedule_conflicts': self.identify_schedule_conflicts(initial_schedule, family_preferences),
                'required_adjustments': self.determine_required_adjustments(initial_schedule, family_preferences)
            }
            
            # Generate family conversation
            conversation = self.generate_family_conversation({
                'type': 'morning_planning',
                'discussion_points': discussion_points,
                'family_members': family_members
            })
            
            # Adjust schedule based on discussion
            final_schedule = self.adjust_family_schedule(
                initial_schedule,
                discussion_points,
                family_preferences
            )
            
            # Record the discussion
            self.memory_manager.add_memory(
                agent_name=self.name,
                memory_text="Led morning family discussion to finalize daily schedule",
                memory_type="family_planning",
                timestamp=current_time,
                importance=0.8,
                details={
                    'participants': [member.name for member in family_members],
                    'discussion_points': discussion_points,
                    'final_schedule': final_schedule
                }
            )
            
            return {
                'final_schedule': final_schedule,
                'conversation': conversation
            }
            
        except Exception as e:
            print(f"Error in family discussion coordination for {self.name}: {str(e)}")
            return {'final_schedule': initial_schedule, 'conversation': "Basic schedule discussion"}

    def participate_in_family_discussion(self, current_time, parent, my_preferences):
        """Participate in family discussion with preferences"""
        try:
            # Generate response to parent's schedule
            response = self.generate_family_conversation({
                'type': 'schedule_response',
                'preferences': my_preferences,
                'parent_name': parent.name,
                'age': getattr(self, 'age', 18)
            })
            
            # Record participation
            self.memory_manager.add_memory(
                agent_name=self.name,
                memory_text=f"Participated in morning family discussion with {parent.name}",
                memory_type="family_planning",
                timestamp=current_time,
                importance=0.7,
                details={
                    'my_preferences': my_preferences,
                    'response': response
                }
            )
            
            return "shared preferences and discussed daily plan"
            
        except Exception as e:
            print(f"Error in family discussion participation for {self.name}: {str(e)}")
            return "listened to family discussion"

# Then the initialize_agents function that uses it
def initialize_agents(town_data, experiment_settings, metrics, memory_mgr, locations):
    agents = {}
    
    # Process family units
    for family_name, family_data in town_data['family_units'].items():
        # Initialize parents
        for parent in family_data['members']['parents']:
            agent = Agent(
                name=parent['name'],
                location=family_data['residence'],
                money=calculate_initial_money(parent['income']),
                occupation=parent['occupation'],
                experiment_settings=experiment_settings,
                metrics=metrics,
                memory_manager=memory_mgr,
                locations=locations
            )
            agents[parent['name']] = agent
            
        # Initialize children
        for child in family_data['members'].get('children', []):
            agent = Agent(
                name=child['name'],
                location=family_data['residence'],
                money=calculate_initial_money(child.get('income', None)),  # Some older children might have income
                occupation=child['occupation'],
                experiment_settings=experiment_settings,
                metrics=metrics,
                memory_manager=memory_mgr,
                locations=locations
            )
            agents[child['name']] = agent

    # Initialize individual town people
    for person_name, person_data in town_data.get('town_people', {}).items():
        basics = person_data['basics']
        agent = Agent(
            name=person_name,
            location=basics['residence'],
            money=calculate_initial_money(basics['income']),  # Also fix this to use calculate_initial_money
            occupation=basics['occupation'],
            experiment_settings=experiment_settings,
            metrics=metrics,
            memory_manager=memory_mgr,
            locations=locations  # Add this line
        )
        agent.description = basics
        agent.residence = basics['residence']
        agent.workplace = basics['workplace']
        agent.income = basics['income']
        agents[person_name] = agent

    return agents

# 2. Load configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, 'Stability_Agents_Config.json')

print(f"Loading config from: {config_path}")
with open(config_path, 'r') as f:
    town_data = json.load(f)

# 3. Initialize locations
locations = {}
for category, category_locations in town_data['town_areas'].items():
    for location_name, info in category_locations.items():
        if isinstance(info, str):
            info = {
                "description": info,
                "type": category,
                "hours": {"open": 0, "close": 24}
            }
        locations[location_name] = Location(location_name, info)

print(f"Initialized locations: {list(locations.keys())}")
print(f"Loaded locations: {list(town_data['town_areas'].keys())}")

# 4. Create metrics object
fried_chicken_location = locations.get("Fried Chicken Shop")
discount_settings = getattr(fried_chicken_location, 'discount', None)
metrics = FriedChickenMetrics(settings=discount_settings)

# 5. Create memory manager
memory_mgr = MemoryManager()

# 6. NOW initialize agents (after everything is defined)
agents = initialize_agents(town_data, experiment_settings, metrics, memory_mgr, locations)

# Initialize households after agents are created
households = {}
for agent_name, agent in agents.items():  # Change to iterate over items() since agents is a dictionary
    if not isinstance(agent, Agent):  # Skip if not an Agent object
        continue
        
    residence = agent.residence
    if residence not in households:
        households[residence] = {'members': [], 'money': 0}
    households[residence]['members'].append(agent)
    
    # Add income if agent has one
    if hasattr(agent, 'description') and agent.description:
        income = agent.description.get('income', {})
        if income:
            if income.get('type') == 'monthly':
                households[residence]['money'] += income.get('amount', 0)
            elif income.get('type') == 'annual':
                households[residence]['money'] += income.get('amount', 0) // 12
            elif income.get('type') == 'hourly':
                households[residence]['money'] += income.get('amount', 0) * 160  # Assuming 160 hours per month
            elif income.get('type') == 'pension':
                households[residence]['money'] += income.get('amount', 0)

# Main simulation loop
def run_simulation():
    try:
        global global_time
        whole_simulation_output = ""
        
        print("\n=== Starting Simulation ===")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        
        total_hours = 163  # Total simulation hours
        first_day = True
        
        while global_time < total_hours:
            current_day = (global_time // 24) + 1
            current_hour = global_time % 24
            
            # Print current timestamp
            print(f"\nDay {current_day}, Hour {current_hour:02d}:00")
            
            # Skip early hours only on first day
            if first_day and current_hour < 5:
                global_time += 1
                continue
            
            if current_hour == 0:
                first_day = False
            
            # Evening reflections and planning for ALL agents at 22:00
            if current_hour == 22:
                print(f"Day {current_day} - Evening reflection time")
                for agent in agents.values():
                    # Everyone does evening reflection
                    reflection = agent.evening_reflection(global_time)
                    
                    # Different planning based on agent type
                    if agent.family_role == 'parent':
                        next_day_plan = agent.plan_next_day_parent(reflection, global_time)
                    elif agent.family_role == 'child':
                        next_day_plan = agent.plan_next_day_child(reflection, global_time)
                    else:
                        next_day_plan = agent.plan_next_day_individual(reflection, global_time)
                    
                    agent.store_next_day_plan(next_day_plan, global_time)

            # Morning coordination at 5:00
            elif current_hour == 5:
                print(f"Day {current_day} - Morning coordination time")
                for agent in agents.values():
                    if agent.family_role == 'parent':
                        agent.share_daily_plan(global_time)
                    else:
                        agent.review_and_adjust_plan(global_time)
            
            # Regular hour processing...
            for agent in agents.values():
                try:
                    plan = agent.plan(global_time)
                    action = agent.execute_action(agents.values(), agent.location, global_time)
                    # Log action if enabled
                    if log_actions:
                        print(f"{agent.name} at {current_hour:02d}:00: {action}")
                except Exception as e:
                    print(f"Error processing agent {agent.name} at Day {current_day}, Hour {current_hour:02d}:00: {str(e)}")
                    continue  # Continue with next agent even if one fails
            
            global_time += 1
            
        return whole_simulation_output
        
    except Exception as e:
        print(f"Critical simulation error at Day {global_time//24 + 1}, Hour {global_time%24:02d}:00: {str(e)}")
        return str(e)

# Add validation for experiment settings
def validate_experiment_settings():
    required_fields = ['simulation', 'fried_chicken_shop', 'discount_settings']
    with open('experiment_settings.json', 'r') as f:
        settings = json.load(f)
        
    for field in required_fields:
        if field not in settings:
            raise ValueError(f"Missing required field in experiment_settings.json: {field}")
    
    return settings

# Add to main simulation
if __name__ == "__main__":
    try:
        print("\n=== Starting Simulation ===")
        print(f"Simulation started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Initialize agents as a dictionary
        agents = initialize_agents(town_data, experiment_settings, metrics, memory_mgr, locations)
        
        # Initialize households using the agents dictionary
        households = {}
        for agent_name, agent in agents.items():
            residence = agent.residence
            if residence not in households:
                households[residence] = {'members': [], 'money': 0}
            households[residence]['members'].append(agent)
            
            # Add income if agent has income
            if hasattr(agent, 'income'):
                income_data = agent.income
                if income_data['type'] == 'monthly':
                    households[residence]['money'] += income_data['amount']
                elif income_data['type'] == 'annual':
                    households[residence]['money'] += income_data['amount'] / 12
                elif income_data['type'] == 'hourly':
                    hours = income_data.get('hours_per_week', 40)
                    households[residence]['money'] += income_data['amount'] * hours * 4
        
        run_simulation()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        if 'memory_mgr' in locals():
            memory_file = memory_mgr.save_to_file()
        if 'metrics' in locals():
            metrics_file = metrics.save_metrics()

# At the start of the simulation, after imports
def load_configuration():
    """Single source for loading and validating configuration"""
    try:
        # Load experiment settings
        with open('experiment_settings.json', 'r') as f:
            experiment_settings = json.load(f)
            
        # Load town configuration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'Stability_Agents_Config.json')
        with open(config_path, 'r') as f:
            town_data = json.load(f)
            
        # Validate configuration
        validate_configuration(town_data, experiment_settings)
        
        return experiment_settings, town_data
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

def generate_daily_summary(self, current_day):
    """Generate a comprehensive daily summary"""
    summary = f"\n=== Daily Summary for Day {current_day} ===\n"
    
    # Location activity summary
    summary += "\nLocation Activity:\n"
    for location_name, location in self.locations.items():
        if hasattr(location, 'daily_visitors'):
            unique_visitors = len(set(location.daily_visitors))
            total_visits = len(location.daily_visitors)
            summary += f"{location_name}: {unique_visitors} unique visitors, {total_visits} total visits\n"
    
    # Family activity summary
    summary += "\nFamily Activity:\n"
    for family_name, family in self.family_units.items():
        summary += f"\n{family_name}:"
        for member_type in ['parents', 'children']:
            if member_type in family['members']:
                for member in family['members'][member_type]:
                    if isinstance(member, dict):
                        member_name = member['name']
                    else:
                        member_name = member
                    if member_name in self.agents:
                        agent = self.agents[member_name]
                        summary += f"\n  - {member_name}: Last location: {agent.current_location}"

    # Reset daily tracking
    for location in self.locations.values():
        if hasattr(location, 'daily_visitors'):
            location.daily_visitors = []

    print(summary)
    return summary

def run_simulation(self):
    sim_data = initialize_simulation()
    locations = sim_data['locations']
    agents = sim_data['agents']
    metrics = sim_data['metrics']
    settings = sim_data['settings']
    
    global_time = 5  # Start at 5 AM
    
    while global_time < (settings['duration_days'] * 24):
        current_day = global_time // 24
        
        for agent in agents.values():
            # Pass required parameters to methods
            if global_time % 24 == 5:  # 5 AM
                agent.create_daily_plan(global_time)
            
            # Pass all required parameters to execute_action
            agent.execute_action(agents, agent.location, global_time)
            
            # Evening reflection needs all context
            if global_time % 24 == 22:  # 10 PM
                reflection = agent.evening_reflection(global_time)
                next_day_plan = agent.plan_next_day(reflection, global_time)

def initialize_simulation():
    # Load configurations first
    config = load_configuration()
    experiment_settings = config['simulation']
    location_settings = config['location_prices']
    
    # Initialize core components with settings
    memory_mgr = MemoryManager()
    metrics = FriedChickenMetrics(settings=location_settings['dining']['Fried Chicken Shop'])
    
    # Initialize locations with settings
    locations = {}
    with open('LLMAgentsTown_experiment/Stability_Agents_Config.json') as f:
        town_data = json.load(f)
        for category, category_locations in town_data['town_areas'].items():
            for location_name, info in category_locations.items():
                locations[location_name] = Location(location_name, info)
    
    # Initialize agents with required parameters
    agents = initialize_agents(town_data, experiment_settings, metrics, memory_mgr, locations)
    
    return {
        'locations': locations,
        'agents': agents,
        'metrics': metrics,
        'memory_mgr': memory_mgr,
        'settings': experiment_settings
    }
