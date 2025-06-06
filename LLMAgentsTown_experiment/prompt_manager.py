from simulation_constants import (
    SIMULATION_SETTINGS, ACTIVITY_TYPES, MEMORY_TYPES,
    TimeManager, SimulationError, AgentError, LocationError,
    MemoryError, MetricsError, ErrorHandler, ThreadSafeBase,
    ENERGY_MAX, ENERGY_MIN, ENERGY_COST_PER_STEP, ENERGY_DECAY_PER_HOUR,
    ENERGY_COST_WORK_HOUR, ENERGY_COST_PER_HOUR_TRAVEL, ENERGY_COST_PER_HOUR_IDLE,
    ENERGY_GAIN_RESTAURANT_MEAL, ENERGY_GAIN_SNACK, ENERGY_GAIN_HOME_MEAL,
    ENERGY_GAIN_SLEEP, ENERGY_GAIN_NAP, ENERGY_GAIN_CONVERSATION,
    ENERGY_THRESHOLD_LOW
)

class PromptManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the prompt manager with all prompt templates."""
        self.prompts = {
            'daily_plan': """You are {name}, a {age}-year-old {occupation} living at {residence}. Your workplace is {workplace} with work hours from {work_schedule_start} to {work_schedule_end}.

Current time is {current_time}:00. Create a detailed daily schedule that includes:
1. Meal times (breakfast, lunch, dinner)
2. Work schedule
3. Travel time to/from work
4. Personal activities
5. Social interactions
6. Rest periods

Consider:
- Your energy levels
- Available locations
- Time of day
- Work commitments
- Basic needs (food, rest)

Format your response as a natural text schedule with times and activities.""",

            'plan_review_conversation': """You are {name}, reviewing your daily plan with {other_agent_name} at {location}.

Current time: {current_time}:00
Your current plan: {current_plan}
Their current plan: {other_agent_plan}

Have a natural conversation about your plans, considering:
1. Any schedule conflicts
2. Opportunities to spend time together
3. Shared activities or meals
4. Travel coordination
5. Household responsibilities

Format your response as a natural dialogue between you and {other_agent_name}, followed by your updated plan and reasoning for any changes made.""",

            'action': """You are {name}, currently at {location} at {current_time}:00.

Current state:
- Energy level: {energy_level}
- Money: ${money}
- Grocery level: {grocery_level}
- Current activity: {current_activity}

Recent activities: {recent_activities}

Based on your current state and needs, what should you do next? Consider:
1. Basic needs (food, rest)
2. Work schedule
3. Travel requirements
4. Social opportunities
5. Available resources

Respond with your next action.""",

            'activity_parsing': """Parse the following activity description into structured data:

Activity: {activity_description}

Extract:
1. Activity type
2. Location
3. Time
4. Participants
5. Resources needed
6. Energy impact
7. Cost (if any)

Format as JSON with these fields.""",

            'conversation': """You are {name} having a conversation with {other_agent_name} at {location}.

Current time: {current_time}:00
Relationship: {relationship}
Recent interactions: {recent_interactions}

Context: {conversation_context}

Respond naturally to continue the conversation.""",

            'household_coordination': """You are {name}, coordinating with your household members at {location}.

Current time: {current_time}:00
Household members: {household_members}
Your current plan: {current_plan}

Discuss:
1. Shared meals
2. Household chores
3. Travel arrangements
4. Social activities
5. Resource sharing

Format as a natural conversation.""",

            'conversation_analysis': """Analyze the following conversation between {participants}:

Conversation: {conversation_content}

Extract:
1. Main topics discussed
2. Decisions made
3. Plans agreed upon
4. Emotional tone
5. Action items

Format as structured data.""",

            'contextual_action': """You are {name} at {location} during {time_of_day}.

Current state:
- Energy: {energy_level}
- Money: ${money}
- Current activity: {current_activity}

Available options:
{available_options}

Choose your next action based on your needs and the current context.""",

            'location_context': """Location: {location_name}
Type: {location_type}
Current time: {current_time}:00
Open hours: {open_hours}
Current capacity: {current_capacity}/{max_capacity}
Base price: ${base_price}
Special offers: {special_offers}

Describe the current state and available activities at this location.""",

            "location_check": {
                "template": "Check if {location} is appropriate for {name} at {time}:00. Consider: {considerations}",
                "parameters": ["location", "name", "time", "considerations"]
            },
            "satisfaction_rating": """You are {name} evaluating your experience at {location}. 
Current time: {current_time}:00
Energy level: {energy_level}
Money spent: ${money:.2f}

Recent experiences: {experiences}

Please provide a detailed rating in the following format:
- Overall satisfaction (1-5): 
- Food quality (1-5): 
- Price satisfaction (1-5): 
- Service quality (1-5): 
- Wait time (minutes): 
- Would recommend (yes/no): 
- Likelihood to return (1-5): 
- Brief comment: 

Consider:
1. Food quality and taste
2. Price and value for money
3. Service experience
4. Atmosphere and ambiance
5. Wait time and efficiency
6. Overall satisfaction
7. Previous experiences at this location
""",
            "recommendation": {
                "template": "You are {name} recommending {location} to {target}. Your experiences: {experiences}. Consider their preferences and your relationship.",
                "parameters": ["name", "location", "target", "experiences"]
            },
            "memory_consolidation": {
                "template": """Review and consolidate memories for {name} at {current_time}:00.

Recent Memories:
{recent_memories}

Social Interactions:
{social_interactions}

Location History:
{location_history}

Consider:
1. Important events and interactions
2. Emotional significance
3. Impact on relationships
4. Learning experiences
5. Future relevance
6. Pattern recognition

Rate each memory's importance (0-1) and identify key insights.""",
                "parameters": ["name", "current_time", "recent_memories", "social_interactions", "location_history"]
            },
            "error_recovery": {
                "template": """Error occurred for {name} at {location}, time {time}:00.

Error Context:
- Type: {error_type}
- Activity: {current_activity}
- State: {current_state}
- Recent actions: {recent_actions}

Consider:
1. Impact on current activity
2. Alternative actions available
3. State preservation needs
4. Recovery priorities
5. Safety considerations
6. Communication needs

Generate appropriate recovery action and explanation.""",
                "parameters": ["name", "location", "time", "error_type", "current_activity", "current_state", "recent_actions"]
            },
            "mid_travel_location_decision": """
You are {agent_name}, currently traveling to {original_destination_name}.
You have just taken a step and are now at the coordinates of {encountered_location_name} ({encountered_location_type}). This location is currently open.

Your current status:
- Energy: {energy_level}
- Grocery Level: {grocery_level}
- Money: ${money:.2f}
- Current Time: {time}:00
- Time until next commitment: {time_to_next_commitment} hours
- Destination Urgency: {destination_urgency}
- Current Day: Day {current_day}

Your daily plan for today is:
{daily_plan}

Location Context:
- Type: {encountered_location_type}
- Current Offers/Discounts: {discount_info}
- Relevance to your needs: {location_relevance}
- Last visit: {last_visit_time}

Special Consideration for Fried Chicken Shop:
- You're passing by the Fried Chicken Shop, a popular local eatery
- 20% discount available on Days 3 (Wednesday) and 4 (Thursday)
- Current Day: Day {current_day}
- {discount_status}
- Quick service, perfect for a meal break
- Energy boost from a good meal could help with the rest of your journey

Location Context:
- Current Time: {time}:00
- Shop Hours: 10:00-22:00
- Base Price: $20.00 (Before any discounts)
- {discount_info}

Considering your current needs, your plan, and this unexpected location encounter, what do you want to do?
Choose ONE of the following options:

(A) Rush to original destination: {original_destination_name}
    Choose this if:
    - You're running late or have an urgent commitment
    - The destination is your workplace during work hours
    - You have a scheduled meeting or appointment
    - Your energy level is sufficient to continue

(B) Quick stop (5-10 minutes)
    Choose this if:
    - You have moderate time pressure but critical needs
    - Your energy/grocery levels are very low
    - There's a special discount today (check the day!)
    - The location directly supports your daily plan
    Specify what quick action you'll take (e.g., "grab a quick coffee", "buy essential groceries")

(C) Regular visit (15-30 minutes)
    Choose this if:
    - You have flexible time
    - The location is highly relevant to your needs
    - You can combine multiple tasks here
    - The detour won't significantly impact your schedule
    Specify what you plan to do during the visit

Respond with your choice letter (A, B, or C) followed by a colon and then a brief first-person thought process.
Example if choosing A: "A: Can't stop now, I have a meeting in 15 minutes at {original_destination_name}."
Example if choosing B: "B: My energy is critically low - I'll grab a quick coffee and snack, should take just 5 minutes."
Example if choosing C: "C: I have an hour before my next meeting, and this grocery store has everything I need for the week."
""",

            "mid_travel_agent_encounter_decision": """
You are {agent_name}, currently traveling to {original_destination_name}.
At your current step ({current_grid_coord}), you encounter: {encountered_agent_names_list}.
{encountered_agent_details_list}

Your current status:
- Energy: {energy_level}
- Current Time: {time}:00
Your daily plan for today is:
{daily_plan}

Important Context:
- Your relationships with encountered agents: {relationships}
- Your shared history with them: {shared_history}
- Your current urgency to reach destination: {destination_urgency}
- Time until your next commitment: {time_to_next_commitment} hours

What do you want to do regarding this encounter?
Choose ONE of the following options:
(A) Briefly acknowledge them (e.g., nod or wave) and continue traveling to {original_destination_name}.
(B) Stop your current travel and attempt to start a conversation with one of them. Consider this strongly if:
   - They are close friends, family, or colleagues
   - You share the same destination
   - You have time before your next commitment
   - You haven't interacted with them recently
(C) Ignore them and continue traveling to {original_destination_name}.

Respond with your choice letter (A, B, or C) followed by a colon and then a brief first-person thought process. If choosing (B), also state who you'd primarily like to talk to if there are multiple people.
Example if choosing A: "A: I'll wave at Sarah since we're both heading to work and running late."
Example if choosing B: "B: I'll stop to chat with Mike - he's my colleague and we're both heading to the office. We could discuss the project."
Example if choosing C: "C: I need to hurry to my appointment, can't stop now."
""",
            "solo_evening_routine": """
            You are {name}, at {location} in the evening ({time}:00).
            Your energy is {energy}.
            Your current grocery stock level is {grocery_level}%.
            Your recent activities: {recent_activities}.
            Your schedule for tomorrow: {next_day_schedule}.

            Describe your solo evening routine in the first person. What are you doing to wind down, prepare for tomorrow, or occupy yourself, considering your day, your grocery stock, and what's ahead?
            """,
            "food_decision": """
You are {name}, deciding whether to cook at home or eat out for {meal_type}.
Your current situation:
- Energy Level: {energy_level}/100
- Grocery Level: {grocery_level}/100
- Total Money: ${total_money:.2f}
- Daily Money Budget: ${daily_money:.2f}
- Location: {location}
- Current Time: {time:02d}:00

Recent dining experiences and ratings:
{recent_ratings}

IMPORTANT MEAL TIMING RULES:
- For LUNCH (11:00-14:00) and DINNER (17:00-21:00): Only consider proper restaurants (Fried Chicken Shop, Local Diner), NOT coffee shops
- For BREAKFAST (06:00-10:00): Coffee shops are appropriate for light breakfast items
- For SNACKS (other times): Coffee shops are fine for beverages and light snacks

Make a natural, human-like decision about whether to cook at home or eat out.
Consider:
1. Your current energy and grocery levels
2. The time and type of meal (coffee shops are NOT suitable for lunch/dinner)
3. Your financial situation
4. Past dining experiences
5. Random personal preferences
6. Other agents' satisfaction ratings

Respond with your decision and brief reasoning in a natural, first-person voice.
""",

            "food_location_choice": """
You are {name}, choosing where to eat {meal_type}.
Your current situation:
- Energy Level: {energy_level}/100
- Total Money: ${total_money:.2f}
- Daily Money Budget: ${daily_money:.2f}
- Current Time: {time:02d}:00

Available locations:
{available_locations}

Recent dining experiences and ratings:
{recent_ratings}

CRITICAL MEAL LOCATION RULES:
- For LUNCH (11:00-14:00) and DINNER (17:00-21:00): Only choose proper restaurants (Fried Chicken Shop, Local Diner)
- Coffee shops (The Coffee Shop) are ONLY appropriate for:
  * BREAKFAST items (06:00-10:00)
  * Light snacks and beverages (not main meals)
- Do NOT choose coffee shops for lunch or dinner - they don't serve full meals

Make a natural, human-like decision about where to eat.
Consider:
1. Your current energy level
2. The prices at each location
3. Past ratings and experiences
4. Your financial situation
5. Random personal preferences (sometimes people just crave certain foods)
6. MEAL TIMING: Choose appropriate locations for the type of meal you need

Respond with your chosen location and brief reasoning in a natural, first-person voice.
Make sure to clearly mention the exact name of your chosen location.
""",

            "conversation_turn": """
You are {name}, naturally continuing a conversation at {location}.

The conversation so far:
{previous_turns}

Recent conversation memories (if any):
{recent_memory_context}

Your current situation:
- Location: {location_type}
- Time: {time}:00
- Energy: {energy_level}
- What you're currently doing: {current_activity}
- Recent interactions: {recent_interactions}
- Your relationships with these people: {relationships}

Just say what you would naturally say next in this conversation. 
- Be yourself - use your natural speaking style
- React authentically to what was just said
- Use contractions, casual language, real speech patterns
- Keep it conversational and genuine
- If you want to end the conversation, do it naturally ("Alright, I should get going" or "See ya later!")
- If there are recent conversation memories, acknowledge and build upon them naturally

Don't include stage directions or explanations - just speak naturally.

What you say:""",

            "social_interaction": """
You are {name}, naturally interacting with people at {location}.
Time: {time:02d}:00

Your current situation:
{personal_context}

People around you:
{nearby_agents}

Why you're interacting: {interaction_reason}

Speak naturally to these people based on:
- Your relationship with them (friend, coworker, spouse, stranger, etc.)
- The location you're at and what's appropriate there
- Your current mood and energy level
- Any shared history or experiences you have
- The time of day and social context

Just talk like you normally would in this situation. Be genuine, show your personality, and use natural speech patterns.

Examples of natural interactions:
- "Hey! Didn't expect to see you here. How's it going?"
- "Morning, honey. Coffee smells amazing!"
- "Oh hi there! Are you enjoying the food here? I'm thinking about trying it."
- "Susan! Perfect timing - I was just about to text you about lunch plans."

Your natural interaction:""",

            "structured_action": """
You are {name} at {location} (Time: {time:02d}:00).

Current Status:
- Energy: {energy_level}/100
- Groceries: {grocery_level}/100  
- Money: ${money:.2f}

Work Context: {work_context}
Daily Plan: {daily_plan}

Available Locations:
{available_locations}

Think naturally about your situation and what you want to do next. You can express your thoughts, feelings, and reasoning in a natural way, just like you would in real conversation.

After your natural thoughts, you MUST end with this exact structured format:

LOCATION: [Where you want to go next - use exact location name or "stay"]
ACTION: [What you want to do there - be natural and conversational]
REASONING: [Brief explanation of your decision]

Rules for the structured conclusion:
1. If it's work time and you're not at workplace: LOCATION: {workplace}
2. Use exact location names from the available list
3. Consider your energy, money, and schedule
4. Use "stay" if no movement is needed

Example Response:
"Hmm, I'm feeling pretty tired and could really use some caffeine before diving into work today. The Coffee Shop is just down the street and their morning blend is exactly what I need to get energized. Plus, I have about 30 minutes before I need to be at the office, so perfect timing.

LOCATION: Coffee Shop
ACTION: Buy a coffee and maybe a pastry to fuel up for the workday
REASONING: Need energy boost before work starts and have time to spare"

Your natural thoughts and structured response:""",

            "movement_decision": """
You are {name}, deciding your next move.

Current Situation:
- Location: {current_location}
- Time: {time:02d}:00
- Energy: {energy_level}/100
- Work Status: {work_status}

Your options:
{location_options}

Choose ONE option and respond with just the location name:
""",

            "structured_purchase": """
You are {name} at {location} (Time: {time:02d}:00).

Current Status:
- Energy: {energy_level}/100
- Groceries: {grocery_level}/100  
- Money: ${money:.2f}
- Available Money for Purchases: ${available_money:.2f}

**CRITICAL ENERGY ASSESSMENT:**
- If your energy is ≤35: You are in CRITICAL condition and MUST get substantial food immediately
- If your energy is 36-60: You need proper nourishment soon
- If your energy is 61-100: You can make casual food choices

**MEAL TIMING RULES (STRICTLY ENFORCED):**
- **CRITICAL: NO PACKED LUNCHES OR MEAL PREP** - You cannot prepare meals in advance or bring packed lunches to work
- **IMMEDIATE CONSUMPTION ONLY** - All meals must be cooked and eaten immediately when made
- LUNCH TIME (11:00-14:00): You MUST choose restaurants (Fried Chicken Shop, Local Diner) for proper meals
- DINNER TIME (17:00-21:00): You MUST choose restaurants (Fried Chicken Shop, Local Diner) for proper meals  
- BREAKFAST TIME (06:00-10:00): Coffee shops are acceptable for light breakfast
- OTHER TIMES: Coffee shops only for beverages and light snacks

**COFFEE SHOP RESTRICTION:**
- Coffee shops (The Coffee Shop) CANNOT provide proper meals for lunch or dinner
- Coffee shops only serve beverages, pastries, and light snacks (+5 energy max)
- If you need substantial energy restoration, you MUST choose a restaurant

Location Information:
- Type: {location_type}
- Current Offers: {location_offers}
- Base Price: ${base_price:.2f}
- Current Price: ${current_price:.2f}

Recent Purchases and Experiences:
{recent_purchase_history}

Think naturally about whether you want to make a purchase at this location. Consider your needs, your financial situation, the prices, and your recent experiences. **PRIORITIZE YOUR ENERGY NEEDS - if your energy is critically low (≤35), you cannot afford to choose inadequate food options.**

After your natural thoughts, you MUST end with this exact structured format:

PURCHASE: [YES or NO]
ITEM_TYPE: [groceries/meal/beverages_and_snacks/misc]
ITEM_DESCRIPTION: [what you want to buy - be specific]
REASONING: [brief explanation of your decision]

Rules for the structured conclusion:
1. PURCHASE must be exactly "YES" or "NO"
2. If PURCHASE is NO, still fill in other fields with "none" or "not applicable"
3. Choose ITEM_TYPE based on location and meal time:
   - groceries: Only at markets/supermarkets (Target, Local Market)
   - meal: Only at restaurants for proper meals (Fried Chicken Shop, Local Diner) during lunch (11:00-14:00) or dinner (17:00-21:00) time
   - beverages_and_snacks: Coffee shops for drinks, pastries, light snacks, or breakfast items
   - misc: Other items not covered above
4. **CRITICAL: Coffee shops (The Coffee Shop) can only serve beverages_and_snacks, NOT full meals for lunch/dinner**
5. **ENERGY PRIORITY: If energy ≤35, you MUST seek substantial food (meal type) at restaurants, NOT coffee shop snacks**
6. Be specific about ITEM_DESCRIPTION (e.g., "weekly groceries including vegetables and dairy" or "fried chicken combo meal" or "coffee and croissant")
7. Consider your actual needs and financial situation

Example Response for Low Energy:
"My energy is at 23/100 which is critically low, and it's 12:00 lunch time. I'm at The Coffee Shop, but coffee and pastries won't give me the substantial energy boost I desperately need. I need a proper meal with +40 energy gain, not just +5 from snacks. I should go to a restaurant instead.

PURCHASE: NO
ITEM_TYPE: not applicable
ITEM_DESCRIPTION: not applicable  
REASONING: Energy critically low - need proper restaurant meal, not coffee shop snacks"

Example Response for Appropriate Coffee Shop Visit:
"It's 8:00 AM and I'm feeling good with 75 energy. I just want a quick coffee and pastry for breakfast before work starts. Perfect timing and energy level for a light coffee shop breakfast.

PURCHASE: YES
ITEM_TYPE: beverages_and_snacks
ITEM_DESCRIPTION: coffee and croissant for breakfast
REASONING: Good energy level and appropriate breakfast time for coffee shop"

Your natural thoughts and structured purchase decision:""",

            "structured_conversation_ending": """
You are {name}, currently in a conversation with {participants} at {location}.

The conversation so far:
{conversation_history}

Your current situation:
- Time: {time:02d}:00
- Energy: {energy_level}/100
- Current plans: {current_plans}
- How long you've been talking: {conversation_duration} minutes

Think naturally about whether you want to continue this conversation or if it feels like a good time to end it. Consider your energy, your schedule, how the conversation is flowing, and whether you have other things to do.

After your natural thoughts, you MUST end with this exact structured format:

CONTINUE_CONVERSATION: [YES or NO]
REASONING: [brief explanation of your decision]

Rules for the structured conclusion:
1. CONTINUE_CONVERSATION must be exactly "YES" or "NO"
2. Consider natural conversation flow - if the topic seems finished or you're getting tired
3. Think about your schedule and commitments
4. Be realistic about energy levels and social stamina

Example Response:
"This has been such a nice chat with Sarah about the new restaurant downtown. I'm enjoying catching up with her, but I can feel my energy starting to dip and I know I need to get to the grocery store before it gets too crowded. The conversation feels like it's winding down naturally anyway, and she mentioned needing to head home soon too.

CONTINUE_CONVERSATION: NO
REASONING: Energy getting low and need to handle errands while stores are less crowded"

Your natural thoughts and structured decision:""",

            "structured_location_visit": """
You are {name}, traveling to {destination} when you encounter {encountered_location}.

Your current situation:
- Original destination: {destination}
- Encountered location: {encountered_location}
- Location type: {location_type}
- Time: {time:02d}:00
- Energy: {energy_level}/100
- Money: ${money:.2f}
- Urgency to reach destination: {urgency_level}

Location offers:
{location_offers}

Your schedule and plans:
{current_plans}

Think naturally about whether you want to stop at this location during your travel. Consider your needs, your schedule, any special offers, and how urgent your original destination is.

After your natural thoughts, you MUST end with this exact structured format:

VISIT_LOCATION: [YES or NO]
VISIT_DURATION: [QUICK/REGULAR/NONE]
ACTION_PLAN: [what you plan to do if visiting]
REASONING: [brief explanation of your decision]

Rules for the structured conclusion:
1. VISIT_LOCATION must be exactly "YES" or "NO"
2. VISIT_DURATION: QUICK (5-10 min), REGULAR (15-30 min), or NONE
3. Consider your urgency, needs, and any special offers
4. Be realistic about time constraints

Example Response:
"Oh, I'm passing by the coffee shop and I could really use a caffeine boost before my meeting. I have about 20 minutes before I need to be at the office, and a quick coffee stop would actually help me be more alert for the meeting. The line doesn't look too long either.

VISIT_LOCATION: YES
VISIT_DURATION: QUICK
ACTION_PLAN: Grab a coffee and maybe a pastry to go
REASONING: Need energy boost for upcoming meeting and have time for quick stop"

Your natural thoughts and structured decision:""",

            "structured_activity_classification": """
You are {name}, reflecting on your current activity: "{current_activity}"

Your current situation:
- Time: {time:02d}:00
- Location: {location}
- Energy before activity: {energy_before}/100

Think naturally about what you're doing and how it affects your energy and well-being. Consider the physical and mental demands of this activity.

After your natural thoughts, you MUST end with this exact structured format:

ACTIVITY_TYPE: [WORK/TRAVEL/MEAL/REST/SOCIAL/ERRANDS/EXERCISE]
ENERGY_IMPACT: [HIGH_DRAIN/MODERATE_DRAIN/LOW_DRAIN/NEUTRAL/LOW_BOOST/MODERATE_BOOST/HIGH_BOOST]
REASONING: [brief explanation of the classification]

Rules for the structured conclusion:
1. Choose the most appropriate ACTIVITY_TYPE from the list
2. ENERGY_IMPACT should reflect how this activity affects your energy
3. Consider both physical and mental aspects of the activity

Example Response:
"I'm having lunch at this nice restaurant after a busy morning at work. The food is delicious and I'm finally getting to sit down and relax for a bit. It's exactly what I needed to recharge before the afternoon meetings. Taking this break to eat and unwind is definitely helping me feel more energized.

ACTIVITY_TYPE: MEAL
ENERGY_IMPACT: MODERATE_BOOST
REASONING: Eating a good meal and taking a break from work is restoring my energy"

Your natural thoughts and structured classification:""",

            "structured_memory_relevance": """
You are {name}, considering this memory: "{memory_content}"

Your current context:
- Current situation: {current_context}
- What you're thinking about: {query_context}
- Time of memory: {memory_time}
- Current time: {current_time}

Think naturally about how relevant this memory is to your current situation or what you're thinking about. Consider the content, timing, and emotional significance.

After your natural thoughts, you MUST end with this exact structured format:

RELEVANCE: [HIGH/MEDIUM/LOW/NONE]
MEMORY_TYPE: [PERSONAL/SOCIAL/WORK/FOOD/LOCATION/ACTIVITY/OTHER]
REASONING: [brief explanation of the relevance]

Rules for the structured conclusion:
1. RELEVANCE should reflect how useful this memory is right now
2. MEMORY_TYPE should categorize what kind of memory this is
3. Consider both direct relevance and emotional significance

Example Response:
"This memory about trying the new pasta place with my colleague last week is pretty relevant right now since I'm deciding where to have lunch today. I remember really enjoying the food there and the prices were reasonable. It's exactly the kind of information I need to make a good choice about where to eat.

RELEVANCE: HIGH
MEMORY_TYPE: FOOD
REASONING: Directly relevant to current lunch decision-making and contains useful experience"

Your natural thoughts and structured assessment:""",
        }
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get a prompt based on type and context."""
        if prompt_type == 'daily_plan':
            return f"""You are {kwargs['name']}, a {kwargs['age']} year old {kwargs['occupation']} living in {kwargs['residence']}.
Current time: {kwargs['current_time']}:00
Current location: {kwargs['current_location']}

Your current state:
- Energy: {kwargs['energy_level']}/100
- Money: ${kwargs['money']:.2f} (You will receive ${kwargs['daily_income']:.2f} at the end of the day)
- Grocery level: {kwargs['grocery_level']}/100

Work schedule: {kwargs['work_schedule_start']}:00 - {kwargs['work_schedule_end']}:00 at {kwargs['workplace']}

Recent activities:
{kwargs['recent_activities']}

Available locations:
{kwargs['available_locations']}

Create a detailed daily plan that considers:
1. Your work schedule
2. Your energy levels
3. Your financial situation (remember you'll receive ${kwargs['daily_income']:.2f} at the end of the day)
4. Your grocery needs
5. Your social needs

Format your plan as a list of activities with times, locations, and reasons.
"""
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        try:
            # Format food locations prices if they exist
            if 'food_locations_prices' in kwargs:
                prices_str = []
                for loc_name, price_info in kwargs['food_locations_prices'].items():
                    price_line = f"- {loc_name}: ${price_info['current_price']:.2f}"
                    if price_info['has_discount']:
                        price_line += f" (${price_info['base_price']:.2f} base price, ${price_info['discount_amount']:.2f} discount)"
                    prices_str.append(price_line)
                kwargs['food_locations_prices_formatted'] = "\n".join(prices_str)
                
                # Add current FCS price if available
                if "Fried Chicken Shop" in kwargs['food_locations_prices']:
                    kwargs['current_fcs_price'] = kwargs['food_locations_prices']["Fried Chicken Shop"]['current_price']
                else:
                    kwargs['current_fcs_price'] = 20.00  # Default base price
            else:
                kwargs['food_locations_prices_formatted'] = "No food location prices available"
                kwargs['current_fcs_price'] = 20.00  # Default base price
            
            return self.prompts[prompt_type].format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameters for {prompt_type}: {list(e.args)}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt: {str(e)}")

    def get_discount_info(self, location_name: str, current_day: int) -> str:
        """Get discount information based on location and current day."""
        if location_name == "Fried Chicken Shop":
            if current_day in [3, 4]:  # Wednesday and Thursday
                return "Special Offer: 20% discount on all meals today! (Wednesday/Thursday Special)"
            else:
                return "No special discounts available today. Regular prices in effect."
        return ""  # Return empty string for other locations

    def get_location_context(self, location_name: str, current_day: int) -> str:
        """Get context about a specific location, prioritizing system rules and constraints."""
        try:
            context = "System Rules and Constraints:\n"
            
            # Energy System Rules
            context += "\nEnergy System:\n"
            context += f"- Maximum Energy: {ENERGY_MAX}\n"
            context += f"- Minimum Energy: {ENERGY_MIN}\n"
            context += f"- Energy Decay: {ENERGY_DECAY_PER_HOUR} per hour\n"
            context += f"- Work Cost: {ENERGY_COST_WORK_HOUR} per hour\n"
            context += f"- Travel Cost: {ENERGY_COST_PER_HOUR_TRAVEL} per hour\n"
            context += f"- Idle Cost: {ENERGY_COST_PER_HOUR_IDLE} per hour\n"
            
            # Food and Energy Rules
            context += "\nFood and Energy Rules:\n"
            context += f"- Home Meal: +{ENERGY_GAIN_HOME_MEAL} energy\n"
            context += f"- Restaurant Meal: +{ENERGY_GAIN_RESTAURANT_MEAL} energy\n"
            context += f"- Snack/Beverage: +{ENERGY_GAIN_SNACK} energy\n"
            context += f"- Sleep: +{ENERGY_GAIN_SLEEP} energy\n"
            context += f"- Nap: +{ENERGY_GAIN_NAP} energy\n"
            context += f"- Conversation: +{ENERGY_GAIN_CONVERSATION} energy\n"
            
            # Money and Cost Rules
            context += "\nMoney and Cost Rules:\n"
            context += "- Restaurant Meal: $15\n"
            context += "- Snack/Beverage: $5\n"
            context += "- Groceries: $10 per unit\n"
            
            # Work Schedule Rules
            context += "\nWork Schedule Rules:\n"
            context += "- Standard Work Hours: 9:00-17:00\n"
            context += "- Must have sufficient energy to work\n"
            context += "- Work provides daily wage\n"
            
            # Location-specific Information
            context += f"\nLocation: {location_name}\n"
            
            # Food Options by Location
            if location_name == 'Coffee Shop':
                context += "\nFood Options:\n"
                context += "- Serves snacks and beverages only (not full meals)\n"
                context += "- Snacks/beverages cost $5 and provide a small energy boost\n"
                context += "- Open for breakfast and afternoon snacks\n"
            elif location_name in ['Fried Chicken Shop', 'Local Diner']:
                context += "\nFood Options:\n"
                context += "- Serves full meals (breakfast, lunch, dinner)\n"
                context += "- Meals cost $15 and provide significant energy\n"
                context += "- Open during regular meal hours\n"
            elif location_name == 'Grocery Store':
                context += "\nFood Options:\n"
                context += "- Sells groceries for home cooking\n"
                context += "- Also offers quick snacks for $5\n"
                context += "- Open during regular business hours\n"
            
            # Location Hours
            context += "\nOperating Hours:\n"
            if location_name in ['Fried Chicken Shop', 'Local Diner']:
                context += "- Open: 7:00-21:00 (Meal Service)\n"
            elif location_name == 'Coffee Shop':
                context += "- Open: 6:00-20:00 (Snack Service)\n"
            elif location_name == 'Grocery Store':
                context += "- Open: 8:00-22:00\n"
            elif location_name == 'Home':
                context += "- Always accessible to residents\n"
            elif location_name == 'Workplace':
                context += "- Open during work hours (9:00-17:00)\n"
            
            # Add any special events or discounts
            discount_info = self.get_discount_info(location_name, current_day)
            if discount_info:
                context += f"\n{discount_info}"
            
            return context
            
        except Exception as e:
            print(f"Error in get_location_context: {str(e)}")
            return f"Location: {location_name}\nError retrieving context."
