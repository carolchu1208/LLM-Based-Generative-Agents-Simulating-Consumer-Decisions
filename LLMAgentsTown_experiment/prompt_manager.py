class PromptManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.prompts = {
            "contextual_action": """
As {name}, describe my next action in first person ("I").
Current location: {location}
Time: {time}:00
Energy level: {energy_level}
Grocery level: {grocery_level}
Money available: ${money:.2f}

People nearby: {nearby_agents}
Recent activities: {recent_activities}
Current activity: {current_activity}
Location type: {location_type}
Work time: {is_work_time}
My daily plan: {daily_plan}

Important Rules:
1. To get food from a restaurant/shop, I must physically visit the location
2. I can choose to dine in or take food to go, but no delivery service is available
3. To eat at home, I must either have sufficient grocery level or bring takeout food
4. Grocery stores are for buying groceries to cook at home

Consider:
1. My current needs and energy level
2. My daily plan and commitments
3. Opportunities for natural social interaction
4. The current location and who's around me
5. Recent activities and ongoing tasks

Respond with 1-2 natural sentences in first person, describing my actions and any social interactions that arise naturally.
""",

            "conversation": """I am {name} at {location}. Time is {time}:00.

Current context:
- Location: {location}
- Nearby: {nearby_agents}
- Social context: {social_context}
- Relationship: {relationships}
- History: {shared_history}
- Living situation: {living_arrangements}
- My current activity: {current_activity}
- Recent interactions: {recent_interactions}

Generate a natural conversation that arises from this context. The interaction should:
1. Flow naturally from the current situation
2. Reflect existing relationships and shared history
3. Consider the location and current activities
4. Include appropriate emotional responses and body language
5. Allow for natural adjustments to plans if relevant

Example format:
Me (looking up from my work as David enters the break room): "Hey David! Taking a coffee break too?"
David (heading to the coffee machine): "Yeah, needed a quick breather. That project deadline's coming up fast."
Me (checking the time): "Tell me about it. Say, since we're both here, want to go over those design changes while we have our coffee?"

Keep the conversation natural and contextual, with realistic dialogue and brief action descriptions.""",

            "daily_plan": """
I am {name} starting my day.
Current time: {time}:00
My location: {location}
My energy level: {energy}
My current grocery stock level: {grocery_level}%
Available locations: {available_locations}

My recent activities and context: {recent_activities}

Important Rules:
1. To get food from a restaurant/shop, I must physically visit the location
2. I can choose to dine in or take food to go, but no delivery service is available
3. To eat at home, I must either have sufficient grocery level or bring takeout food
4. Grocery stores are for buying groceries to cook at home

Think as {name} and create a natural plan for the day ahead. Consider:
1. My usual routine and preferences
2. Work or study commitments
3. Personal needs and errands
4. Potential social interactions with friends or colleagues
5. Energy levels and timing of activities
6. Need to physically visit locations for food

Express the plan naturally in first person, as internal thoughts about the day ahead. The plan should reflect personal style and priorities while remaining flexible for natural interactions.

Example:
"I'm ready to start another day! I'll head to the Tech Hub for my usual 9 AM start - got that important project to work on. I think I'll stop by the Local Diner for lunch - might eat there if Sarah's around, or grab something to go if I'm busy. My grocery supply is getting low, so I should stop by Target after work. If I'm not too tired, I could use some exercise at the Community Center Gym in the evening."
""",

            "household_coordination": """
I am {name} at {location}.
Time: {time}:00
Household members present: {members}

Our recent household activities: {recent_activities}
Our work schedules: {work_schedules}
Our shared meal plans: {shared_meals}
Our personal activities: {personal_activities}
Our relationships: {relationships}

Generate a brief household interaction focusing on coordination and shared activities.
Keep it focused on important matters and under 3 exchanges.
""",

            "location_context": {
                "template": """Location Information for {location}:
{description}
Hours: {open_time}:00 to {close_time}:00
{discount_info}""",
                "parameters": ["location", "description", "open_time", "close_time", "discount_info"]
            },
            "location_check": {
                "template": "Check if {location} is appropriate for {name} at {time}:00. Consider: {considerations}",
                "parameters": ["location", "name", "time", "considerations"]
            },
            "satisfaction_rating": {
                "template": "You are {name} evaluating your experience at {location}. Recent experiences: {experiences}. Rate: Overall (1-5), Food (1-5), Price (1-5), Service (1-5), Wait time (mins), Would recommend (yes/no), Return likelihood (1-5).",
                "parameters": ["name", "location", "experiences"]
            },
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

Your daily plan for today is:
{daily_plan}

Location Context:
- Type: {encountered_location_type}
- Current Offers/Discounts: {location_offers}
- Relevance to your needs: {location_relevance}
- Last visit: {last_visit_time}

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
    - There's a special discount you can't miss
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
            """
        }
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get a prompt template and fill it with provided parameters"""
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        try:
            return self.prompts[prompt_type].format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameters for {prompt_type}: {list(e.args)}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt: {str(e)}")
