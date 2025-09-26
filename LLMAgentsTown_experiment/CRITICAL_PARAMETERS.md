# ⚙️ Critical System Parameters

> **⚠️ WARNING**: These parameters are **calibrated** to match paper results. Modifying them may cause agent starvation, simulation failure, or inability to reproduce research findings.

---

## 🔋 Energy System Constants

All energy constants are defined in `simulation_constants.py`.

### Energy Thresholds
```python
ENERGY_MAX = 100                    # Maximum agent energy
ENERGY_MIN = 0                      # Minimum agent energy
ENERGY_THRESHOLD_LOW = 20           # 🚨 Critical: Triggers food-seeking behavior
ENERGY_THRESHOLD_FOOD = 25          # Preventive food planning threshold
```

### Energy Decay Rates (per simulation hour)
```python
ENERGY_DECAY_PER_HOUR = 5          # 🚨 Critical: Base energy loss rate
ENERGY_COST_WORK_HOUR = 5          # Energy cost during work hours
ENERGY_COST_PER_STEP = 1           # Energy cost per movement step
ENERGY_COST_PER_HOUR_IDLE = 1      # Energy cost while idle
```

### Energy Recovery Rates
```python
ENERGY_GAIN_RESTAURANT_MEAL = 45    # 🚨 Critical: Restaurant meal recovery
ENERGY_GAIN_HOME_MEAL = 25          # Home cooking recovery
ENERGY_GAIN_SNACK = 10             # Snack energy gain
ENERGY_GAIN_NAP = 15               # Nap recovery during work hours (11:00-15:00)
ENERGY_GAIN_CONVERSATION = 5        # Social interaction energy gain
```

### Sleep System
- **Hours**: 23:00-06:00 (automatic)
- **Mechanism**: Sets energy to `ENERGY_MAX` (100) every hour
- **Purpose**: Compensates for natural decay, prevents starvation
- **Note**: Sleep does NOT add energy, it SETS energy to maximum

---

## 🍽️ Menu Configuration

Restaurant menus are defined in `agent_configuration.json` under `town_areas.dining`.

### Menu Structure
Each restaurant must include:
```json
{
  "menu": {
    "breakfast": {
      "available_hours": [7, 8, 9],
      "item": "Item Name",
      "base_price": 15.0
    },
    "lunch": { ... },
    "dinner": { ... },
    "snack": { ... }
  },
  "discount": {
    "type": "percentage",
    "value": 20,
    "days": [3, 4]
  }
}
```

### Critical Parameters
- **Fried Chicken Shop Discount**: 20% on Days 3-4 (DO NOT MODIFY)
- **Available Hours**: Must match meal_type expectations
- **Base Prices**: Calibrated for agent income levels

### Energy-Meal Type Mapping
```python
# Defined in simulation_execution_classes.py
meal_type == "breakfast" → ENERGY_GAIN_RESTAURANT_MEAL (45)
meal_type == "lunch"     → ENERGY_GAIN_RESTAURANT_MEAL (45)
meal_type == "dinner"    → ENERGY_GAIN_RESTAURANT_MEAL (45)
meal_type == "snack"     → ENERGY_GAIN_SNACK (10)
home cooking             → ENERGY_GAIN_HOME_MEAL (25)
```

---

## 👥 Agent Configuration

11 agents with specific demographics are required for paper reproducibility.

### Agent Personas (defined in `agent_configuration.json`)
| Name | Role | Age | Income | Relationships |
|------|------|-----|--------|---------------|
| Kevin Chen | Fried Chicken Shop supervisor | 23 | $18.50/hour | Employee at target restaurant |
| Sophie Martinez | Local Market owner | 27 | $6000/month | Business owner |
| David Kim | Software Engineer | 32 | $5000/month | Married to Lisa Kim |
| Lisa Kim | Marketing Manager | 28 | $4500/month | Married to David Kim |
| Rebecca Queen | Fried Chicken employee | 22 | $17/hour | Works with Kevin Chen |
| Alex Thompson | Barista | 25 | $16/hour | Roommate with Jordan Lee |
| Jordan Lee | Graphic Designer | 26 | $22/hour | Roommate with Alex Thompson |
| Maria Rodriguez | Chef | 30 | $24/hour | Dating Sarah Chen |
| Sarah Chen | Waitress | 24 | $15/hour | Dating Maria Rodriguez |
| Mike Johnson | Retail Manager | 31 | $27/hour | Single |
| Emma Wilson | Marketing Assistant | 28 | $21/hour | Single |

### Critical Demographic Elements
- **Income Diversity**: $15-27/hour ranges ensure varied purchasing power
- **Relationships**: Married couples, dating couples, roommates drive social coordination
- **Occupations**: Varied work schedules create different dining patterns
- **Employee Connections**: Kevin and Rebecca at Fried Chicken Shop provide insider perspective

---

## 🏙️ Town Map & Locations

Town layout is defined in `agent_configuration.json` under `town_areas`.

### Location Categories
1. **Dining**: Fried Chicken Shop, Local Diner, Coffee Shop
2. **Retail & Grocery**: Local Market, Convenience Store
3. **Workplace**: Various offices and restaurants
4. **Leisure**: Park, Community Center
5. **Residential**: Individual homes for each agent

### Coordinate System
- Grid-based (x, y) coordinates
- Travel cost: 1 energy per step moved
- Agents can complete entire path in one simulation hour
- Distance affects energy cost but not time

---

## 🎯 Simulation Parameters

### Time System
```python
SIMULATION_START_DAY = 1           # Start day
SIMULATION_DAYS = 7                # Total simulation days
SIMULATION_START_HOUR = 7          # Start at 7:00 AM
SIMULATION_END_HOUR = 23           # End at 11:00 PM
```

### Money System
```python
STARTING_MONEY = 1000              # Initial agent money
GROCERY_COST_HOME_MEAL = 10        # Cost to cook at home
# Individual meal prices defined in menu configuration
```

### Work Hours
- Standard: 9:00-17:00
- Income earned based on hourly_wage from agent config
- Energy cost: 5 per hour worked

---

## 🛡️ Safety Mechanisms

### Menu Validation (`menu_validator.py`)
- Validates all agent food requests against actual restaurant menus
- Provides emergency food alternatives when requests fail
- Logs invalid requests for debugging
- Prevents agent starvation from LLM hallucination

### Energy Recovery System
- **Emergency Detection**: When agent energy < 20
- **Automatic Return**: When agent energy ≤ 0, forced return to residence
- **Sleep Enforcement**: Automatic sleep during 23:00-06:00 with forced home travel
- **Nap Availability**: During work hours (11:00-15:00) at workplace

### Location Validation
- All locations dynamically loaded from `agent_configuration.json`
- Prompts include only valid locations
- Invalid location requests logged and rejected

---

## 📊 Expected System Behavior

### Daily Energy Cycle (Example)
```
Hour 7:  100 energy (wake from sleep)
Hour 8:  95 (travel: -1, decay: -5, breakfast: +45, net: -5)
Hour 9:  85 (work: -5, decay: -5)
Hour 12: 70 (continue work + decay)
Hour 13: 110 → 100 (lunch: +45, capped at max)
Hour 17: 80 (end work)
Hour 19: 95 (dinner: +45, decay: -5)
Hour 23: 100 (automatic sleep sets to max)
```

### Revenue Impact (Days 2→3 with discount)
- Fried Chicken Shop: ~51% revenue increase
- Market Share: 30% → 41%
- Local Diner: 62% → 48% (substitution effect)

---

## ⚠️ Modification Guidelines

### ✅ Safe to Modify
- Agent names and basic demographics (if not reproducing paper)
- Conversation topics
- Simulation duration
- Output file locations
- New restaurant locations (with proper menu setup)

### ⚠️ Modify with Caution
- Energy constants (may break agent survival)
- Menu items and pricing (affects paper reproducibility)
- Town layout coordinates
- Agent count (affects social dynamics)

### ❌ Do NOT Modify (For Paper Reproducibility)
- Fried Chicken Shop discount (Days 3-4, 20% off)
- Core energy thresholds (20, 25)
- Energy decay rate (5 per hour)
- Restaurant meal recovery (45 energy)
- Sleep mechanism (sets to 100)
- 11 agent configuration
- Agent relationships and roles

---

## 🔧 Troubleshooting

### Agent Death / Starvation
**Symptoms**: Agent energy reaches 0, simulation fails
**Causes**:
- Energy decay too high
- Meal recovery too low
- Sleep not restoring properly
- Agents not eating due to LLM hallucination

**Solutions**:
1. Verify `ENERGY_DECAY_PER_HOUR = 5` in `simulation_constants.py`
2. Check sleep system sets energy to 100 (not adds to current)
3. Review `menu_validator.py` logs for failed food requests
4. Ensure `ENERGY_GAIN_RESTAURANT_MEAL = 45`

### LLM Hallucination
**Symptoms**: Agents request non-existent food or locations
**Causes**:
- LLM generating creative but invalid responses
- Prompts not constraining to valid options

**Solutions**:
1. Check `menu_validator.py` is catching invalid requests
2. Review `prompt_manager.py` includes valid location lists
3. Monitor logs for validation warnings
4. Use fallback responses when needed

### Revenue Anomalies
**Symptoms**: Restaurant revenue doesn't match expected patterns
**Causes**:
- Discount not applied correctly
- Agents avoiding restaurants due to energy issues
- Menu pricing misconfigurations

**Solutions**:
1. Verify discount days in `agent_configuration.json`
2. Check agent energy levels are healthy (>30 average)
3. Review `simulation_metrics/` for dining patterns
4. Confirm menu prices match configuration

---

## 📚 Related Files

- **`simulation_constants.py`**: All energy and cost constants
- **`agent_configuration.json`**: Agent personas, town map, menus, discounts
- **`simulation_execution_classes.py`**: Action execution, energy system logic
- **`menu_validator.py`**: Menu validation and safety checks
- **`prompt_manager.py`**: LLM prompts with rules and constraints
- **`simulation_types.py`**: Core data types, energy calculations

---

## 📞 Support

If parameters need adjustment for your research:
1. Document current values before changes
2. Test with short simulations (1-2 days) first
3. Monitor agent energy levels in logs
4. Review `LLMAgentsTown_memory_records/` for behavior changes
5. Restore original values if simulation becomes unstable

**Remember**: These parameters were calibrated through extensive testing to balance realism, agent survival, and paper reproducibility.