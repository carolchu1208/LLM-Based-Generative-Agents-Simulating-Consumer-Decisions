# LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior

<img width="2275" height="1234" alt="agents drawio-1" src="https://github.com/user-attachments/assets/9786d838-7e82-4c94-9707-b84a86e62995" />

---

## 📖 Overview
This repository implements a **multi-LLM generative agent framework** designed to simulate consumer decision-making and social interactions in a virtual town environment.
The project evaluates how marketing strategies—particularly **price discount promotions**—influence consumer behavior, loyalty, and emergent social dynamics.

Building on the [Generative Agents framework by Park et al. (2023)](https://github.com/joonspk-research/generative_agents), this research extends LLM-powered multi-agent simulations to the **marketing and consumer behavior domain**.

---

## ✨ Key Contributions
- **LLM-Powered Agents:**
  Each agent is equipped with memory, planning, reflection, and conversational abilities, enabling realistic decision-making and social interaction.

- **Parallel Multi-Agent Town Simulation:**
  11 agents navigate across 10 locations (dining, shopping, work, leisure, and residential), making daily plans and interacting in real-time with **thread-safe parallel execution**.

- **Price Discount Strategy Analysis:**
  A midweek 20% discount at a Fried Chicken Shop was embedded into the simulation to study its impact on revenue, market share shifts, and consumer loyalty formation.

- **Emergent Behaviors:**
  - Social coordination through conversations (e.g., planning group dining events).
  - Word-of-mouth style information diffusion.
  - Consumer loyalty, substitution effects, and habit formation—without hard-coded rules.

---

## 🏗️ Repository Structure

### 📂 Core Simulation Code
```
LLMAgentsTown_experiment/
├── main_simulation.py                 # 🚀 Main simulation runner - Start here!
├── simulation_execution_classes.py    # 🤖 Agent, Location, PlanExecutor classes
├── menu_validator.py                  # 🛡️ LLM response and menu validation system - prevent unmatch food needs
├── simulation_constants.py            # ⚙️ Critical system parameters (energy, costs setup, etc)
├── agent_configuration.json           # 👥 Agent personas & town configuration
├── memory_manager.py                  # 🧠 Agent memory & conversation system
├── metrics_manager.py                 # 📊 Business analytics & performance tracking
├── prompt_manager.py                  # 💬 LLM prompt (cache rules, templates)
├── llm_deepseek_manager.py            # 🔗 DeepSeek API interface
├── simulation_types.py                # 📋 Core data types & utilities
└── debug_file/                        # 🐛 Debugging utilities
    └── debug_with_saved_plans.py      # Debug tool for reproducible testing
```

### 📂 Simulation Data Storage
```
LLMAgentsTown_memory_records/
├── simulation_agents/          # Agent memory records & state history
├── simulation_conversations/   # Conversation logs & social interactions
├── simulation_daily_summaries/ # Daily business performance reports
├── simulation_metrics/         # Detailed analytics & KPI tracking
└── simulation_plans/          # Saved agent plans for debugging
```

---

## ⚙️ Critical System Parameters

### 🔋 Energy System Constants (**DO NOT MODIFY**)
These values are **calibrated** to match paper results and prevent agent death:

```python
# Energy Thresholds (simulation_constants.py)
ENERGY_MAX = 100                    # Maximum agent energy
ENERGY_THRESHOLD_LOW = 20           # 🚨 Critical: Triggers food-seeking behavior
ENERGY_THRESHOLD_FOOD = 25          # Preventive food planning threshold

# Energy Decay Rates (per simulation hour)
ENERGY_DECAY_PER_HOUR = 5          # 🚨 Critical: Base energy loss rate
ENERGY_COST_WORK_HOUR = 5          # Energy cost during work hours
ENERGY_COST_PER_STEP = 1           # Energy cost per movement step

# Energy Recovery Rates
ENERGY_GAIN_RESTAURANT_MEAL = 45    # 🚨 Critical: Restaurant meal recovery
ENERGY_GAIN_HOME_MEAL = 25          # Home cooking recovery
ENERGY_GAIN_SNACK = 10             # Snack energy gain
ENERGY_GAIN_NAP = 15               # Nap recovery during work hours
ENERGY_GAIN_CONVERSATION = 5        # Social interaction energy gain
# Sleep: Sets energy to ENERGY_MAX (100) every hour during 23:00-06:00
```

**⚠️ WARNING**: Modifying energy values may cause agent starvation and simulation failure!

### 🍽️ Menu Configuration (**DO NOT MODIFY**)
Restaurant menus are **precisely defined** in `Stability_Agents_Config_Test.json`:

```json
"Fried Chicken Shop": {
  "menu": {
    "lunch": {
      "available_hours": [11, 12, 13, 14],
      "item": "Fried Chicken Tender Meal Set",
      "base_price": 20.0
    },
    "dinner": {
      "available_hours": [17, 18, 19, 20],
      "item": "Fried Chicken Wings Meal Set",
      "base_price": 20.0
    },
    "snack": {
      "available_hours": [10, 15, 16, 21, 22],
      "item": "Fried Chicken Nugget 6pcs",
      "base_price": 10.0
    }
  },
  "discount": {
    "type": "percentage",
    "value": 20,           # 🚨 Critical: 20% discount rate
    "days": [3, 4]         # 🚨 Critical: Discount applied on days 3-4
  }
}
```

### 👥 Agent Configuration (**CALIBRATED - DO NOT MODIFY**)
11 agents with **specific demographics** required for paper results:
- **Kevin Chen**: Fried Chicken Shop supervisor (age 23, income $18.50/hour)
- **Sophie Martinez**: Local Market owner (age 27, income $6000/month)
- **David Kim & Lisa Kim**: Married couple (tech worker + marketing manager)
- **Rebecca Queen**: Fried Chicken employee (age 22)
- **Alex Thompson & Jordan Lee**: Roommates (barista + designer)
- **Maria Rodriguez & Sarah Chen**: Dating couple (chef + waitress)
- **Mike Johnson**: Retail manager (age 31)
- **Emma Wilson**: Marketing assistant (age 28)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- DeepSeek API key ([Get one here](https://platform.deepseek.com/))

### Installation & Setup

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/carolchu1208/LLM-Based-Generative-Agents-Simulating-Consumer-Decisions.git
cd LLM-Based-Generative-Agents-Simulating-Consumer-Decisions
```

#### 2️⃣ Configure API Key (Choose One Method)

**Method A: Temporary (Current Session Only)**
```bash
export DEEPSEEK_API_KEY='your-api-key-here'
```
> ⚠️ This only works for the current terminal session. You'll need to set it again if you close the terminal.

**Method B: Permanent (Recommended)**

For **macOS/Linux** users:
```bash
# Add to your shell configuration file
echo 'export DEEPSEEK_API_KEY="your-api-key-here"' >> ~/.bashrc

# Apply changes immediately
source ~/.bashrc

# Verify it's set
echo $DEEPSEEK_API_KEY
```

> 💡 **Note**: If you're using **zsh** (default on newer macOS), replace `~/.bashrc` with `~/.zshrc`

For **Windows** users:
```powershell
# PowerShell
[System.Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', 'your-api-key-here', 'User')

# Restart terminal and verify
echo $env:DEEPSEEK_API_KEY
```

**⚠️ IMPORTANT**: You **must** set the API key before running the simulation. The code will raise an error if `DEEPSEEK_API_KEY` is not set.

#### 3️⃣ Run Simulation
```bash
cd LLMAgentsTown_experiment
python main_simulation.py
```

### 🐛 Debugging Mode
Use pre-saved agent plans to skip LLM generation for faster debugging:
```bash
cd LLMAgentsTown_experiment/debug_file
python debug_with_saved_plans.py --list-saves    # Show available saved plans
python debug_with_saved_plans.py                 # Run with most recent saves
```

---

## 📊 Expected Results & Findings

### 💰 Revenue & Market Share Impact
- **51% revenue increase** for Fried Chicken Shop during discount period (Day 2 → Day 3)
- **Market share shift**: Fried Chicken Shop 30% → 41%, Local Diner 62% → 48%
- **Substitution effects**: Local Diner loses customers, Coffee Shop remains stable

### 🧠 Consumer Behavior Patterns
- **Deal Promotion Proneness (DPP)**: Varied agent sensitivity to discounts
- **Habit Formation**: Repeat visits persist beyond discount period
- **Social Coordination**: Natural group dining and word-of-mouth diffusion
- **Loyalty Development**: Emergent brand preferences without hard-coding

### 📈 Business Intelligence Outputs
- Hourly customer traffic patterns
- Agent-specific purchasing behavior
- Conversation topic analysis (food, work, social themes)
- Relationship strength tracking between agents

---

## 🛡️ Agent Death Prevention System

This repository includes **robust safeguards** to prevent simulation failure:

### Menu Validation (`menu_validator.py`)
- Validates all agent food requests against actual restaurant menus
- Provides emergency food alternatives when requests fail
- Logs invalid requests for debugging
- Prevents agent starvation from LLM hallucination

### Energy Recovery System
- **Emergency food detection**: When agent energy < 20
- **Automatic residence return**: When agent energy ≤ 0
- **Sleep recovery**: Full energy restoration during night hours (23:00-06:00)
- **Improved energy constants**: Reduced decay, increased meal recovery

---

## 📌 Known Limitations & Solutions

| Issue | Impact | Solution Implemented |
|-------|--------|---------------------|
| **LLM Hallucination** | Agents request non-existent food | Menu validation system |
| **Energy Depletion** | Agents "die" from starvation | Improved energy constants + emergency recovery |
| **Invalid Locations** | Agents try to visit non-existent places | Location validation in prompts |
| **Demographic Bias** | Child/elderly agents act like adults | Acknowledged limitation (future work) |

---

## 🔧 Development & Customization

### ✅ Safe to Modify
- **Agent names** and basic demographics
- **Conversation topics** and social relationships
- **Business hours** for shops
- **Simulation duration** (currently 7 days)
- **Output file locations**

### ⚠️ Modify with Caution
- **Energy constants** (may break agent survival)
- **Menu items** and pricing (affects paper reproducibility)
- **Town layout** and location coordinates
- **Discount timing** and percentages

### ❌ Do NOT Modify
- **Core agent count** (11 agents required)
- **Fried Chicken Shop discount** (Days 3-4, 20% off)
- **Menu availability hours**
- **Energy threshold values**

---

## 📂 Related Work & References

- [Park et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior](https://github.com/joonspk-research/generative_agents)
- Lichtenstein et al. (1997). Deal Promotion Proneness and Consumer Behavior
- Consumer journey frameworks: AIDA → AIDMA → AISAS → AIDEES evolution

---

## 👩‍💻 Authors

**Man-Lin Chu** (First Author)
M.S. Business Analytics, Clark University
📧 mchu@clarku.edu | manlin.chu1998@gmail.com

**Co-authors:**
Lucian Terhorst, Kadin Reed, Tom Ni, Weiwei Chen (Clark University)
Rongyu Lin (Quinnipiac University)

---

## 📜 Citation

If you use this repository in your research, please cite:

```bibtex
@article{chu2025multiagent,
  title={LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior},
  author={Chu, Man-Lin and Terhorst, Lucian and Reed, Kadin and Ni, Tom and Chen, Weiwei and Lin, Rongyu},
  year={2025},
  journal={arXiv preprint},
  url={https://github.com/carolchu1208/LLM-Based-Generative-Agents-Simulating-Consumer-Decisions}
}
```

---

## 🛠️ Technical Support

For technical issues:
1. **Check energy constants** in `simulation_constants.py`
2. **Verify API key** is set: `echo $DEEPSEEK_API_KEY`
3. **Use debug mode** with saved plans for faster testing
4. **Review logs** in `LLMAgentsTown_memory_records/`

**⚡ Quick Test**: Run `python debug_file/debug_with_saved_plans.py` to verify setup

---

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

**🎯 Ready for arXiv submission and academic reproducibility!**
