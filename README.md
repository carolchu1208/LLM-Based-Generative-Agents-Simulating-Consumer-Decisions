# LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior

<img width="2275" height="1234" alt="agents drawio-1" src="https://github.com/user-attachments/assets/9786d838-7e82-4c94-9707-b84a86e62995" />

---

## 📄 Paper

**Authors:** Man-Lin Chu, Lucian Terhorst, Kadin Reed, Tom Ni, Weiwei Chen, Rongyu Lin*

*Corresponding author: Rongyu Lin (Rongyu.Lin@quinnipiac.edu)

**Status:** Preprint (2025)

**Built on:** [Generative Agents: Interactive Simulacra of Human Behavior](https://github.com/joonspk-research/generative_agents) (Park et al., 2023)

> **Abstract:** This paper presents a novel multi-agent LLM framework for simulating consumer decision-making and analyzing marketing strategies in virtual environments. Building on the foundational Generative Agents framework by Park et al. (2023), we extend LLM-powered simulations from general human behavior to the specialized domain of marketing and consumer behavior. Our system orchestrates 11 autonomous agents in a virtual town with realistic economic constraints (energy systems, wage-based income, food consumption) to study how price discount promotions influence purchasing behavior, brand loyalty, and emergent social dynamics. We implement a 20% midweek discount strategy at a Fried Chicken Shop across a 7-day simulation and observe emergent behaviors including social coordination, word-of-mouth information diffusion, and consumer substitution effects—without hard-coded rules. The framework demonstrates how LLM-based agents can generate realistic consumer behavior patterns for marketing strategy evaluation.

**Citation:**
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

## 📖 Key Contributions

- **LLM-Powered Autonomous Agents:** Each agent possesses memory retrieval (recency, importance, relevance scoring), planning, reflection, and conversational abilities for realistic decision-making.

- **Parallel Multi-Agent Town Simulation:** 11 agents navigate a 10×10 grid town with 10 locations (dining, shopping, work, leisure, residential) using thread-safe parallel execution.

- **Marketing Strategy Evaluation:** Embedded 20% discount at Fried Chicken Shop on Days 3-4 to analyze impact on revenue, market share shifts, and consumer loyalty formation.

- **Emergent Social Behaviors:** Agents demonstrate social coordination (group dining), word-of-mouth diffusion, loyalty patterns, and substitution effects without hard-coded behavioral rules.

---

## 🔬 Code Overview

### System Architecture & Methodology

This repository implements the paper's multi-agent simulation methodology through a modular pipeline:

**System Execution Flow:**
```
Agent Configuration → Prompt Generation → LLM Planning → Plan Execution → State Tracking → Data Recording
        ↓                    ↓                  ↓                ↓                ↓              ↓
agent_configuration  prompt_manager  llm_deepseek_manager  PlanExecutor  shared_trackers  metrics_manager
                                                                 ↓
                                                     ┌───────────┴───────────┐
                                                     ↓                       ↓
                                            Location Tracker          Memory Manager
                                            (Position & State)      (Conversations & Reflections)
```

**Pipeline Stages:**

1. **Configuration & Initialization** (`agent_configuration.json`, `simulation_constants.py`)
   - Defines 11 agent personas with demographics, occupations, relationships, residences
   - Specifies 10×10 grid town map with 10 locations and coordinates
   - Configures restaurant menus (food items, prices, energy values)
   - Sets discount strategy: 20% off at Fried Chicken Shop on Days 3-4
   - Establishes energy system parameters: -5/hour decay, +45 from meals, +10 from snacks, sleep→100
   - **Daily schedule**: Agents sleep hours 23-6, wake at hour 7 for planning

2. **Prompt Generation** (`prompt_manager.py`)
   - **Timing**: Executes at hour 7 each day when agents wake up
   - Constructs contextualized prompts with agent state (location, energy, time, relationships)
   - Dynamically displays discount information: "$25 → $20 (20% off on Days 3-4)" only on active days
   - Implements prompt caching for DeepSeek API efficiency

3. **LLM Planning** (`llm_deepseek_manager.py`)
   - **Timing**: Daily plans generated at hour 7 after agents wake from sleep
   - Sends prompts to DeepSeek API for daily plan generation (hours 7-22, skipping sleep hours 23-6)
   - Handles retry logic, error recovery, and JSON response parsing
   - Agents autonomously decide activities: dining, working, shopping, socializing

4. **Plan Execution** (`simulation_execution_classes.py`)
   - **Timing**: Execution begins at hour 7 after planning, continues through hour 22
   - **PlanExecutor**: Orchestrates hourly action execution in parallel (threading)
   - **Movement**: BFS pathfinding on 10×10 grid with coordinate tracking
   - **Dining**: Food ordering → menu validation → payment → energy restoration
   - **Work**: Wage payment ($10-30/hour based on occupation)
   - **Conversation**: Multi-turn dialogues between agents at shared locations
   - **Sleep** (hours 23-6): Energy restoration to 100, automatic sleep force when energy ≤ 0

5. **State Tracking** (`shared_trackers.py`)
   - **LocationTracker**: Thread-safe tracking of agent positions with coordinates
   - **StateManager**: Tracks agent states (planning, moving, eating, working, conversing, sleeping)

6. **Memory & Social System** (`memory_manager.py`)
   - Stores observations, conversations, reflections with timestamps
   - Retrieves memories using scoring: recency (exponential decay) + importance (LLM-rated 1-10) + relevance (embedding similarity)
   - Enables agents to remember past interactions and form preferences

7. **Validation & Safety** (`menu_validator.py`)
   - Validates food orders against actual menus to prevent LLM hallucination
   - Provides emergency alternatives when invalid orders occur
   - Prevents agent death from starvation

8. **Data Recording** (`metrics_manager.py`)
   - Records sales transactions: location, item, price, discount status, timestamp
   - Generates daily business summaries: revenue by location, item popularity
   - Enables comparative analysis between baseline and discount periods

### Repository Structure

```
LLMAgentsTown_experiment/           # Core simulation code
├── main_simulation.py              # Entry point: orchestrates 7-day, 168-hour simulation
├── simulation_execution_classes.py # Agent, Location, PlanExecutor classes
├── agent_configuration.json        # Agent personas, town map, menus, discount config
├── simulation_constants.py         # Energy/economic parameters
├── prompt_manager.py               # Dynamic prompt generation with discount display
├── llm_deepseek_manager.py         # DeepSeek API interface with caching
├── memory_manager.py               # Memory retrieval & conversation orchestration
├── metrics_manager.py              # Sales tracking & business analytics
├── menu_validator.py               # LLM response validation
├── shared_trackers.py              # LocationTracker & StateManager
├── simulation_types.py             # Data structures & utilities (BFS pathfinding)
├── CRITICAL_PARAMETERS.md          # Detailed parameter documentation
└── debug_file/
    └── debug_with_saved_plans.py   # Debug mode with pre-saved plans

LLMAgentsTown_memory_records/       # Simulation output data
├── simulation_agents/              # Agent memories & state history (JSONL)
├── simulation_conversations/       # Conversation logs (JSONL, Git LFS for >100MB)
├── simulation_daily_summaries/     # Daily business performance (JSON)
├── simulation_metrics/             # Sales transactions & KPIs (JSONL)
└── simulation_plans/               # Saved agent plans for debugging (JSON)

Each output folder contains 2 representative simulation runs:
- 20250716_013356: July 2025 run
- 20251019_125347: October 2025 run
```

---

## ⚙️ Critical System Parameters

> **📋 For detailed parameter documentation, see [`LLMAgentsTown_experiment/CRITICAL_PARAMETERS.md`](LLMAgentsTown_experiment/CRITICAL_PARAMETERS.md)**

### Quick Reference
These values are **calibrated** to match paper results and prevent agent death:

- **Energy System**: Decay 5/hour, Restaurant meals +45, Snacks +10, Sleep sets to 100
- **Agent Count**: 11 agents with specific demographics and relationships
- **Discount Strategy**: 20% off at Fried Chicken Shop on Days 3-4
- **Menu Configuration**: Defined in `agent_configuration.json`

**⚠️ WARNING**: Modifying these parameters may cause simulation failure or prevent reproducibility of paper results. See detailed documentation in `CRITICAL_PARAMETERS.md` before making changes.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- DeepSeek API key ([Get one here](https://platform.deepseek.com/))
- `pip` (Python package installer)

### Installation & Setup

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/carolchu1208/LLM-Based-Generative-Agents-Simulating-Consumer-Decisions.git
cd LLM-Based-Generative-Agents-Simulating-Consumer-Decisions
```

#### 2️⃣ Set Up Virtual Environment & Install Dependencies

**Create and activate virtual environment:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# For macOS/Linux:
source venv/bin/activate
# For Windows:
venv\Scripts\activate

# Install required packages
pip install requests
```

> 💡 **Why virtual environment?** This isolates the simulation's dependencies from your system Python, preventing conflicts.

> ⚠️ **Important**: Always activate the virtual environment before running the simulation!

#### 3️⃣ Configure API Key (Choose One Method)

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

#### 4️⃣ Run Simulation
```bash
# Make sure virtual environment is activated (you should see (venv) in your terminal)
source venv/bin/activate  # Skip if already activated

# Navigate to experiment folder and run
cd LLMAgentsTown_experiment
python3 main_simulation.py
```

### 🐛 Debugging Mode
Use pre-saved agent plans to skip LLM generation for faster debugging:
```bash
cd LLMAgentsTown_experiment/debug_file
python debug_with_saved_plans.py --list-saves    # Show available saved plans
python debug_with_saved_plans.py                 # Run with most recent saves
```
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

## 📚 Related Work & References

### Foundational Framework
This work builds directly on:

**Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023).**
*Generative Agents: Interactive Simulacra of Human Behavior.*
In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23).
[GitHub Repository](https://github.com/joonspk-research/generative_agents) | [Paper PDF](https://arxiv.org/abs/2304.03442)

**Our Extension:** We adapt their agent architecture (memory, planning, reflection) from general human behavior simulation to the specialized domain of marketing and consumer decision-making, adding economic constraints (energy systems, wage-based income), discount strategy evaluation, and business analytics.

### Additional References
- Lichtenstein, D. R., Netemeyer, R. G., & Burton, S. (1997). Deal proneness and consumer behavior: A meta-analytic review. *Journal of Retailing*, 73(3), 331-361.
- Consumer journey frameworks: AIDA → AIDMA → AISAS → AIDEES evolution
- BDI (Belief-Desire-Intention) agent architecture foundations

---

## 👩‍💻 Authors

**First Author: Man Lin Chu**
📧 mchu@clarku.edu | manlin.chu1998@gmail.com

**Co-authors:**
Lucian Terhorst, Kadin Reed, Tom Ni, Weiwei Chen, Rongyu Lin

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
