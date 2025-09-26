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

**Co-authors:**
Rongyu Lin (Quinnipiac University) - Rongyu.Lin@quinnipiac.edu

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
