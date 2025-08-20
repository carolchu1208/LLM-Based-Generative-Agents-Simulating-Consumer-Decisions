# LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior

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

## 🏗️ Project Structure  
├── LLMAgentsTown_experiment (Simulation codebase)
                            ├── Agents_Config_Test.json (Ageng Persona Setup)
                            ├── Memory_Manager.py(Memory System Setup)
                            ├── Metrics_Manager.py(Purchase Metrics System Setup)
                            ├── Prompt_Manager.py(Prompt System Setup)
                            ├── deepseek_model_manager.py(Deepseek API Response System Setup)
                            ├── simulation_constants.py(Simulation Basic System Setup, e.g:Energy, Money levels)
                            ├── shared_trackers.py(Agent Location Traking System)
                            ├── thread_safe_base.py(Memory Saveing & Excution Thread Safe Setup)
                            ├── classes.py (Main Execution Functions Setup)
├── LLMAgentsTown_memory_records (Logs of interactions, memory records, and purchase data)
└── README.md ← Project documentation


---

## ⚙️ Getting Started  

### Requirements  
- Python 3.9+  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt


📊 Results & Findings
- Revenue & Market Share:
The 20% discount increased Fried Chicken Shop’s revenue by 51% (Day 2 → Day 3) despite the lower price, showing strong promotional responsiveness.
- Consumer Behavior:
Agents displayed deal promotion proneness (DPP), with varied sensitivity to discounts.
Repeat visits and loyalty persisted beyond the discount period, suggesting habit formation.
Substitution effects observed: Local Diner lost market share, while Coffee Shop remained stable, reflecting differentiated consumer segments.
- Emergent Social Dynamics:
Agents naturally coordinated group dining plans, spreading information peer-to-peer, mimicking word-of-mouth diffusion.

📌 Known Limitations
- LLM Sensitivity: Agents may occasionally hallucinate nonexistent locations or plans.
- Energy & Resource Fidelity: Agents sometimes mis-handle meal planning, requiring fallback recovery logic.
- Demographic Realism: Child and elderly personas exhibited less realistic behavior due to LLM pretraining bias.

📂 Related Work
- Park et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. [GitHub]
- Chu, M.-L. et al. (2025). LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior. (Full paper in repo)

👩‍💻 Author
Man-Lin Chu
M.S. Business Analytics, Clark University
Research focus: Generative AI, consumer behavior simulation, and marketing strategy analysis.
📫 Contact: mchu@clarku.edu | manlin.chu1998@gmail.com

📜 Citation
If you use this repository in your work, please cite:
@article{chu2025multiagent,
  title={LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior},
  author={Chu, Man-Lin and Terhorst, Lucian and Reed, Kadin and Ni, Tom and Chen, Weiwei and Lin, Rongyu},
  year={2025}
}
