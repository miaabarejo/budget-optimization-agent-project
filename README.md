# 📊 Budget Optimization Agent

## Project Overview

The **Budget Optimization Agent** is an AI-powered financial web application built to analyze and optimize departmental spending using large language models. It provides quick insights into budget performance, highlights areas of inefficiency, and generates actionable, prioritized recommendations.

### Key Features
* **Data Visualization:** Displays budget vs. actual spending using a clear, comma-formatted stacked bar chart for immediate visual assessment.
* **Intelligent Analysis:** Uses **LangChain** and **OpenAI (GPT-3.5/4)** to interpret financial data (allocated vs. actual spending).
* **Actionable Recommendations:** Generates suggestions for budget reallocation and cost-saving strategies in clear bullet points.
* **Secure Deployment:** Utilizes Streamlit Cloud's secrets management to protect the private OpenAI API key.

### Technology Stack
| Category | Tools |
| :--- | :--- |
| **Framework** | Streamlit |
| **AI/LLM** | LangChain, `langchain-openai` (GPT-3.5/4) |
| **Data/Viz** | Pandas, Matplotlib |
| **Environment** | Python 3.10+, Conda |

***

## 🚀 Live Application

You can access the live, deployed version of the application here:

🌐 Live Demo: [Budget Optimizer on Streamlit](https://budget-optimization-agent-project.streamlit.app/)

***

## ⚙️ Setup and Installation (Local Development)

To run this application on your local machine, follow these steps.

### Prerequisites
* Python 3.10+
* Conda/Anaconda (Recommended for environment management)
* **OpenAI API Key**

### Step 1: Clone the Repository
```bash
git clone [https://github.com/YourUsername/budget-optimization-agent-project.git](https://github.com/YourUsername/budget-optimization-agent-project.git)
cd budget-optimization-agent-project
