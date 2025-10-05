import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI # Use the modern, installed package

# --- SECURITY STEP: Load API Key from Streamlit Secrets ---
# The code checks for the secret and sets it as an OS environment variable
# so LangChain's ChatOpenAI can automatically find it.
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
else:
    st.error("OpenAI API Key not found. Please configure the 'OPENAI_API_KEY' secret in Streamlit Cloud.")


# --- Data, Prompt, and Analysis Logic ---

def load_budget_data():
    # Mock data function
    data = {
        "Department": ["Marketing", "Sales", "Engineering", "HR", "IT", "Operations"],
        "Allocated Budget": [80000, 60000, 150000, 40000, 50000, 70000],
        "Actual Spending": [95000, 58000, 140000, 30000, 60000, 85000]
    }
    return pd.DataFrame(data)

budget_template = PromptTemplate.from_template("""
You are a financial planning assistant. Given the following department-level budget data:
{budget_table}
Tasks:
1. Identify departments that overspent or underspent.
2. Suggest budget reallocation for next quarter.
3. Recommend cost-saving strategies.
Present your suggestions in clear bullet points.
""")

def analyze_budget(budget_df, prompt_template):
    # Check for API key presence before calling LLM
    if not os.environ.get('OPENAI_API_KEY'):
        return budget_df, "Error: API Key is missing. Cannot run analysis."
    
    table = budget_df.to_string(index=False)
    llm = ChatOpenAI(temperature=0.3)
    prompt = budget_template.format(budget_table=table)
    
    # Use .invoke() for modern LangChain prediction
    summary = llm.invoke(prompt).content 
    return budget_df, summary


# --- Streamlit Application Layout ---

st.set_page_config(layout="wide")
st.title("📊 Budget Optimization Agent")

df = load_budget_data()

st.subheader("💵 Current Budget Overview")
st.dataframe(df) # Display the raw data

if st.button("🧠 Run AI Budget Analysis"):
    # Use a spinner to show progress while calling the external API
    with st.spinner("Calling GPT to generate recommendations..."):
        df_result, result_text = analyze_budget(df, budget_template)
    
    st.subheader("📈 Budget Performance Chart")
    
    # Create and display the Matplotlib chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_result["Department"], df_result["Allocated Budget"], label="Allocated Budget", alpha=0.6, color='skyblue')
    ax.bar(df_result["Department"], df_result["Actual Spending"], label="Actual Spending", alpha=0.6, color='salmon')
    ax.set_ylabel("USD")
    ax.set_title("Departmental Budget vs. Actual Spending")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("✨ AI Recommendations & Cost-Saving Measures")
    st.markdown(result_text)
