import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(layout="wide")
st.title("📊 Budget Optimization Agent")

# Define the expected columns for consistency
EXPECTED_COLUMNS = ["Department", "Allocated Budget", "Actual Spending"]

# Define a simple DataFrame structure for the editor's initial state
EDITOR_INITIAL_DATA = {
    "Department": ["Marketing", "Sales", "Engineering", "HR", "IT"],
    "Allocated Budget": [80000, 60000, 150000, 40000, 50000],
    "Actual Spending": [95000, 58000, 140000, 30000, 60000]
}

# --- SESSION STATE INITIALIZATION ---
# Initialize session state for the editable DataFrame if it doesn't exist
if 'editable_df' not in st.session_state:
    st.session_state.editable_df = pd.DataFrame(EDITOR_INITIAL_DATA)


# --- SECURITY STEP: Load API Key from Streamlit Secrets ---
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
else:
    st.error("OpenAI API Key not found. Please configure the 'OPENAI_API_KEY' secret in Streamlit Cloud.")


# --- LLM PROMPT TEMPLATE ---
budget_template = PromptTemplate.from_template("""
You are a financial planning assistant. Given the following department-level budget data:
{budget_table}

Tasks:
1. Identify departments that overspent or underspent.
2. Suggest budget reallocation for next quarter.
3. Recommend cost-saving strategies.

Present your suggestions in clear bullet points.
""")

# --- ANALYSIS LOGIC ---
def analyze_budget(budget_df, prompt_template):
    # Check for API key presence before calling LLM
    if not os.environ.get('OPENAI_API_KEY'):
        return budget_df, "Error: API Key is missing. Cannot run analysis."
    
    # Ensure money columns are numeric before sending to LLM
    budget_df["Allocated Budget"] = pd.to_numeric(budget_df["Allocated Budget"], errors='coerce')
    budget_df["Actual Spending"] = pd.to_numeric(budget_df["Actual Spending"], errors='coerce')
    budget_df = budget_df.dropna(subset=["Allocated Budget", "Actual Spending"])
    
    table = budget_df.to_string(index=False)
    llm = ChatOpenAI(temperature=0.3)
    prompt = budget_template.format(budget_table=table)
    
    # Use .invoke() for modern LangChain prediction
    summary = llm.invoke(prompt).content 
    return budget_df, summary


# --- DATA LOADING FUNCTIONS ---

def load_uploaded_data(uploaded_file):
    """Loads CSV or Excel data from the uploader."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        
        # Assume the first three columns are the budget data
        if df.shape[1] < 3:
            st.error("File must contain at least three columns: Department, Allocated Budget, Actual Spending.")
            return None

        # Standardize columns based on the expected input
        df.columns = EXPECTED_COLUMNS + list(df.columns[3:])
        df = df[EXPECTED_COLUMNS]
        
        # Ensure money columns are numeric
        df["Allocated Budget"] = pd.to_numeric(df["Allocated Budget"], errors='coerce')
        df["Actual Spending"] = pd.to_numeric(df["Actual Spending"], errors='coerce')
        
        return df.dropna(subset=EXPECTED_COLUMNS[1:])

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# -------------------------------------------------------------
# --- STREAMLIT APPLICATION LAYOUT ---
# -------------------------------------------------------------

data_input_mode = st.radio(
    "Select Data Input Mode:",
    ('Upload File', 'Manual Data Entry')
)

df = None

if data_input_mode == 'Upload File':
    uploaded_file = st.file_uploader(
        "Upload Budget Data (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        key="uploader_key",
        help="File must have exactly three columns in this order: Department, Allocated Budget, Actual Spending."
    )
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)

elif data_input_mode == 'Manual Data Entry':
    st.subheader("📝 Edit Budget Data Manually")
    st.info("Edit values, add new rows (➕), or delete rows (🗑️) as needed.")
    
    # Use st.data_editor to allow adding/deleting rows
    edited_df = st.data_editor(
        st.session_state.editable_df,
        column_config={
            "Department": st.column_config.TextColumn("Department"),
            "Allocated Budget": st.column_config.NumberColumn("Allocated Budget", format="%,d"),
            "Actual Spending": st.column_config.NumberColumn("Actual Spending", format="%,d")
        },
        num_rows="dynamic", # Key setting to allow adding/deleting rows
        hide_index=True
    )
    # Update the session state and use the edited DataFrame for analysis
    st.session_state.editable_df = edited_df
    df = edited_df.copy()


# --- DISPLAY DATA AND RUN ANALYSIS ---

if df is not None and not df.empty:
    st.subheader("💵 Current Budget Overview")
    
    # Display the final, formatted DataFrame used for analysis
    st.dataframe(
        df.style.format(
            {
                "Allocated Budget": "{:,.0f}",
                "Actual Spending": "{:,.0f}"
            }
        ),
        hide_index=True
    )

    # 2. Analysis Button and Results Section
    if st.button("🧠 Run AI Budget Analysis"):
        if 'OPENAI_API_KEY' not in os.environ:
             st.error("API Key is missing. Please configure the 'OPENAI_API_KEY' secret in Streamlit Cloud.")
        else:
            with st.spinner("Calling GPT to generate recommendations..."):
                # Pass the data to the analysis function
                df_result, result_text = analyze_budget(df.copy(), budget_template)
            
            # --- Chart Display ---
            st.subheader("📈 Budget Performance Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df_result["Department"], df_result["Allocated Budget"], label="Allocated Budget", alpha=0.6, color='skyblue')
            ax.bar(df_result["Department"], df_result["Actual Spending"], label="Actual Spending", alpha=0.6, color='salmon')
            
            # Add Comma Separator to Y-Axis
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) 
            
            ax.set_ylabel("USD")
            ax.set_title("Departmental Budget vs. Actual Spending")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # --- AI Recommendations Display ---
            st.subheader("✨ AI Recommendations & Cost-Saving Measures")
            st.markdown(result_text)
            
else:
    # Display instruction when no file is uploaded or data is entered
    st.info("Please select an input mode and provide data to begin the analysis.")
