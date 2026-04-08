import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit.streamlit_callback_handler import LLMThoughtLabeler

class AgentLabeler(LLMThoughtLabeler):
    def get_final_agent_thought_label(self) -> str:
        return '✅ Analysis Complete!'
    
    # Use the actual emoji character 🧠 instead of the :thinking_face: text
    def get_initial_thought_label(self) -> str:
        return "🧠 Thinking..."

    def get_tool_label(self, tool, tool_input) -> str:
        return f"🛠️ Processing: {tool}" # Added more detail for your demo

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_thinking = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_0p38m099.json")

# Engine
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
st.set_page_config(page_title = 'Agent',
                   page_icon = '📊',
                   layout = 'wide')

st.markdown("""
    <style>
    /* 1. Global Page & Fade-In Animation */
    .stApp {
        background: radial-gradient(circle at top right, #1a0505, #050505);
        color: #f0f0f0;
        animation: fadeIn 1.2s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* 2. Header & Shimmering Typography */
    h1 {
        background: linear-gradient(90deg, #ff3131, #8b0000, #ff3131);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    /* 3. Sidebar: Clean Containers & Crimson Metrics */
    section[data-testid="stSidebar"] {
        background-color: #0a0000 !important;
        border-right: 1px solid rgba(255, 49, 49, 0.2);
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 49, 49, 0.2) !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }

    [data-testid="stMetricValue"] {
        color: #ff3131 !important;
        font-family: 'Courier New', Courier, monospace;
    }

    /* 4. Chat Messages: Glassmorphism & Slide-Up */
    [data-testid="stChatMessage"] {
        background: rgba(20, 20, 20, 0.7) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 49, 49, 0.1);
        border-radius: 15px !important;
        margin-bottom: 12px;
        animation: fadeInUp 0.5s ease-out forwards;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stChatMessageUser"] { border-right: 4px solid #ff3131 !important; }
    [data-testid="stChatMessageAssistant"] { border-left: 4px solid #8b0000 !important; }

    /* 5. Clean File Uploader: Full Click-Zone & No Outline */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 49, 49, 0.2) !important;
        border-radius: 15px !important;
        padding: 10px !important;
        transition: all 0.3s ease !important;
    }

    /* Target the native dashed border to hide it completely */
    [data-testid="stFileUploader"] section {
        border: none !important;
        padding: 20px !important;
    }

    [data-testid="stFileUploader"]:hover {
        background: rgba(255, 49, 49, 0.05) !important;
        border: 1px solid #ff3131 !important;
        box-shadow: 0 0 20px rgba(255, 49, 49, 0.2);
        cursor: pointer;
    }

    /* Style the internal Upload button to match the theme */
    [data-testid="stFileUploader"] button {
        background-color: #ff3131 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
    }

    /* 6. Action Buttons & Tabs */
    button {
        background-color: rgba(255, 49, 49, 0.05) !important;
        color: #ff3131 !important;
        border: 1px solid #ff3131 !important;
        border-radius: 10px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
    }

    button:hover {
        background-color: #ff3131 !important;
        color: #000000 !important;
        box-shadow: 0px 0px 18px rgba(255, 49, 49, 0.6);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        color: #ff3131 !important;
        border-bottom-color: #ff3131 !important;
    }

    /* 7. Plot Styling: Clean & No Fullscreen Button */
    .stPyplot {
        background-color: #ffffff !important; 
        border: 2px solid rgba(255, 49, 49, 0.2) !important;
        border-radius: 15px !important;
        padding: 12px !important;
        box-sizing: border-box !important;
        margin-top: 10px !important;
    }
    
    .stPyplot img, .stPyplot canvas {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 8px !important; 
        display: block !important;
        margin: 0 auto !important; 
    }

    /* Hide the fullscreen/expand button on plots */
    button[aria-label="View fullscreen"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title('ERU Agentic Data Analyst')
st.divider()

# Instructions
with st.expander('💡 How to use ERU Agentic Data Analyst', expanded = True):
    col1, col2, col3 = st.columns(3)
    col1.markdown('**1. Upload**\nDrop a CSV file into the uploader below.')
    col2.markdown('**2. Ask**\nType a question regarding the data')
    col3.markdown('**3. Watch**\nClick the "Thinking" box to see the agent write code, analyze, and submit insights live!')
    
st.markdown('---')

# Data Upload
uploaded_file = st.file_uploader('Upload your dataset (CSV)', type = 'csv')

# LLM Instructions
custom_prefix = """
You are a highly technical Data Analyst sub-module. Your only goal is to process the 
dataframe 'df' and provide insights or visualizations.

### STRICT OPERATIONAL RULES:
1. NO THINK TAGS: Do NOT output <think> or </think> tags. Do not explain your inner reasoning.
2. FORMAT: You MUST only use the 'Thought:', 'Action:', 'Action Input:', 'Observation:', 'Final Answer:' format.
3. DATA SOURCE: The dataset is ALREADY loaded into a variable named 'df'. NEVER use 'pd.read_csv'.
4. PLOTTING: You MUST use 'plt.figure(figsize=(10, 6))' for every plot. 
5. STREAMLIT OUTPUT: You MUST end every plotting code block with 'st.pyplot(plt.gcf(), use_container_width=False)'.
6. ATOMIC PLOTTING: You MUST perform all plotting steps (figure creation, plotting, and st.pyplot) in ONE single Action Input block to avoid empty canvases. Always use: st.pyplot(plt.gcf(), use_container_width=False)

### MANDATORY IMPORTS:
Every 'Action Input' containing code MUST start with:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
plt.clf() # Clear previous figures to prevent white plots

### PAIRPLOT PROTOCOL:
If the user asks for a pairplot or 'all relationships':
1. Select only the 4-5 most relevant numerical columns (e.g., 'Survived', 'Pclass', 'Age', 'Fare').
2. Use 'g = sns.pairplot(df[selected_cols])'.
3. Use 'st.pyplot(g.fig, use_container_width=False)' specifically to ensure the entire grid is captured.
4. Set 'plt.clf()' before starting to clear the buffer.
"""

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Metadata & Metrics
    with st.sidebar:
            st.header("📊 Dataset Intelligence")
            st.success("File Linked Successfully")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Rows", df.shape[0])
            col_m2.metric("Features", df.shape[1])
            
            missing = df.isnull().any(axis=1).sum()
            if missing > 0:
                st.warning(f"⚠️ {missing} missing values.")
            else:
                st.info("✅ No missing values.")
            
            st.write("---")
            st.caption("Data Types Summary")
            st.write(df.dtypes.value_counts())
    
    # Tabs
    tab_preview, tab_chat = st.tabs(['🔎 View Dataset', '💬 Ask The Analyst'])
    
    with tab_preview:
        st.write(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')
        st.dataframe(df, use_container_width = True)
        
    with tab_chat:
        # Session State for Tab History
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Displaying Message History
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                
        # Suggested prompts as clickable buttons
        active_prompt = None
        
        st.write("---")
        st.caption("Common Analysis Requests")
        cols = st.columns(4)
        suggestions = [
            "🔍 Summarize the data", 
            "📈 Show correlations", 
            "📉 Outlier analysis", 
            "🎯 Relationship of Target"
        ]

        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, key = f'btn_{i}', use_container_width = True):
                active_prompt = suggestion
        
        # Chat Input
        chat_input = st.chat_input("Ask me about trends, outliers, or summary statistics. Let's get to work!")
        if chat_input:
            active_prompt = chat_input
        
        if active_prompt:
            st.session_state.messages.append({'role': 'user', 'content': active_prompt})
            with st.chat_message('user'):
                st.markdown(active_prompt)
            
            with st.chat_message('assistant'):
                # 1. Manually create the status box with the correct emoji
                # This prevents the callback from creating its own ":thinking_face:" box
                with st.status("🧠 Thinking...", expanded = False) as status:
                    
                    # 2. Show Lottie inside this status
                    if lottie_thinking:
                        st_lottie(lottie_thinking, height=100, key=f"lottie_{len(st.session_state.messages)}")
                    
                    # 3. Point the callback to this specific status object
                    st_callback = StreamlitCallbackHandler(
                        st.container(), 
                        thought_labeler=AgentLabeler()
                    )
                try:
                    llm = ChatGroq(
                        model_name = 'qwen/qwen3-32b',
                        temperature = 0,
                        groq_api_key = api_key,
                        max_tokens = 1024
                    )
                    
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose = True,
                        allow_dangerous_code = True,
                        handle_parsing_errors = True,
                        prefix = custom_prefix,
                        max_iterations=5,          
                        max_execution_time=30.0,   
                        include_df_in_prompt=True  
                    )
                    
                    response = agent.run(active_prompt, callbacks=[st_callback])
                    
                    status.update(label = '✅ Analysis Complete!', state = 'complete', expanded = False)
                
                except Exception as e:
                    st.error(f'Something went wrong:{e}')
                    response = None
                
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({'role': 'assistant', 'content': response})
                    
else:
    st.info("Please upload a CSV file and let's get to work!")
    
# Footer
st.markdown('---')
st.caption('Powered by Groq Qwen 3 | Developed By Ahmed Walid 🩶')