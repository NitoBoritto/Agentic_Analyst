import streamlit as st
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
    def get_tool_label(self, tool, tool_input) -> str:
        return "Processing Data..."

# Engine
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
st.set_page_config(page_title = 'Agent',
                   page_icon = '📊',
                   layout = 'wide')

st.markdown("""
            <style>
            .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
            .main {background-color: #0e1117 }
            </style>
            """, unsafe_allow_html = True)

# Header
st.title('📊 ERU Agentic Data Analyst')
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
4. PLOTTING: You MUST use 'plt.figure(figsize=(12, 8))' for every plot. 
5. STREAMLIT OUTPUT: You MUST end every plotting code block with 'st.pyplot(plt.gcf())'. 
6. ATOMIC PLOTTING: You MUST perform all plotting steps (figure creation, plotting, and st.pyplot) in ONE single Action Input block to avoid empty canvases.

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
3. Use 'st.pyplot(g.fig)' specifically to ensure the entire grid is captured.
4. Set 'plt.clf()' before starting to clear the buffer.
"""

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
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
        
        # Chat Input
        if prompt := st.chat_input('Ask me about trends, outliers, or summary statistics. I am here to assist you 👏'):
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)
                
            with st.chat_message('assistant'):
                # "Thinking" Loop Container
                st_callback = StreamlitCallbackHandler(st.container(),
                                                      thought_labeler=AgentLabeler())
                
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
                    
                    response = agent.run(prompt, callbacks = [st_callback])
                    
                    st.markdown(response)
                    st.session_state.messages.append({'role': 'assistant', 'content': response})
                
                except Exception as e:
                    st.error(f'Something went wrong:{e}')
                    
else:
    st.info("Please upload a CSV file and let's get to work!")
    
# Footer
st.markdown('---')
st.caption('Powered by Groq Qwen 3 | Developed By Ahmed Walid 🩶')