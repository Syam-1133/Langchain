import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


# Set page configuration
st.set_page_config(
    page_title="AI Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for seamless professional styling
st.markdown("""
<style>
    /* Main styling - remove all white backgrounds */
    .main-header {
        font-size: 3.5rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Remove white backgrounds from all containers */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        background: transparent;
    }
    
    /* Feature cards - transparent with border only */
    .feature-card {
        background: transparent;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #e3f2fd;
        transition: all 0.3s ease;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        border-color: #2563eb;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.15);
    }
    
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2563eb;
        margin: 5px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #6b7280;
        line-height: 1.4;
    }
    
    /* Chat area - transparent background */
    .chat-area {
        max-height: 500px;
        overflow-y: auto;
        padding: 20px;
        background: transparent;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    /* Message styling - keep bubbles but remove white backgrounds */
    .message-wrapper {
        display: flex;
        margin-bottom: 16px;
        align-items: flex-start;
    }
    
    .user-message-wrapper {
        justify-content: flex-end;
    }
    
    .ai-message-wrapper {
        justify-content: flex-start;
    }
    
    .user-message {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 5px 18px;
        max-width: 70%;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
        font-size: 14px;
        line-height: 1.5;
    }
    
    .ai-message {
        background: rgba(248, 250, 252, 0.9);
        color: #374151;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 5px;
        max-width: 70%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid rgba(229, 231, 235, 0.5);
        font-size: 14px;
        line-height: 1.5;
        backdrop-filter: blur(5px);
    }
    
    .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        margin: 0 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 12px;
    }
    
    .user-avatar {
        background: #2563eb;
        color: white;
    }
    
    .ai-avatar {
        background: #10b981;
        color: white;
    }
    
    /* Input area - simplified without extra bars */
    .input-container {
        background: rgba(15, 23, 42, 0.9);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(100, 116, 139, 0.5);
        margin-top: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Input field styling - white text on dark background */
    .stTextInput>div>div>input {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(100, 116, 139, 0.5);
        color: white !important;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 14px;
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #94a3b8 !important;
        opacity: 0.8;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.3);
        background: rgba(30, 41, 59, 0.9);
        color: white !important;
    }
    
    /* Ensure text remains white when typing */
    .stTextInput>div>div>input[type="text"] {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        height: 44px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.3);
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    }
    
    /* Sidebar styling - make transparent */
    .sidebar .sidebar-content {
        background: rgba(248, 250, 252, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(229, 231, 235, 0.5);
    }
    
    /* Remove all default white backgrounds */
    section.main.css-1v3fvcr.eczjsme3 {
        background: transparent;
    }
    
    div.block-container.css-1y4p8pa.eczjsme4 {
        background: transparent;
        padding-bottom: 0px;
    }
    
    /* Hide Streamlit default elements and remove white bars */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove the bottom white bar completely */
    .css-1dp5vir {
        display: none;
    }
    
    /* Make the main area transparent */
    .css-1v3fvcr {
        background: transparent;
    }
    
    /* Remove padding that creates white space */
    .css-1v3fvcr .css-1v3fvcr {
        padding: 0px;
    }
    
    /* Welcome message styling */
    .welcome-message {
        text-align: center;
        padding: 40px 20px;
        color: #6b7280;
        background: transparent;
    }
    
    /* Section headers */
    h2, h3 {
        color: #2563eb;
        margin-bottom: 1rem;
    }
    
    /* Remove extra form borders and backgrounds */
    form {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stForm {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove any extra borders from form containers */
    div[data-testid="stForm"] {
        border: none !important;
        background: transparent !important;
    }
    
    /* Ensure no extra spacing */
    .row-widget.stButton {
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional AI assistant. Provide clear, detailed, and helpful responses. Be conversational but maintain professionalism."),
        ("user", "Question: {question}")
    ])
    llm = OllamaLLM(model="gemma:2b")
    output_parser = StrOutputParser()
    st.session_state.chain = prompt | llm | output_parser

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px; padding: 20px 0; border-bottom: 1px solid rgba(229, 231, 235, 0.5);'>
        <h1 style='color: #2563eb; font-size: 1.5rem; font-weight: 700; margin: 0;'>AI Assistant Pro</h1>
        <p style='color: #6b7280; font-size: 0.9rem; margin: 5px 0 0 0;'>Enterprise AI Solutions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Configuration")
    
    model_option = st.selectbox(
        "AI Model",
        ["gemma:2b", "gemma:7b", "llama2", "mistral"],
        index=0
    )
    
    temperature = st.slider(
        "Response Creativity",
        min_value=0.1,
        max_value=1.0,
        value=0.7
    )
    
    st.markdown("---")
    st.markdown("### 📊 Usage Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Chats", len(st.session_state.messages)//2)
    with col2:
        st.metric("Today", "8")
    
    st.markdown("---")
    
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Header section
    st.markdown('<div class="main-header">Syam AI Assistant </div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Smart AI assistance built for workplace productivity</div>', unsafe_allow_html=True)
    
   
   
    
    st.markdown("---")
    
    # Chat area
    st.markdown("### 💬 Live Conversation")
    
    # Chat container with transparent background
    with st.container():
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        
        if not st.session_state.messages:
            # Welcome message
            st.markdown("""
            <div class="welcome-message">
                <div style='font-size: 3rem; margin-bottom: 10px;'>👋</div>
                <h3 style='color: #374151; margin-bottom: 10px;'>Welcome to Syam AI Assistant</h3>
                <p>Start a conversation by typing your question below. I'm here to help with any task!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display conversation
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="message-wrapper user-message-wrapper">
                        <div class="user-message">
                            {message["content"]}
                        </div>
                        <div class="avatar user-avatar">YOU</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="message-wrapper ai-message-wrapper">
                        <div class="avatar ai-avatar">AI</div>
                        <div class="ai-message">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
  
    
    # Use columns directly without extra form containers
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message:",
            placeholder="Type your message here...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.button(
            "Send →",
            use_container_width=True,
            type="primary"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Handle user input
if submit_button and user_input.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show typing indicator
    with st.spinner("Syam AI is thinking...🤔 "):
        try:
            # Get AI response
            ai_response = st.session_state.chain.invoke({"question": user_input})
            
            # Add AI response
            st.session_state.messages.append({"role": "ai", "content": ai_response})
            
            # Rerun to update display
            st.rerun()
            
        except Exception as e:
            error_msg = "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
            st.session_state.messages.append({"role": "ai", "content": error_msg})
            st.rerun()

# Minimal footer
st.markdown("""
<div style='margin-top: 50px; padding: 20px; text-align: center; color: #9ca3af;'>
    <p style='font-size: 0.9rem; margin: 0;'>© 2024 AI Assistant Pro • Built with Streamlit & Ollama</p>
</div>
""", unsafe_allow_html=True)