st.markdown("""
<style>
    /* Background & Container Adjustments */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        color: #F8FAFC;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1250px;
    }

    /* Vibrant Gradient Title */
    .app-title {
        font-weight: 900;
        font-size: 2.3rem;
        line-height: 1.2;
        background: linear-gradient(90deg, #A855F7 0%, #3B82F6 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 0 30px rgba(168, 85, 247, 0.3);
    }
    .app-subtitle {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: #8B5CF6;
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -10px rgba(139, 92, 246, 0.5);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #34D399 0%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #CBD5E1;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    /* Lively Passage Result Cards */
    .passage-card {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #8B5CF6;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        transition: all 0.25s ease-in-out;
    }
    .passage-card:hover {
        transform: translateX(4px);
        background: rgba(30, 41, 59, 0.85);
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.4);
    }
    .passage-card-hybrid { 
        border-left-color: #10B981; 
    }
    
    .score-badge {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 12px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-left: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Custom Input Styling */
    .stTextInput input, .stTextArea textarea, .stSelectbox > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #A855F7 !important;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.3) !important;
    }

    /* Glowing Primary Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366F1 0%, #A855F7 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6) !important;
    }

    /* Custom Tabs Styling */
    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    button[aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)
