import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# Initialize session state variables
if 'completed_steps' not in st.session_state:
    st.session_state.completed_steps = set()
if 'current_step' not in st.session_state:
    st.session_state.current_step = "Upload Data"

# Configure page
st.set_page_config(
    page_title="MLmadeEasy - Cyberpunk ML Platform",
    page_icon="https://img.icons8.com/nolan/64/cyborg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Cyberpunk dark theme styling
<style>
    body, .stApp {
        background-image: url('https://i.pinimg.com/originals/a1/52/43/a1524316b48635306bd6339d78c465ae.gif');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #ff00ff;
    }
    .main {
        background-color: rgba(10, 15, 30, 0.85);
        border-radius: 10px;
        border: 1px solid #00f2ff;
        box-shadow: 0 0 15px #00f2ff, 0 0 15px rgba(255, 0, 255, 0.5);
        color: #00f2ff;
    }
    h1, h2, h3 {
        color: #ff00ff;
        font-weight: 700;
        text-shadow: 0 0 10px #ff00ff;
    }
    .stButton>button {
        background-color: #1a0038;
        color: #00f2ff;
        border: 2px solid #00f2ff;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        text-transform: uppercase;
        box-shadow: 0 0 10px #00f2ff;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #00f2ff;
        color: #1a0038;
        box-shadow: 0 0 20px #00f2ff;
    }
    .stDownloadButton>button {
        background-color: #420042;
        color: #00ff9f;
        border: 2px solid #00ff9f;
        box-shadow: 0 0 10px #00ff9f;
        font-weight: bold;
    }
    .stDownloadButton>button:hover {
        background-color: #00ff9f;
        color: #420042;
        box-shadow: 0 0 20px #00ff9f;
    }
    .sidebar .sidebar-content {
        background-color: rgba(30, 10, 50, 0.9);
        border-right: 1px solid #ff00ff;
        color: #00f2ff;
    }
    .css-hxt7ib {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .data-warning {
        color: #ff375f;
        font-weight: bold;
        text-shadow: 0 0 5px #ff375f;
    }
    .feature-selector {
        background-color: rgba(25, 25, 75, 0.85);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ff00ff;
        color: #00f2ff;
        box-shadow: 0 0 10px rgba(255, 0, 255, 0.5);
    }
    .st-expanderContent {
        background-color: rgba(25, 25, 75, 0.95);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00f2ff;
    }
    .stDataFrame {
        background-color: rgba(15, 10, 30, 0.7);
        border: 1px solid #00f2ff;
        border-radius: 5px;
    }
    /* Cyberpunk scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1a0038;
    }
    ::-webkit-scrollbar-thumb {
        background: #00f2ff;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #ff00ff;
    }
    /* Metrics styling */
    div.css-1xarl3l {
        background-color: rgba(20, 20, 40, 0.7);
        border: 1px solid #00f2ff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
    }
    /* Input fields */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stMultiselect>div>div>div {
        background-color: rgba(30, 10, 50, 0.7);
        color: #00f2ff;
        border: 1px solid #ff00ff;
    }
    .stSlider>div>div>div {
        color: #00f2ff;
    }
    .stSlider>div>div>div>div {
        background-color: #ff00ff;
    }
    /* Plotly charts */
    .js-plotly-plot {
        background-color: rgba(10, 15, 30, 0.7);
        border-radius: 10px;
        border: 1px solid #00f2ff;
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Function to display the cyberpunk journey flow chart
def show_journey_steps():
    steps = [
        {"name": "Upload Data", "icon": "📊", "description": "Start by uploading your financial data or fetching from Yahoo Finance"},
        {"name": "Preprocessing", "icon": "🔧", "description": "Clean and prepare your data for analysis"},
        {"name": "Feature Engineering", "icon": "🛠️", "description": "Select the most important features for your model"},
        {"name": "Train/Test Split", "icon": "📈", "description": "Divide data into training and testing sets"},
        {"name": "Model Training", "icon": "🤖", "description": "Choose and train your ML model"},
        {"name": "Model Evaluation", "icon": "📊", "description": "Analyze model performance and metrics"},
        {"name": "Predictions", "icon": "🔮", "description": "Make predictions with your trained model"},
        {"name": "Export Results", "icon": "📤", "description": "Save and export your predictions"}
    ]
    
    st.markdown('<div class="ml-journey-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="journey-title pixel-flicker">Your ML Journey</h2>', unsafe_allow_html=True)
    
    # Display journey steps
    st.markdown('<div class="journey-steps">', unsafe_allow_html=True)
    
    for i, step in enumerate(steps[:4]):  # Display first 4 steps in the first row
        step_class = ""
        if step["name"] == st.session_state.current_step:
            step_class = "step-active"
        elif step["name"] in st.session_state.completed_steps:
            step_class = "step-completed"
        
        st.markdown(f"""
        <div class="journey-step {step_class}">
            <div class="step-number">{i+1}</div>
            <div class="step-icon">{step["icon"]}</div>
            <div class="step-name">{step["name"]}</div>
            <div class="step-description">{step["description"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row for remaining steps
    st.markdown('<div class="journey-steps" style="margin-top: 15px;">', unsafe_allow_html=True)
    
    for i, step in enumerate(steps[4:]):  # Display last 4 steps in the second row
        step_class = ""
        if step["name"] == st.session_state.current_step:
            step_class = "step-active"
        elif step["name"] in st.session_state.completed_steps:
            step_class = "step-completed"
        
        st.markdown(f"""
        <div class="journey-step {step_class}">
            <div class="step-number">{i+5}</div>
            <div class="step-icon">{step["icon"]}</div>
            <div class="step-name">{step["name"]}</div>
            <div class="step-description">{step["description"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to show the progress checklist sidebar
def show_progress_checklist():
    st.sidebar.markdown("""
    <div class="progress-checklist pixel-flicker">
        <h3>ML PIPELINE PROGRESS</h3>
    """, unsafe_allow_html=True)
    
    steps = [
        ("Upload Data", "📊"),
        ("Preprocessing", "🔧"),
        ("Feature Engineering", "🛠️"),
        ("Train/Test Split", "📈"),
        ("Model Training", "🤖"),
        ("Model Evaluation", "📊"),
        ("Predictions", "🔮"),
        ("Export Results", "📤")
    ]
    
    for step, emoji in steps:
        status = "completed" if step in st.session_state.completed_steps else "pending"
        st.sidebar.markdown(f"""
        <li class="{status}">
            {emoji} {step}
        </li>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Cyberpunk header with animated GIF
def display_header():
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <img src="https://i.gifer.com/YAhr.gif" width="150px" style="margin-bottom: 10px;">
        <h1 style="font-size: 3rem; letter-spacing: 3px;" class="glitch">MLmadeEasy</h1>
        <p style="color: #00f2ff; font-size: 1.2rem;">[ NEURAL-NET PREDICTION ENGINE v2.0.77 ]</p>
        <div style="background: linear-gradient(90deg, rgba(255,0,255,0) 0%, rgba(255,0,255,1) 50%, rgba(255,0,255,0) 100%); height: 2px; margin: 20px 0;"></div>
        <p style="color: #00f2ff; font-size: 1rem;">Your Journey to Machine Learning Starts Here!</p>
    </div>
    """, unsafe_allow_html=True)

# Cyberpunk themed sidebar header
def custom_sidebar_header():
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://i.pinimg.com/originals/1c/62/fe/1c62fe38d065c0a2929357f01c1e118e.gif" width="100%">
        <h3 style="color: #ff00ff; letter-spacing: 2px; margin-top: 10px;" class="pixel-flicker">NEURAL INTERFACE</h3>
    </div>
    """, unsafe_allow_html=True)

# Cyberpunk loading animation
def cyberpunk_loading_animation():
    return """
    <div style="text-align: center; margin: 20px 0;">
        <img src="https://i.gifer.com/5IPd.gif" width="150px">
        <p style="color: #00f2ff; margin-top: 10px; font-family: monospace; font-size: 1.2rem;">
            INITIALIZING NEURAL NETWORKS...
        </p>
    </div>
    """

# Feature importance visualization with cyberpunk styling
def cyberpunk_feature_importance(importance_df):
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance', 
        color_continuous_scale=['#120052', '#4b0082', '#9400d3', '#ff00ff']
    )
    fig.update_layout(
        plot_bgcolor='rgba(10, 10, 30, 0.7)',
        paper_bgcolor='rgba(10, 10, 30, 0)',
        font=dict(color='#00f2ff'),
        title_font=dict(color='#ff00ff'),
        xaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff'),
            gridcolor='rgba(0, 242, 255, 0.2)'
        ),
        yaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff'),
            gridcolor='rgba(0, 242, 255, 0.2)'
        )
    )
    return fig

# Cyberpunk scatter plot with neon styling
def cyberpunk_scatter_plot(df, x, y):
    # Create scatter plot without trendline first
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        height=400,
        color_discrete_sequence=['#ff00ff']
    )
    
    # Update marker styling
    fig.update_traces(
        marker=dict(
            size=8,
            symbol='circle',
            line=dict(width=1, color='#00f2ff'),
            opacity=0.8
        )
    )
    
    # Try to add trendline manually using numpy's polyfit
    try:
        import numpy as np
        from scipy import stats
        
        # Get x and y values
        x_values = df[x].values
        y_values = df[y].values
        
        # Remove NaN values
        mask = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[mask]
        y_values = y_values[mask]
        
        if len(x_values) > 1:
            # Calculate trendline
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            x_range = np.linspace(min(x_values), max(x_values), 100)
            y_range = slope * x_range + intercept
            
            # Add trendline to plot
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='#00f2ff', width=3),
                    name='Trend'
                )
            )
    except:
        # If trendline calculation fails, just continue without it
        pass
        
    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(10, 10, 30, 0.7)',
        paper_bgcolor='rgba(10, 10, 30, 0)',
        font=dict(color='#00f2ff'),
        title_font=dict(color='#ff00ff'),
        xaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff'),
            gridcolor='rgba(0, 242, 255, 0.2)',
            zerolinecolor='#ff00ff'
        ),
        yaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff'),
            gridcolor='rgba(0, 242, 255, 0.2)',
            zerolinecolor='#ff00ff'
        )
    )
    
    return fig

# Cyberpunk correlation matrix
def cyberpunk_correlation_matrix(corr_matrix):
    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale=['#120052', '#4b0082', '#9400d3', '#ff00ff'],
        aspect="auto"
    )
    fig.update_layout(
        plot_bgcolor='rgba(10, 10, 30, 0.7)',
        paper_bgcolor='rgba(10, 10, 30, 0)',
        font=dict(color='#00f2ff'),
        title_font=dict(color='#ff00ff'),
        xaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff')
        ),
        yaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff')
        )
    )
    return fig

# Cyberpunk actual vs predicted plot
def cyberpunk_actual_vs_predicted(results):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results.index, 
            y=results['Actual'], 
            name='Actual', 
            mode='markers', 
            marker=dict(
                color='#ff00ff',
                size=8,
                line=dict(width=1, color='#ff00ff'),
                symbol='circle',
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index, 
            y=results['Predicted'], 
            name='Predicted', 
            mode='markers', 
            marker=dict(
                color='#00f2ff',
                size=8,
                line=dict(width=1, color='#00f2ff'),
                symbol='diamond',
            )
        )
    )
    fig.update_layout(
        xaxis_title="Sample Index",
        yaxis_title="Value",
        height=500,
        legend=dict(font=dict(color='#00f2ff')),
        plot_bgcolor='rgba(10, 10, 30, 0.7)',
        paper_bgcolor='rgba(10, 10, 30, 0)',
        font=dict(color='#00f2ff'),
        xaxis=dict(
            gridcolor='rgba(0, 242, 255, 0.2)',
            zerolinecolor='#ff00ff'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 242, 255, 0.2)',
            zerolinecolor='#ff00ff'
        )
    )
    return fig

# Success message with cyberpunk styling
def cyber_success(message):
    return st.markdown(f"""
    <div style="background-color: rgba(0, 50, 20, 0.7); border: 1px solid #00ff9f; border-radius: 5px; padding: 10px; margin: 10px 0; box-shadow: 0 0 10px #00ff9f;">
        <p style="color: #00ff9f; margin: 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">✅</span> {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Error message with cyberpunk styling
def cyber_error(message):
    return st.markdown(f"""
    <div style="background-color: rgba(50, 0, 20, 0.7); border: 1px solid #ff375f; border-radius: 5px; padding: 10px; margin: 10px 0; box-shadow: 0 0 10px #ff375f;">
        <p style="color: #ff375f; margin: 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">⚠️</span> {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Function to mark step as completed
def complete_step(step_name):
    st.session_state.completed_steps.add(step_name)
    next_steps = {
        "Upload Data": "Preprocessing",
        "Preprocessing": "Feature Engineering",
        "Feature Engineering": "Train/Test Split",
        "Train/Test Split": "Model Training",
        "Model Training": "Model Evaluation",
        "Model Evaluation": "Predictions",
        "Predictions": "Export Results"
    }
    if step_name in next_steps:
        st.session_state.current_step = next_steps[step_name]

# Function to render a quick start guide
def display_quick_start_guide():
    st.markdown("""
    <div class="ml-journey-container">
        <h2 class="journey-title pixel-flicker">Quick Start Guide</h2>
        <div class="journey-steps">
            <div class="journey-step">
                <div class="step-number">1</div>
                <div class="step-icon">📊</div>
                <div class="step-name">Upload Data</div>
                <div class="step-description">Start by uploading your financial data or fetching from Yahoo Finance</div>
            </div>
            <div class="journey-step">
                <div class="step-number">2</div>
                <div class="step-icon">🔧</div>
                <div class="step-name">Preprocess</div>
                <div class="step-description">Clean and prepare your data for analysis</div>
            </div>
            <div class="journey-step">
                <div class="step-number">3</div>
                <div class="step-icon">🤖</div>
                <div class="step-name">Train Model</div>
                <div class="step-description">Choose and train your ML model</div>
            </div>
            <div class="journey-step">
                <div class="step-number">4</div>
                <div class="step-icon">📈</div>
                <div class="step-name">Evaluate</div>
                <div class="step-description">Analyze model performance and make predictions</div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 30px;">
            <button style="background-color: #1a0038; color: #00f2ff; border: 2px solid #00f2ff; padding: 10px 20px; box-shadow: 0 0 10px #00f2ff; font-family: 'Press Start 2P', cursive; font-size: 0.8rem; cursor: pointer;">
                START YOUR ML JOURNEY
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Function
def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    # Display cyberpunk header
    display_header()
    
    # Session state initialization
    session_defaults = {
        'data': None, 'model': None, 'features': [], 'target': None,
        'steps': {'loaded': False, 'processed': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Show ML Journey Flow Chart
    show_journey_steps()

    # Sidebar Configuration
    with st.sidebar:
        custom_sidebar_header()
        
        # Show ML Progress Checklist
        show_progress_checklist()
        
        st.markdown("""
        <div style="border: 1px solid #ff00ff; padding: 15px; margin-bottom: 20px; background-color: rgba(25, 10, 40, 0.7);">
            <h4 style="color: #ff00ff; text-align: center; text-shadow: 0 0 5px #ff00ff; letter-spacing: 2px;">DATA INPUT</h4>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("UPLOAD DATASET:", type=["csv", "xlsx"])
        
        if uploaded_file is None:
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://i.gifer.com/embedded/download/7SVt.gif" width="100%">
                <p style="color: #00f2ff; margin-top: 10px; font-size: 0.8rem;">AWAITING DATA UPLOAD...</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="border: 1px solid #00f2ff; padding: 15px; background-color: rgba(25, 10, 40, 0.7);">
            <h4 style="color: #00f2ff; text-align: center; text-shadow: 0 0 5px #00f2ff; letter-spacing: 2px;">NEURAL NET PARAMETERS</h4>
        """, unsafe_allow_html=True)
        
        model_type = st.selectbox("SELECT ALGORITHM:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("TEST DATA RATIO:", 0.1, 0.5, 0.2)
        
        if st.button("RESET SYSTEM"):
            st.session_state.clear()
            st.experimental_rerun()
        
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <img src="https://i.gifer.com/7D7o.gif" width="100%">
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 1: Data Upload
    step_section = "Upload Data"
    is_current = step_section == st.session_state.current_step
    is_completed = step_section in st.session_state.completed_steps
    
    section_border = "#ff00ff" if is_current else "#00f2ff" if is_completed else "#ff00ff"
    section_glow = "0 0 15px #ff00ff" if is_current else "0 0 15px #00f2ff" if is_completed else "none"
    
    st.markdown(f"""
    <div style="background-color: rgba(20, 10, 40, 0.7); border: 2px solid {section_border}; padding: 15px; margin-bottom: 20px; box-shadow: {section_glow};">
            <h2 style="display: flex; align-items: center; gap: 10px;" class="pixel-flicker">
                <img src="https://i.gifer.com/7H27.gif" width="30px"> 
                STEP 4: TRAIN/TEST SPLIT
            </h2>
        """, unsafe_allow_html=True)
        
        if is_current and not is_completed:
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://i.gifer.com/embedded/download/3udK.gif" width="150px">
                <p style="color: #00f2ff; margin-top: 15px; font-family: 'Share Tech Mono', monospace; font-size: 1.2rem;">
                    SPLITTING DATA INTO TRAINING AND TESTING SETS...
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate data splitting with progress bar
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            cyber_success("DATA SPLITTING COMPLETE")
            
            # Mark step as completed
            if step_section == st.session_state.current_step:
                complete_step(step_section)
        
        df = st.session_state.data
        
        if df is not None and len(st.session_state.features) > 0:
            features = st.session_state.features
            target = st.session_state.target
            test_ratio = st.session_state.get('test_size', 0.2)
            
            # Display training and testing data visualization
            st.markdown("""
            <div style="display: flex; align-items: center; margin-top: 20px;">
                <img src="https://i.gifer.com/embedded/download/Ymn.gif" width="40px" style="margin-right: 10px;">
                <h3 style="color: #00f2ff; margin: 0;">TRAIN/TEST SPLIT</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a visual representation of the train/test split
            col1, col2 = st.columns(2)
            
            with col1:
                train_size = 1 - test_ratio
                train_count = int(len(df) * train_size)
                test_count = int(len(df) * test_ratio)
                
                st.markdown(f"""
                <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #00f2ff; padding: 15px; 
                            text-align: center; box-shadow: 0 0 10px rgba(0, 242, 255, 0.5);">
                    <div style="color: #ff00ff; font-family: 'Press Start 2P', cursive; font-size: 1rem; margin-bottom: 10px;">
                        TRAINING SET
                    </div>
                    <div style="color: #00f2ff; font-size: 2rem; font-weight: bold;">
                        {train_count}
                    </div>
                    <div style="color: #cccccc; font-size: 0.8rem;">
                        {train_size:.0%} OF DATA
                    </div>
                    <div style="margin-top: 10px;">
                        <img src="https://i.gifer.com/embedded/download/Pijt.gif" width="80px">
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #ff00ff; padding: 15px; 
                            text-align: center; box-shadow: 0 0 10px rgba(255, 0, 255, 0.5);">
                    <div style="color: #00f2ff; font-family: 'Press Start 2P', cursive; font-size: 1rem; margin-bottom: 10px;">
                        TESTING SET
                    </div>
                    <div style="color: #ff00ff; font-size: 2rem; font-weight: bold;">
                        {test_count}
                    </div>
                    <div style="color: #cccccc; font-size: 0.8rem;">
                        {test_ratio:.0%} OF DATA
                    </div>
                    <div style="margin-top: 10px;">
                        <img src="https://i.gifer.com/embedded/download/Qz4D.gif" width="80px">
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a "Next Step" button
            next_step_button_text = "PROCEED TO MODEL TRAINING"
            if st.button(next_step_button_text, key="next_to_model_training"):
                complete_step(step_section)
                st.experimental_rerun()
        else:
            st.warning("Please complete the previous steps first.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 5: Model Training
    step_section = "Model Training"
    is_current = step_section == st.session_state.current_step
    is_completed = step_section in st.session_state.completed_steps
    
    if is_current or is_completed:
        section_border = "#ff00ff" if is_current else "#00f2ff" if is_completed else "#ff00ff"
        section_glow = "0 0 15px #ff00ff" if is_current else "0 0 15px #00f2ff" if is_completed else "none"
        
        st.markdown(f"""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 2px solid {section_border}; padding: 15px; margin-bottom: 20px; box-shadow: {section_glow};">
            <h2 style="display: flex; align-items: center; gap: 10px;" class="pixel-flicker">
                <img src="https://i.gifer.com/3Q7h.gif" width="30px"> 
                STEP 5: MODEL TRAINING
            </h2>
        """, unsafe_allow_html=True)
        
        if is_current and not is_completed:
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://i.gifer.com/QQrP.gif" width="200px">
                <p style="color: #ff00ff; margin-top: 15px; font-family: 'Share Tech Mono', monospace; font-size: 1.3rem;">
                    TRAINING NEURAL NETWORKS...
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            df = st.session_state.data
            
            if df is not None and len(st.session_state.features) > 0:
                features = st.session_state.features
                target = st.session_state.target
                test_size = st.session_state.get('test_size', 0.2)
                model_type = st.session_state.get('model_type', "Linear Regression")
                
                X = df[features]
                y = df[target]
                
                # Create train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Simulate model training with progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)
                
                # Fit model
                model.fit(X_train_scaled, y_train)
                
                # Store model and predictions
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.scaler = scaler
                
                # Generate predictions
                y_pred = model.predict(X_test_scaled)
                st.session_state.y_pred = y_pred
                
                cyber_success("MODEL TRAINED SUCCESSFULLY!")
                st.balloons()
                
                # Mark step as completed
                if step_section == st.session_state.current_step:
                    complete_step(step_section)
        
        # Display model information
        if is_completed or (is_current and 'model' in st.session_state):
            model_type = st.session_state.get('model_type', "Linear Regression")
            
            st.markdown("""
            <div style="display: flex; align-items: center; margin-top: 20px;">
                <img src="https://i.gifer.com/embedded/download/5Z3.gif" width="40px" style="margin-right: 10px;">
                <h3 style="color: #00f2ff; margin: 0;">MODEL INFORMATION</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #ff00ff; padding: 15px; 
                            text-align: center; box-shadow: 0 0 10px rgba(255, 0, 255, 0.5);">
                    <div style="color: #00f2ff; font-family: 'Press Start 2P', cursive; font-size: 1rem; margin-bottom: 10px;">
                        ALGORITHM
                    </div>
                    <div style="color: #ff00ff; font-size: 1.5rem; font-weight: bold;">
                        {model_type}
                    </div>
                    <div style="margin-top: 10px;">
                        <img src="https://i.gifer.com/embedded/download/g0R5.gif" width="80px">
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                features = st.session_state.features
                
                st.markdown(f"""
                <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #00f2ff; padding: 15px; 
                            text-align: center; box-shadow: 0 0 10px rgba(0, 242, 255, 0.5);">
                    <div style="color: #ff00ff; font-family: 'Press Start 2P', cursive; font-size: 1rem; margin-bottom: 10px;">
                        FEATURES
                    </div>
                    <div style="color: #00f2ff; font-size: 1.5rem; font-weight: bold;">
                        {len(features)}
                    </div>
                    <div style="color: #cccccc; font-size: 0.8rem;">
                        INPUT VARIABLES
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a "Next Step" button
            next_step_button_text = "PROCEED TO MODEL EVALUATION"
            if st.button(next_step_button_text, key="next_to_model_evaluation"):
                complete_step(step_section)
                st.experimental_rerun()
        else:
            st.warning("Please complete the previous steps first.")
            
        st.markdown("</div>", unsafe_allow_html=True)section_glow};">
        <h2 style="display: flex; align-items: center; gap: 10px;" class="pixel-flicker">
            <img src="https://i.gifer.com/7JcU.gif" width="30px"> 
            STEP 1: DATA UPLOAD & SELECTION
        </h2>
    """, unsafe_allow_html=True)
    
    if uploaded_file:
        try:
            with st.markdown(cyberpunk_loading_animation(), unsafe_allow_html=True):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) < 2:
                    cyber_error("DATASET NEEDS AT LEAST 2 NUMERIC COLUMNS FOR ANALYSIS")
                    return
                    
                st.session_state.data = df
                st.session_state.steps['loaded'] = True
            
            # Mark this step as completed if it's the current step
            if step_section == st.session_state.current_step:
                complete_step(step_section)
            
            cyber_success(f"SUCCESSFULLY LOADED {len(df)} RECORDS")
            
            st.markdown("""
            <div style="display: flex; align-items: center; margin-top: 20px;">
                <img src="https://i.gifer.com/embedded/download/ZKZx.gif" width="40px" style="margin-right: 10px;">
                <h3 style="color: #00f2ff; margin: 0;">DATASET PREVIEW:</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)
            
            with st.expander("🔍 SELECT FEATURES & TARGET VARIABLES"):
                st.markdown("<div class='feature-selector'>", unsafe_allow_html=True)
                
                st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <img src="https://i.gifer.com/KbIm.gif" width="100px">
                </div>
                """, unsafe_allow_html=True)
                
                all_cols = df.columns.tolist()
                target = st.selectbox("SELECT TARGET VARIABLE:", numeric_cols, index=len(numeric_cols)-1)
                default_features = [col for col in numeric_cols if col != target][:3]
                features = st.multiselect("SELECT FEATURES:", numeric_cols, default=default_features)
                
                if st.button("CONFIRM SELECTION", key="confirm_features"):
                    if len(features) < 1:
                        cyber_error("PLEASE SELECT AT LEAST ONE FEATURE")
                    elif target in features:
                        cyber_error("TARGET VARIABLE CANNOT BE A FEATURE")
                    else:
                        st.session_state.features = features
                        st.session_state.target = target
                        st.session_state.steps['processed'] = True
                        cyber_success("FEATURES AND TARGET CONFIRMED!")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display a "Next Step" button to move to preprocessing
            if st.button("PROCEED TO PREPROCESSING", key="next_to_preprocessing"):
                if step_section != st.session_state.current_step and not is_completed:
                    st.session_state.current_step = step_section
                    st.experimental_rerun()
                elif is_completed or step_section == st.session_state.current_step:
                    complete_step(step_section)
                    st.experimental_rerun()
                
        except Exception as e:
            cyber_error(f"ERROR LOADING FILE: {str(e)}")
    else:
        st.markdown("""
        <div class='feature-selector' style="text-align: center;">
            <img src="https://i.gifer.com/WMT.gif" width="200px" style="margin: 20px 0;">
            <h3 style="color: #ff00ff; margin-bottom: 15px;" class="pixel-flicker">HOW TO USE MLmadeEasy:</h3>
            <ol style="text-align: left; color: #00f2ff; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem;">
                <li>Upload any CSV or Excel file with numeric data</li>
                <li>Select target variable (what you want to predict)</li>
                <li>Choose features (variables used for prediction)</li>
                <li>Follow the ML journey steps in sequence</li>
            </ol>
            <div style="margin-top: 20px;">
                <img src="https://i.gifer.com/5VYH.gif" width="80%">
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Step 2: Preprocessing and Data Analysis
    step_section = "Preprocessing"
    is_current = step_section == st.session_state.current_step
    is_completed = step_section in st.session_state.completed_steps
    
    if st.session_state.steps['processed'] or is_current or is_completed:
        section_border = "#ff00ff" if is_current else "#00f2ff" if is_completed else "#ff00ff"
        section_glow = "0 0 15px #ff00ff" if is_current else "0 0 15px #00f2ff" if is_completed else "none"
        
        st.markdown(f"""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 2px solid {section_border}; padding: 15px; margin-bottom: 20px; box-shadow: {section_glow};">
            <h2 style="display: flex; align-items: center; gap: 10px;" class="pixel-flicker">
                <img src="https://i.gifer.com/XOsX.gif" width="30px"> 
                STEP 2: DATA PREPROCESSING
            </h2>
        """, unsafe_allow_html=True)
        
        df = st.session_state.data
        
        if df is not None:
            features = st.session_state.features
            target = st.session_state.target
            
            # Show data preprocessing animation
            if is_current and not is_completed:
                st.markdown("""
                <div style="text-align: center; margin: 20px 0;">
                    <img src="https://i.gifer.com/embedded/download/3HeQ.gif" width="150px">
                    <p style="color: #00f2ff; margin-top: 15px; font-family: 'Share Tech Mono', monospace; font-size: 1.2rem;">
                        DATA PREPROCESSING IN PROGRESS...
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                cyber_success("DATA PREPROCESSING COMPLETE")
                
                # Mark step as completed
                if step_section == st.session_state.current_step:
                    complete_step(step_section)
            
            st.markdown("""
            <div style="display: flex; align-items: center; margin-top: 20px;">
                <img src="https://i.gifer.com/embedded/download/2uU.gif" width="40px" style="margin-right: 10px;">
                <h3 style="color: #00f2ff; margin: 0;">DATA ANALYSIS</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <h3 style="color: #ff00ff; text-align: center;" class="pixel-flicker">FEATURE-TARGET RELATIONSHIPS</h3>
                """, unsafe_allow_html=True)
                
                selected_feature = st.selectbox("SELECT FEATURE TO PLOT:", features)
                fig = cyberpunk_scatter_plot(df, selected_feature, target)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown("""
                <h3 style="color: #ff00ff; text-align: center;" class="pixel-flicker">CORRELATION MATRIX</h3>
                """, unsafe_allow_html=True)
                
                corr_matrix = df[features + [target]].corr()
                fig = cyberpunk_correlation_matrix(corr_matrix)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://i.gifer.com/A34R.gif" width="100px" style="margin-bottom: 10px;">
            </div>
            """, unsafe_allow_html=True)
            
            # Add a "Next Step" button
            next_step_button_text = "PROCEED TO FEATURE ENGINEERING"
            if st.button(next_step_button_text, key="next_to_feature_engineering"):
                complete_step(step_section)
                st.experimental_rerun()
        else:
            st.warning("Please upload and process data first before proceeding with preprocessing.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 3: Feature Engineering
    step_section = "Feature Engineering"
    is_current = step_section == st.session_state.current_step
    is_completed = step_section in st.session_state.completed_steps
    
    if st.session_state.steps['processed'] or is_current or is_completed:
        section_border = "#ff00ff" if is_current else "#00f2ff" if is_completed else "#ff00ff"
        section_glow = "0 0 15px #ff00ff" if is_current else "0 0 15px #00f2ff" if is_completed else "none"
        
        st.markdown(f"""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 2px solid {section_border}; padding: 15px; margin-bottom: 20px; box-shadow: {section_glow};">
            <h2 style="display: flex; align-items: center; gap: 10px;" class="pixel-flicker">
                <img src="https://i.gifer.com/7H27.gif" width="30px"> 
                STEP 3: FEATURE ENGINEERING
            </h2>
        """, unsafe_allow_html=True)
        
        if is_current and not is_completed:
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://i.gifer.com/embedded/download/Fwo.gif" width="150px">
                <p style="color: #00f2ff; margin-top: 15px; font-family: 'Share Tech Mono', monospace; font-size: 1.2rem;">
                    FEATURE OPTIMIZATION IN PROGRESS...
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show feature importance visualization if Random Forest is selected
            if 'model_type' in st.session_state and st.session_state.get('model_type') == "Random Forest":
                st.markdown("""
                <h3 style="color: #ff00ff; text-align: center;" class="pixel-flicker">FEATURE OPTIMIZATION</h3>
                """, unsafe_allow_html=True)
                
                # Simulate feature selection process with a progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                cyber_success("FEATURE OPTIMIZATION COMPLETE")
                
            # Mark step as completed
            if step_section == st.session_state.current_step:
                complete_step(step_section)
        
        df = st.session_state.data
        if df is not None and len(st.session_state.features) > 0:
            features = st.session_state.features
            
            st.markdown("""
            <div style="display: flex; align-items: center; margin-top: 20px;">
                <img src="https://i.gifer.com/embedded/download/7VvS.gif" width="40px" style="margin-right: 10px;">
                <h3 style="color: #00f2ff; margin: 0;">SELECTED FEATURES</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the selected features in a cyberpunk styled manner
            cols = st.columns(len(features))
            for i, feature in enumerate(features):
                with cols[i]:
                    st.markdown(f"""
                    <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #00f2ff; 
                                padding: 15px; text-align: center; box-shadow: 0 0 10px rgba(0, 242, 255, 0.5);">
                        <div style="color: #ff00ff; font-family: 'Press Start 2P', cursive; font-size: 0.7rem; margin-bottom: 5px;">
                            FEATURE {i+1}
                        </div>
                        <div style="color: #00f2ff; font-size: 1rem; font-weight: bold;">
                            {feature}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add a "Next Step" button
            next_step_button_text = "PROCEED TO TRAIN/TEST SPLIT"
            if st.button(next_step_button_text, key="next_to_train_test_split"):
                complete_step(step_section)
                st.experimental_rerun()
        else:
            st.warning("Please complete the data preprocessing step first.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 4: Train/Test Split
    step_section = "Train/Test Split"
    is_current = step_section == st.session_state.current_step
    is_completed = step_section in st.session_state.completed_steps
    
    if is_current or is_completed:
        section_border = "#ff00ff" if is_current else "#00f2ff" if is_completed else "#ff00ff"
        section_glow = "0 0 15px #ff00ff" if is_current else "0 0 15px #00f2ff" if is_completed else "none"
        
        st.markdown(f"""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 2px solid {

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        st.markdown("""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #00f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h2 style="display: flex; align-items: center; gap: 10px;">
                <img src="https://i.gifer.com/7V7z.gif" width="30px"> 
                MODEL EVALUATION
            </h2>
        """, unsafe_allow_html=True)
        
        predictions = st.session_state.predictions
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        X_test = predictions['X_test']
        
        col1, col2 = st.columns(2)
        with col1:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.markdown(f"""
            <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #ff00ff; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 0 10px #ff00ff;">
                <h4 style="color: #ff00ff; margin-bottom: 10px;">RMSE</h4>
                <p style="color: #00f2ff; font-size: 1.8rem; font-weight: bold; font-family: 'Courier New', monospace; margin: 0;">{rmse:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            r2 = r2_score(y_test, y_pred)
            st.markdown(f"""
            <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #00f2ff; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 0 10px #00f2ff;">
                <h4 style="color: #00f2ff; margin-bottom: 10px;">R² SCORE</h4>
                <p style="color: #ff00ff; font-size: 1.8rem; font-weight: bold; font-family: 'Courier New', monospace; margin: 0;">{r2:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="color: #ff00ff; text-align: center; margin-top: 30px; margin-bottom: 20px; text-shadow: 0 0 10px #ff00ff;">
            ACTUAL VS PREDICTED VALUES
        </h3>
        """, unsafe_allow_html=True)
        
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
        
        fig = cyberpunk_actual_vs_predicted(results)
        st.plotly_chart(fig, use_container_width=True)
        
        if model_type == "Random Forest":
            st.markdown("""
            <h3 style="color: #ff00ff; text-align: center; margin-top: 30px; margin-bottom: 20px; text-shadow: 0 0 10px #ff00ff;">
                FEATURE IMPORTANCE
            </h3>
            """, unsafe_allow_html=True)
            
            importance = pd.DataFrame({'Feature': st.session_state.features, 'Importance': st.session_state.model.feature_importances_})
            importance = importance.sort_values('Importance', ascending=False)
            fig = cyberpunk_feature_importance(importance)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <img src="https://i.gifer.com/DKMj.gif" width="150px" style="margin-bottom: 10px;">
        </div>
        """, unsafe_allow_html=True)
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("DOWNLOAD PREDICTIONS", csv, "cyber_predictions.csv", "text/csv")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Cyberpunk footer
        st.markdown("""
        <div style="text-align: center; margin-top: 30px; opacity: 0.7;">
            <img src="https://i.gifer.com/FEc.gif" width="400px">
            <p style="color: #00f2ff; font-family: 'Courier New', monospace; margin-top: 10px; font-size: 0.8rem;">
                CYBER-ML PRO v2.0.77 | © 2025 SAMAD KIANI | NEURAL-NET ANALYTICS
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Step 8: Export Results
    step_section = "Export Results"
    is_current = step_section == st.session_state.current_step
    is_completed = step_section in st.session_state.completed_steps
    
    if is_current or is_completed:
        section_border = "#ff00ff" if is_current else "#00f2ff" if is_completed else "#ff00ff"
        section_glow = "0 0 15px #ff00ff" if is_current else "0 0 15px #00f2ff" if is_completed else "none"
        
        st.markdown(f"""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 2px solid {section_border}; padding: 15px; margin-bottom: 20px; box-shadow: {section_glow};">
            <h2 style="display: flex; align-items: center; gap: 10px;" class="pixel-flicker">
                <img src="https://i.gifer.com/7D7o.gif" width="30px"> 
                STEP 8: EXPORT RESULTS
            </h2>
        """, unsafe_allow_html=True)
        
        if is_current and not is_completed:
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://i.gifer.com/embedded/download/K5z.gif" width="150px">
                <p style="color: #00f2ff; margin-top: 15px; font-family: 'Share Tech Mono', monospace; font-size: 1.2rem;">
                    PREPARING EXPORT FILES...
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate export preparation with progress bar
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            cyber_success("EXPORT FILES READY")
            
            # Mark step as completed
            if step_section == st.session_state.current_step:
                complete_step(step_section)
        
        # Display export options if previous steps are completed
        if 'y_test' in st.session_state and 'y_pred' in st.session_state:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-top: 20px;">
                <img src="https://i.gifer.com/embedded/download/Fd5g.gif" width="40px" style="margin-right: 10px;">
                <h3 style="color: #00f2ff; margin: 0;">EXPORT OPTIONS</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Export predictions to CSV
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
            
            # Create export cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #00f2ff; padding: 15px; 
                            text-align: center; box-shadow: 0 0 10px rgba(0, 242, 255, 0.5);">
                    <div style="color: #ff00ff; font-family: 'Press Start 2P', cursive; font-size: 0.8rem; margin-bottom: 10px;">
                        PREDICTIONS CSV
                    </div>
                    <div style="margin: 15px 0;">
                        <img src="https://i.gifer.com/embedded/download/g0R5.gif" width="80px">
                    </div>
                    <div style="color: #cccccc; font-size: 0.8rem; margin-bottom: 15px;">
                        EXPORT MODEL PREDICTIONS
                    </div>
                """, unsafe_allow_html=True)
                
                # Generate CSV data
                csv = results.to_csv(index=False).encode('utf-8')
                
                # Download button
                st.download_button(
                    "DOWNLOAD PREDICTIONS", 
                    csv, 
                    "ml_predictions.csv", 
                    "text/csv",
                    key="download_predictions"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Model summary
                target = st.session_state.target
                features = st.session_state.features
                model_type = st.session_state.get('model_type', "Linear Regression")
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                model_report = f"""
                Model Type: {model_type}
                Target Variable: {target}
                Features: {', '.join(features)}
                
                Performance Metrics:
                - RMSE: {rmse:.4f}
                - R² Score: {r2:.4f}
                
                Dataset Size:
                - Total Records: {len(st.session_state.data)}
                - Training Set: {len(st.session_state.y_train)}
                - Testing Set: {len(st.session_state.y_test)}
                
                Generated by MLmadeEasy
                Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.markdown("""
                <div style="background-color: rgba(25, 25, 75, 0.8); border: 1px solid #ff00ff; padding: 15px; 
                            text-align: center; box-shadow: 0 0 10px rgba(255, 0, 255, 0.5);">
                    <div style="color: #00f2ff; font-family: 'Press Start 2P', cursive; font-size: 0.8rem; margin-bottom: 10px;">
                        MODEL REPORT
                    </div>
                    <div style="margin: 15px 0;">
                        <img src="https://i.gifer.com/embedded/download/Pijt.gif" width="80px">
                    </div>
                    <div style="color: #cccccc; font-size: 0.8rem; margin-bottom: 15px;">
                        EXPORT DETAILED MODEL REPORT
                    </div>
                """, unsafe_allow_html=True)
                
                # Generate TXT data
                txt = model_report.encode('utf-8')
                
                # Download button
                st.download_button(
                    "DOWNLOAD REPORT", 
                    txt, 
                    "ml_model_report.txt", 
                    "text/plain",
                    key="download_report"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display completion message if this is the final step
            if is_completed or is_current:
                st.markdown("""
                <div style="background-color: rgba(0, 50, 25, 0.8); border: 2px solid #00ff9f; padding: 20px; 
                            text-align: center; box-shadow: 0 0 15px #00ff9f; margin-top: 30px;">
                    <div style="color: #00ff9f; font-family: 'Press Start 2P', cursive; font-size: 1.2rem; margin-bottom: 15px;">
                        ML JOURNEY COMPLETE!
                    </div>
                    <div style="color: #ffffff; font-size: 1rem; margin-bottom: 20px; font-family: 'Share Tech Mono', monospace;">
                        Congratulations on successfully completing the ML journey!
                    </div>
                    <div style="margin: 15px 0;">
                        <img src="https://i.gifer.com/FEc.gif" width="200px">
                    </div>
                    <div style="color: #00f2ff; font-size: 0.8rem; margin-top: 15px; font-family: 'Share Tech Mono', monospace;">
                        Your ML model has been trained and evaluated. You can now make predictions and export your results.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Reset button for starting a new ML journey
                if st.button("START NEW ML JOURNEY", key="start_new_journey"):
                    st.session_state.clear()
                    st.experimental_rerun()
        else:
            st.warning("Please complete the predictions step first.")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Cyberpunk footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; opacity: 0.7;">
        <img src="https://i.gifer.com/FEc.gif" width="400px">
        <p style="color: #00f2ff; font-family: 'Share Tech Mono', monospace; margin-top: 10px; font-size: 0.8rem;">
            MLmadeEasy v2.0.77 | © 2025 | NEURAL-NET ANALYTICS
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Cyberpunk loading animation
def cyberpunk_loading_animation():
    return """
    <div style="text-align: center; margin: 20px 0;">
        <img src="https://i.gifer.com/5IPd.gif" width="150px">
        <p style="color: #00f2ff; margin-top: 10px; font-family: 'Share Tech Mono', monospace; font-size: 1.2rem;">
            INITIALIZING NEURAL NETWORKS...
        </p>
    </div>
    """

# Feature importance visualization with cyberpunk styling
def cyberpunk_feature_importance(importance_df):
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance', 
        color_continuous_scale=['#120052', '#4b0082', '#9400d3', '#ff00ff']
    )
    fig.update_layout(
        plot_bgcolor='rgba(10, 10, 30, 0.7)',
        paper_bgcolor='rgba(10, 10, 30, 0)',
        font=dict(color='#00f2ff'),
        title_font=dict(color='#ff00ff'),
        xaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff'),
            gridcolor='rgba(0, 242, 255, 0.2)'
        ),
        yaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff'),
            gridcolor='rgba(0, 242, 255, 0.2)'
        )
    )
    return fig

# Cyberpunk scatter plot with neon styling
def cyberpunk_scatter_plot(df, x, y):
    # Create scatter plot without trendline first
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        height=400,
        color_discrete_sequence=['#ff00ff']
    )
    
    # Update marker styling
    fig.update_traces(
        marker=dict(
            size=8,
            symbol='circle',
            line=dict(width=1, color='#00f2ff'),
            opacity=0.8
        )
    )
    
    # Try to add trendline manually using numpy's polyfit
    try:
        import numpy as np
        from scipy import stats
        
        # Get x and y values
        x_values = df[x].values
        y_values = df[y].values
        
        # Remove NaN values
        mask = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[mask]
        y_values = y_values[mask]
        
        if len(x_values) > 1:
            # Calculate trendline
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            x_range = np.linspace(min(x_values), max(x_values), 100)
            y_range = slope * x_range + intercept
            
            # Add trendline to plot
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='#00f2ff', width=3),
                    name='Trend'
                )
            )
    except:
        # If trendline calculation fails, just continue without it
        pass
        
    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(10, 10, 30, 0.7)',
        paper_bgcolor='rgba(10, 10, 30, 0)',
        font=dict(color='#00f2ff'),
        title_font=dict(color='#ff00ff'),
        xaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff'),
            gridcolor='rgba(0, 242, 255, 0.2)',
            zerolinecolor='#ff00ff'
        ),
        yaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff'),
            gridcolor='rgba(0, 242, 255, 0.2)',
            zerolinecolor='#ff00ff'
        )
    )
    
    return fig

# Cyberpunk correlation matrix
def cyberpunk_correlation_matrix(corr_matrix):
    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale=['#120052', '#4b0082', '#9400d3', '#ff00ff'],
        aspect="auto"
    )
    fig.update_layout(
        plot_bgcolor='rgba(10, 10, 30, 0.7)',
        paper_bgcolor='rgba(10, 10, 30, 0)',
        font=dict(color='#00f2ff'),
        title_font=dict(color='#ff00ff'),
        xaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff')
        ),
        yaxis=dict(
            title_font=dict(color='#00f2ff'),
            tickfont=dict(color='#00f2ff')
        )
    )
    return fig

# Cyberpunk actual vs predicted plot
def cyberpunk_actual_vs_predicted(results):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results.index, 
            y=results['Actual'], 
            name='Actual', 
            mode='markers', 
            marker=dict(
                color='#ff00ff',
                size=8,
                line=dict(width=1, color='#ff00ff'),
                symbol='circle',
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index, 
            y=results['Predicted'], 
            name='Predicted', 
            mode='markers', 
            marker=dict(
                color='#00f2ff',
                size=8,
                line=dict(width=1, color='#00f2ff'),
                symbol='diamond',
            )
        )
    )
    fig.update_layout(
        xaxis_title="Sample Index",
        yaxis_title="Value",
        height=500,
        legend=dict(font=dict(color='#00f2ff')),
        plot_bgcolor='rgba(10, 10, 30, 0.7)',
        paper_bgcolor='rgba(10, 10, 30, 0)',
        font=dict(color='#00f2ff'),
        xaxis=dict(
            gridcolor='rgba(0, 242, 255, 0.2)',
            zerolinecolor='#ff00ff'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 242, 255, 0.2)',
            zerolinecolor='#ff00ff'
        )
    )
    return fig

# Success message with cyberpunk styling
def cyber_success(message):
    return st.markdown(f"""
    <div style="background-color: rgba(0, 50, 20, 0.7); border: 1px solid #00ff9f; border-radius: 0; padding: 10px; margin: 10px 0; box-shadow: 0 0 10px #00ff9f;">
        <p style="color: #00ff9f; margin: 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">✅</span> {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Error message with cyberpunk styling
def cyber_error(message):
    return st.markdown(f"""
    <div style="background-color: rgba(50, 0, 20, 0.7); border: 1px solid #ff375f; border-radius: 0; padding: 10px; margin: 10px 0; box-shadow: 0 0 10px #ff375f;">
        <p style="color: #ff375f; margin: 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">⚠️</span> {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    # Check if it's the first run
    if st.session_state.get('first_run', True):
        # Show a quick start guide for the first run
        display_quick_start_guide()
        st.session_state.first_run = False
    else:
        # Run the main application
        main()
