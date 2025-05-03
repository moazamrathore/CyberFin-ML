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

# Configure page
st.set_page_config(
    page_title="CYBER-ML PRO MADE BY SAMAD KIANI",
    page_icon="https://img.icons8.com/nolan/64/cyborg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Cyberpunk dark theme styling
st.markdown("""
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
        padding: 2rem;
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

# Cyberpunk header with animated GIF
def display_header():
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <img src="https://i.gifer.com/YAhr.gif" width="150px" style="margin-bottom: 10px;">
        <h1 style="font-size: 3rem; letter-spacing: 3px;">📊 CYBER-ML ANALYTICS</h1>
        <p style="color: #00f2ff; font-size: 1.2rem;">[ NEURAL-NET PREDICTION ENGINE v2.0.77 ]</p>
        <div style="background: linear-gradient(90deg, rgba(255,0,255,0) 0%, rgba(255,0,255,1) 50%, rgba(255,0,255,0) 100%); height: 2px; margin: 20px 0;"></div>
    </div>
    """, unsafe_allow_html=True)

# Cyberpunk themed sidebar header
def custom_sidebar_header():
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://i.pinimg.com/originals/1c/62/fe/1c62fe38d065c0a2929357f01c1e118e.gif" width="100%">
        <h3 style="color: #ff00ff; letter-spacing: 2px; margin-top: 10px;">SYSTEM CONTROLS</h3>
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
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        trendline="ols", 
        height=400,
        color_discrete_sequence=['#ff00ff']
    )
    fig.update_traces(
        marker=dict(
            size=8,
            symbol='circle',
            line=dict(width=1, color='#00f2ff'),
            opacity=0.8
        )
    )
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
    # Add cyberpunk-styled trendline
    fig.update_traces(
        line=dict(color='#00f2ff', width=3),
        selector=dict(type='scatter', mode='lines')
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

    # Sidebar Configuration
    with st.sidebar:
        custom_sidebar_header()
        
        st.markdown("""
        <div style="border: 1px solid #ff00ff; border-radius: 10px; padding: 15px; margin-bottom: 20px; background-color: rgba(25, 10, 40, 0.7);">
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("UPLOAD DATASET:", type=["csv", "xlsx"])
        
        if uploaded_file is None:
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://i.pinimg.com/originals/16/69/e5/1669e57e3a48988ab764a33e3d050347.gif" width="100%">
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="border: 1px solid #00f2ff; border-radius: 10px; padding: 15px; background-color: rgba(25, 10, 40, 0.7);">
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
    st.markdown("""
    <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #ff00ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        <h2 style="display: flex; align-items: center; gap: 10px;">
            <img src="https://i.gifer.com/7JcU.gif" width="30px"> 
            DATA UPLOAD & SELECTION
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
            
            cyber_success(f"SUCCESSFULLY LOADED {len(df)} RECORDS")
            
            st.markdown("""
            <h3 style="color: #00f2ff; margin-top: 20px;">DATASET PREVIEW:</h3>
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
            
        except Exception as e:
            cyber_error(f"ERROR LOADING FILE: {str(e)}")
    else:
        st.markdown("""
        <div class='feature-selector' style="text-align: center;">
            <img src="https://i.gifer.com/WMT.gif" width="200px" style="margin: 20px 0;">
            <h3 style="color: #ff00ff; margin-bottom: 15px;">HOW TO USE THE CYBER-ML SYSTEM:</h3>
            <ol style="text-align: left; color: #00f2ff; font-family: 'Courier New', monospace; font-size: 1.1rem;">
                <li>Upload any CSV or Excel file with numeric data</li>
                <li>Select target variable (what you want to predict)</li>
                <li>Choose features (variables used for prediction)</li>
                <li>The system will automatically process the neural networks</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Step 2: Data Analysis
    if st.session_state.steps['processed']:
        st.markdown("""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #00f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h2 style="display: flex; align-items: center; gap: 10px;">
                <img src="https://i.gifer.com/XOsX.gif" width="30px"> 
                DATA ANALYSIS
            </h2>
        """, unsafe_allow_html=True)
        
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <h3 style="color: #ff00ff; text-align: center;">FEATURE-TARGET RELATIONSHIPS</h3>
            """, unsafe_allow_html=True)
            
            selected_feature = st.selectbox("SELECT FEATURE TO PLOT:", features)
            fig = cyberpunk_scatter_plot(df, selected_feature, target)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("""
            <h3 style="color: #ff00ff; text-align: center;">CORRELATION MATRIX</h3>
            """, unsafe_allow_html=True)
            
            corr_matrix = df[features + [target]].corr()
            fig = cyberpunk_correlation_matrix(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <img src="https://i.gifer.com/A34R.gif" width="100px" style="margin-bottom: 10px;">
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("INITIALIZE NEURAL NETWORKS", key="train_model"):
            st.session_state.steps['ready_for_model'] = True
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.markdown("""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #ff00ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h2 style="display: flex; align-items: center; gap: 10px;">
                <img src="https://i.gifer.com/3Q7h.gif" width="30px"> 
                MODEL TRAINING
            </h2>
        """, unsafe_allow_html=True)
        
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
        
        with st.spinner(""):
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://i.gifer.com/QQrP.gif" width="200px">
                <p style="color: #ff00ff; margin-top: 15px; font-family: 'Courier New', monospace; font-size: 1.3rem;">
                    TRAINING NEURAL NETWORKS...
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            y_pred = model.predict(X_test_scaled)
            st.session_state.predictions = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}
            
            cyber_success("MODEL TRAINED SUCCESSFULLY!")
            st.balloons()
        
        st.markdown("</div>", unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
