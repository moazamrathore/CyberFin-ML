import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Configure page
st.set_page_config(
    page_title="CYBER-ML PRO",
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
    
    /* Preprocessing section */
    .preprocessing-panel {
        background-color: rgba(15, 25, 40, 0.8);
        border: 1px solid #00f2ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.3);
    }
    
    .preprocessing-option {
        margin-bottom: 10px;
        padding: 10px;
        background-color: rgba(20, 10, 40, 0.7);
        border-left: 3px solid #ff00ff;
    }
    
    .preprocessing-title {
        color: #ff00ff;
        font-weight: bold;
        margin-bottom: 5px;
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

# Cyberpunk actual vs predicted plot with line for actual and scatter for predicted
def cyberpunk_actual_vs_predicted(results):
    fig = go.Figure()
    
    # Sort the results by actual values to make the line continuous
    sorted_results = results.sort_values('Actual')
    
    # Add actual values as a line
    fig.add_trace(
        go.Scatter(
            x=sorted_results.index, 
            y=sorted_results['Actual'], 
            name='Actual', 
            mode='lines', 
            line=dict(
                color='#ff00ff',
                width=3,
            )
        )
    )
    
    # Add predicted values as scatter points
    fig.add_trace(
        go.Scatter(
            x=sorted_results.index, 
            y=sorted_results['Predicted'], 
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
        xaxis_title="Index",
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
        'predictions': None,
        'preprocessing_options': {
            'impute_missing': False,
            'scaling': 'none',
            'drop_duplicates': False,
            'remove_outliers': False
        }
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
                <li>Configure data preprocessing options</li>
                <li>Train your neural network model</li>
                <li>Analyze results and make predictions</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Step 2: Data Preprocessing (NEW SECTION)
    if st.session_state.steps['processed']:
        st.markdown("""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #ff00ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h2 style="display: flex; align-items: center; gap: 10px;">
                <img src="https://i.gifer.com/7SVt.gif" width="30px"> 
                DATA PREPROCESSING
            </h2>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <img src="https://i.gifer.com/embedded/download/KUxx.gif" width="150px">
            <p style="color: #00f2ff; margin-top: 15px; font-family: 'Courier New', monospace; font-size: 1.2rem;">
                CONFIGURE DATA PREPROCESSING OPTIONS
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Preprocessing options section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='preprocessing-panel'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #ff00ff;'>MISSING VALUES HANDLING</h4>", unsafe_allow_html=True)
            
            impute_missing = st.checkbox("IMPUTE MISSING VALUES", value=st.session_state.preprocessing_options['impute_missing'])
            if impute_missing:
                impute_strategy = st.selectbox("IMPUTATION STRATEGY:", 
                                               ["mean", "median", "most_frequent"], 
                                               index=0)
                st.session_state.preprocessing_options['impute_strategy'] = impute_strategy
            
            st.markdown("<div class='preprocessing-option'>", unsafe_allow_html=True)
            st.markdown("<div class='preprocessing-title'>DATA QUALITY</div>", unsafe_allow_html=True)
            drop_duplicates = st.checkbox("DROP DUPLICATE ROWS", value=st.session_state.preprocessing_options['drop_duplicates'])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='preprocessing-panel'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #ff00ff;'>FEATURE SCALING</h4>", unsafe_allow_html=True)
            
            scaling_options = ["none", "standard", "minmax", "robust"]
            scaling_labels = ["NONE", "STANDARD SCALER", "MIN-MAX SCALER", "ROBUST SCALER"]
            
            scaling_index = scaling_options.index(st.session_state.preprocessing_options.get('scaling', 'none'))
            scaling = st.selectbox("SCALING METHOD:", scaling_labels, index=scaling_index)
            
            # Map the selection back to the actual scaler name
            st.session_state.preprocessing_options['scaling'] = scaling_options[scaling_labels.index(scaling)]
            
            st.markdown("<div class='preprocessing-option'>", unsafe_allow_html=True)
            st.markdown("<div class='preprocessing-title'>OUTLIER HANDLING</div>", unsafe_allow_html=True)
            remove_outliers = st.checkbox("REMOVE OUTLIERS", value=st.session_state.preprocessing_options['remove_outliers'])
            if remove_outliers:
                outlier_threshold = st.slider("OUTLIER THRESHOLD (Z-SCORE):", 1.5, 5.0, 3.0, 0.1)
                st.session_state.preprocessing_options['outlier_threshold'] = outlier_threshold
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Update preprocessing options in session state
        st.session_state.preprocessing_options['impute_missing'] = impute_missing
        st.session_state.preprocessing_options['drop_duplicates'] = drop_duplicates
        st.session_state.preprocessing_options['remove_outliers'] = remove_outliers
        
        if st.button("APPLY PREPROCESSING", key="apply_preprocessing"):
            # Get the original data
            df = st.session_state.data.copy()
            features = st.session_state.features
            target = st.session_state.target
            
            # Store preprocessing information for user display
            preprocessing_steps = []
            
            # Apply preprocessing based on selected options
            try:
                # 1. Drop duplicates if selected
                if st.session_state.preprocessing_options['drop_duplicates']:
                    original_rows = len(df)
                    df = df.drop_duplicates()
                    removed_rows = original_rows - len(df)
                    preprocessing_steps.append(f"Removed {removed_rows} duplicate rows")
                
                # 2. Handle missing values if selected
                if st.session_state.preprocessing_options['impute_missing']:
                    strategy = st.session_state.preprocessing_options['impute_strategy']
                    imputer = SimpleImputer(strategy=strategy)
                    
                    # Only impute the selected features, not all columns
                    for feature in features:
                        missing_count = df[feature].isna().sum()
                        if missing_count > 0:
                            df[feature] = imputer.fit_transform(df[feature].values.reshape(-1, 1)).flatten()
                            preprocessing_steps.append(f"Imputed {missing_count} missing values in '{feature}' using {strategy}")
                    
                    # Also handle missing values in target if any
                    missing_count = df[target].isna().sum()
                    if missing_count > 0:
                        df[target] = imputer.fit_transform(df[target].values.reshape(-1, 1)).flatten()
                        preprocessing_steps.append(f"Imputed {missing_count} missing values in target '{target}' using {strategy}")
                
                # 3. Remove outliers if selected
                if st.session_state.preprocessing_options['remove_outliers']:
                    threshold = st.session_state.preprocessing_options['outlier_threshold']
                    original_rows = len(df)
                    
                    # Calculate z-scores for each feature and remove rows with outliers
                    for feature in features:
                        z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
                        df = df[z_scores < threshold]
                    
                    removed_rows = original_rows - len(df)
                    preprocessing_steps.append(f"Removed {removed_rows} outlier rows using z-score threshold of {threshold}")
                
                # Store the preprocessed data back to session state
                st.session_state.data_preprocessed = df
                st.session_state.preprocessing_steps = preprocessing_steps
                st.session_state.steps['preprocessed'] = True
                
                cyber_success("PREPROCESSING COMPLETED SUCCESSFULLY")
                
                # Show preprocessing summary
                if preprocessing_steps:
                    st.markdown("<h4 style='color: #00f2ff;'>PREPROCESSING SUMMARY:</h4>", unsafe_allow_html=True)
                    for step in preprocessing_steps:
                        st.markdown(f"<li style='color: #00ff9f; margin-left: 20px;'>{step}</li>", unsafe_allow_html=True)
                    
                    # Show before/after data shape
                    original_shape = st.session_state.data.shape
                    new_shape = df.shape
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                        <div style="text-align: center; color: #ff00ff;">
                            <div>ORIGINAL DATA</div>
                            <div style="font-size: 1.2rem; color: #00f2ff;">{original_shape[0]} rows × {original_shape[1]} columns</div>
                        </div>
                        <div style="text-align: center; color: #ff00ff;">
                            <div>PREPROCESSED DATA</div>
                            <div style="font-size: 1.2rem; color: #00f2ff;">{new_shape[0]} rows × {new_shape[1]} columns</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No preprocessing steps were applied based on your selections.")
            
            except Exception as e:
                cyber_error(f"ERROR DURING PREPROCESSING: {str(e)}")
        
        # Allow viewing the preprocessed data
        if st.session_state.get('data_preprocessed') is not None:
            if st.button("VIEW PREPROCESSED DATA", key="view_preprocessed"):
                st.dataframe(st.session_state.data_preprocessed.head().style.format("{:.2f}"), height=250)
        
        # Data analysis section
        if st.session_state.get('data_preprocessed') is not None:
            df = st.session_state.data_preprocessed
        else:
            df = st.session_state.data
        
        features = st.session_state.features
        target = st.session_state.target
        
        # Feature-target relationship visualization (without correlation matrix)
        st.markdown("""
        <h3 style="color: #ff00ff; text-align: center; margin-top: 20px;">
            FEATURE-TARGET RELATIONSHIPS
        </h3>
        """, unsafe_allow_html=True)
        
        selected_feature = st.selectbox("SELECT FEATURE TO PLOT:", features)
        fig = cyberpunk_scatter_plot(df, selected_feature, target)
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
        
        # Use preprocessed data if available, otherwise use original data
        if st.session_state.get('data_preprocessed') is not None:
            df = st.session_state.data_preprocessed
        else:
            df = st.session_state.data
            
        features = st.session_state.features
        target = st.session_state.target
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Apply scaling based on the selected method
        scaling_method = st.session_state.preprocessing_options.get('scaling', 'none')
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:  # 'none'
            # Create a dummy scaler that doesn't transform the data
            class IdentityScaler:
                def fit_transform(self, X):
                    return X
                def transform(self, X):
                    return X
            scaler = IdentityScaler()
        
        # Apply scaling only if not using dummy scaler
        if scaling_method != 'none':
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
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
        
        # Modified plot with line for actual values and scatter for predicted values
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
                CYBER-ML v2.0.77 | © MOAZAM RATHORE | CYBER FINANCE ML
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
