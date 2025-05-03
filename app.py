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
    /* Preprocessing options */
    .preprocessing-section {
        background-color: rgba(25, 25, 75, 0.8);
        border: 1px solid #ff00ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 0 10px rgba(255, 0, 255, 0.5);
    }
    .preprocessing-title {
        color: #ff00ff;
        text-align: center;
        margin-bottom: 15px;
        font-size: 1.2rem;
        text-shadow: 0 0 5px #ff00ff;
    }
    /* Model training animation */
    .training-animation {
        text-align: center;
        margin: 20px 0;
        padding: 20px;
        background-color: rgba(26, 0, 56, 0.7);
        border: 1px solid #ff00ff;
        border-radius: 10px;
    }
    .training-text {
        color: #ff00ff;
        font-family: monospace;
        font-size: 1.3rem;
        margin-top: 15px;
        letter-spacing: 2px;
        text-shadow: 0 0 10px #ff00ff;
    }
</style>
""", unsafe_allow_html=True)

# Cyberpunk header
def display_header():
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; letter-spacing: 3px;">📊 CYBER-ML ANALYTICS</h1>
        <p style="color: #00f2ff; font-size: 1.2rem;">[ NEURAL-NET PREDICTION ENGINE v2.0.77 ]</p>
        <div style="background: linear-gradient(90deg, rgba(255,0,255,0) 0%, rgba(255,0,255,1) 50%, rgba(255,0,255,0) 100%); height: 2px; margin: 20px 0;"></div>
    </div>
    """, unsafe_allow_html=True)

# Cyberpunk themed sidebar header
def custom_sidebar_header():
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3 style="color: #ff00ff; letter-spacing: 2px; margin-top: 10px;">SYSTEM CONTROLS</h3>
        <div style="background: linear-gradient(90deg, rgba(0,242,255,0) 0%, rgba(0,242,255,1) 50%, rgba(0,242,255,0) 100%); height: 1px; margin: 10px 0;"></div>
    </div>
    """, unsafe_allow_html=True)

# Cyberpunk loading animation
def cyberpunk_loading_animation():
    return """
    <div style="text-align: center; margin: 20px 0;">
        <p style="color: #00f2ff; margin-top: 10px; font-family: monospace; font-size: 1.2rem;">
            ANALYZING DATA...
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

# Updated cyberpunk actual vs predicted plot with trend line and scattered points
def cyberpunk_actual_vs_predicted(results):
    # Create a figure
    fig = go.Figure()
    
    # Create scatter data for actual vs predicted
    actual = results['Actual']
    predicted = results['Predicted']
    
    # Calculate trend line
    model = LinearRegression()
    model.fit(actual.values.reshape(-1, 1), predicted.values.reshape(-1, 1))
    x_range = np.linspace(min(actual), max(actual), 100)
    y_range = model.predict(x_range.reshape(-1, 1)).flatten()
    
    # Add points
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(
                color='#ff00ff',
                size=8,
                line=dict(width=1, color='#ff00ff'),
                opacity=0.7
            )
        )
    )
    
    # Add trend line
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name='Trend',
            line=dict(color='#00f2ff', width=3)
        )
    )
    
    # Add perfect prediction line (y=x)
    fig.add_trace(
        go.Scatter(
            x=[min(actual), max(actual)],
            y=[min(actual), max(actual)],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='#ffff00', width=2, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Predicted vs Actual Values",
        xaxis_title="Actual",
        yaxis_title="Predicted",
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

# Apply data preprocessing
def preprocess_data(df, preprocessing_options):
    df_processed = df.copy()
    
    # Handle missing values
    if preprocessing_options.get('handle_missing'):
        missing_strategy = preprocessing_options.get('missing_strategy', 'mean')
        for col in df_processed.select_dtypes(include=np.number).columns:
            if df_processed[col].isnull().any():
                imputer = SimpleImputer(strategy=missing_strategy)
                df_processed[col] = imputer.fit_transform(df_processed[[col]])
    
    # Handle outliers
    if preprocessing_options.get('handle_outliers'):
        for col in df_processed.select_dtypes(include=np.number).columns:
            if preprocessing_options.get('outlier_strategy') == 'clip':
                q1 = df_processed[col].quantile(0.25)
                q3 = df_processed[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            elif preprocessing_options.get('outlier_strategy') == 'remove':
                q1 = df_processed[col].quantile(0.25)
                q3 = df_processed[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                mask = (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
                df_processed = df_processed[mask]
    
    # Apply feature transformations
    if preprocessing_options.get('feature_transform'):
        transform_type = preprocessing_options.get('transform_type')
        for col in df_processed.select_dtypes(include=np.number).columns:
            if transform_type == 'log':
                # Handle values <= 0 by adding a small constant if needed
                if (df_processed[col] <= 0).any():
                    min_val = df_processed[col].min()
                    if min_val <= 0:
                        df_processed[col] = df_processed[col] - min_val + 0.01
                df_processed[col] = np.log(df_processed[col])
            elif transform_type == 'sqrt':
                # Handle negative values
                if (df_processed[col] < 0).any():
                    min_val = df_processed[col].min()
                    if min_val < 0:
                        df_processed[col] = df_processed[col] - min_val
                df_processed[col] = np.sqrt(df_processed[col])
    
    # Apply feature scaling
    if preprocessing_options.get('feature_scaling'):
        scaling_method = preprocessing_options.get('scaling_method')
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
            
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    return df_processed

# Main Function
def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    # Display cyberpunk header
    display_header()
    
    # Session state initialization
    session_defaults = {
        'data': None, 'model': None, 'features': [], 'target': None,
        'steps': {'loaded': False, 'processed': False, 'preprocessed': False, 'trained': False},
        'predictions': None,
        'preprocessing_options': {
            'handle_missing': False,
            'missing_strategy': 'mean',
            'handle_outliers': False,
            'outlier_strategy': 'clip',
            'feature_transform': False,
            'transform_type': 'log',
            'feature_scaling': False,
            'scaling_method': 'standard'
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
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 1: Data Upload
    st.markdown("""
    <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #ff00ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        <h2 style="display: flex; align-items: center; gap: 10px;">
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
            <h3 style="color: #ff00ff; margin-bottom: 15px;">HOW TO USE THE CYBER-ML SYSTEM:</h3>
            <ol style="text-align: left; color: #00f2ff; font-family: 'Courier New', monospace; font-size: 1.1rem;">
                <li>Upload any CSV or Excel file with numeric data</li>
                <li>Select target variable (what you want to predict)</li>
                <li>Choose features (variables used for prediction)</li>
                <li>Preprocess data to enhance model performance</li>
                <li>Train and evaluate your neural network</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Step 2: Data Preprocessing (New Section)
    if st.session_state.steps['processed']:
        st.markdown("""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #00f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h2 style="display: flex; align-items: center; gap: 10px;">
                DATA PREPROCESSING
            </h2>
        """, unsafe_allow_html=True)
        
        df = st.session_state.data
        
        # Missing values preprocessing
        with st.expander("🧪 HANDLE MISSING VALUES"):
            st.markdown("<div class='preprocessing-section'>", unsafe_allow_html=True)
            
            missing_values = df.isnull().sum().sum()
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <p style="color: #00f2ff;">DETECTED MISSING VALUES: <span style="color: #ff00ff; font-weight: bold;">{missing_values}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            handle_missing = st.checkbox("ENABLE MISSING VALUES HANDLING", 
                                         value=st.session_state.preprocessing_options['handle_missing'])
            
            if handle_missing:
                missing_strategy = st.radio("IMPUTATION STRATEGY:", 
                                           ["mean", "median", "most_frequent", "constant"],
                                           index=["mean", "median", "most_frequent", "constant"].index(
                                               st.session_state.preprocessing_options['missing_strategy']))
                
                if missing_strategy == "constant":
                    fill_value = st.number_input("FILL VALUE:", value=0.0, step=0.1)
            
            st.session_state.preprocessing_options['handle_missing'] = handle_missing
            st.session_state.preprocessing_options['missing_strategy'] = missing_strategy if handle_missing else "mean"
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Outlier detection and handling
        with st.expander("🔍 HANDLE OUTLIERS"):
            st.markdown("<div class='preprocessing-section'>", unsafe_allow_html=True)
            
            handle_outliers = st.checkbox("ENABLE OUTLIER HANDLING", 
                                         value=st.session_state.preprocessing_options['handle_outliers'])
            
            if handle_outliers:
                outlier_strategy = st.radio("OUTLIER STRATEGY:", 
                                           ["clip", "remove"],
                                           index=["clip", "remove"].index(
                                               st.session_state.preprocessing_options['outlier_strategy']))
                
                st.info("Outliers are defined using the IQR method (values outside 1.5 * IQR)")
            
            st.session_state.preprocessing_options['handle_outliers'] = handle_outliers
            st.session_state.preprocessing_options['outlier_strategy'] = outlier_strategy if handle_outliers else "clip"
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature transformation
        with st.expander("🔄 FEATURE TRANSFORMATION"):
            st.markdown("<div class='preprocessing-section'>", unsafe_allow_html=True)
            
            feature_transform = st.checkbox("ENABLE FEATURE TRANSFORMATION", 
                                           value=st.session_state.preprocessing_options['feature_transform'])
            
            if feature_transform:
                transform_type = st.radio("TRANSFORMATION TYPE:", 
                                         ["log", "sqrt"],
                                         index=["log", "sqrt"].index(
                                             st.session_state.preprocessing_options['transform_type']))
                
                st.info(f"Apply {transform_type} transformation to numeric features")
            
            st.session_state.preprocessing_options['feature_transform'] = feature_transform
            st.session_state.preprocessing_options['transform_type'] = transform_type if feature_transform else "log"
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature scaling
        with st.expander("📊 FEATURE SCALING"):
            st.markdown("<div class='preprocessing-section'>", unsafe_allow_html=True)
            
            feature_scaling = st.checkbox("ENABLE FEATURE SCALING", 
                                         value=st.session_state.preprocessing_options['feature_scaling'])
            
            if feature_scaling:
                scaling_method = st.radio("SCALING METHOD:", 
                                         ["standard", "minmax", "robust"],
                                         index=["standard", "minmax", "robust"].index(
                                             st.session_state.preprocessing_options['scaling_method']))
                
                scaling_descriptions = {
                    "standard": "Standardization (Z-score): scales data to have mean=0 and std=1",
                    "minmax": "Min-Max Scaling: scales data to a range of [0,1]",
                    "robust": "Robust Scaling: scales data based on median and quartiles (robust to outliers)"
                }
                
                st.info(scaling_descriptions[scaling_method])
            
            st.session_state.preprocessing_options['feature_scaling'] = feature_scaling
            st.session_state.preprocessing_options['scaling_method'] = scaling_method if feature_scaling else "standard"
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Apply preprocessing button
        if st.button("APPLY PREPROCESSING", key="apply_preprocessing"):
            with st.spinner("PREPROCESSING DATA..."):
                # Get original data
                raw_df = st.session_state.data
                
                # Apply preprocessing
                preprocessed_df = preprocess_data(raw_df, st.session_state.preprocessing_options)
                
                # Store preprocessed data
                st.session_state.data_preprocessed = preprocessed_df
                st.session_state.steps['preprocessed'] = True
                
                # Display success message
                cyber_success("DATA PREPROCESSING COMPLETE")
                
                # Show before/after stats
                st.markdown("""
                <h3 style="color: #ff00ff; text-align: center; margin-top: 20px;">PREPROCESSING RESULTS</h3>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4 style='color: #00f2ff; text-align: center;'>BEFORE</h4>", unsafe_allow_html=True)
                    st.dataframe(raw_df.describe().style.format("{:.2f}"), height=200)
                    
                with col2:
                    st.markdown("<h4 style='color: #00f2ff; text-align: center;'>AFTER</h4>", unsafe_allow_html=True)
                    st.dataframe(preprocessed_df.describe().style.format("{:.2f}"), height=200)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 3: Data Analysis
    if st.session_state.steps['processed']:
        st.markdown("""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #ff00ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h2 style="display: flex; align-items: center; gap: 10px;">
                DATA ANALYSIS
            </h2>
        """, unsafe_allow_html=True)
        
        # Use preprocessed data if available, otherwise use original data
        if st.session_state.steps.get('preprocessed') and 'data_preprocessed' in st.session_state:
            df = st.session_state.data_preprocessed
            st.success("Using preprocessed data for analysis")
        else:
            df = st.session_state.data
            st.info("Using original data for analysis (no preprocessing applied)")
            
        features = st.session_state.features
        target = st.session_state.target
        
        # Feature-target relationship visualization
        st.markdown("""
        <h3 style="color: #ff00ff; text-align: center;">FEATURE-TARGET RELATIONSHIPS</h3>
        """, unsafe_allow_html=True)
        
        selected_feature = st.selectbox("SELECT FEATURE TO PLOT:", features)
        fig = cyberpunk_scatter_plot(df, selected_feature, target)
        st.plotly_chart(fig, use_container_width=True)
            
        if st.button("TRAIN THE MODEL", key="train_model"):
            st.session_state.steps['ready_for_model'] = True
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 4: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.markdown("""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #00f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h2 style="display: flex; align-items: center; gap: 10px;">
                MODEL TRAINING
            </h2>
        """, unsafe_allow_html=True)
        
        # Use preprocessed data if available, otherwise use original data
        if st.session_state.steps.get('preprocessed') and 'data_preprocessed' in st.session_state:
            df = st.session_state.data_preprocessed
        else:
            df = st.session_state.data
            
        features = st.session_state.features
        target = st.session_state.target
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Only apply scaling if not already done in preprocessing
        if not st.session_state.preprocessing_options.get('feature_scaling'):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
        
        with st.spinner(""):
            st.markdown("""
            <div class="training-animation">
                <p class="training-text">TRAINING NEURAL NETWORKS...</p>
            </div>
            """, unsafe_allow_html=True)
            
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            y_pred = model.predict(X_test_scaled)
            st.session_state.predictions = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}
            
            cyber_success("MODEL TRAINED SUCCESSFULLY!")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 5: Evaluation
    if st.session_state.steps.get('trained'):
        st.markdown("""
        <div style="background-color: rgba(20, 10, 40, 0.7); border: 1px solid #ff00ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h2 style="display: flex; align-items: center; gap: 10px;">
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
            PREDICTED VS ACTUAL VALUES
        </h3>
        """, unsafe_allow_html=True)
        
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
        
        # Use updated visualization function
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
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("DOWNLOAD PREDICTIONS", csv, "cyber_predictions.csv", "text/csv")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Cyberpunk footer
        st.markdown("""
        <div style="text-align: center; margin-top: 30px; opacity: 0.7;">
            <p style="color: #00f2ff; font-family: 'Courier New', monospace; margin-top: 10px; font-size: 0.8rem;">
                CYBER-ML v2.0.77 | ©  MOAZAM RATHORE | CYBER FINANCE ML
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
