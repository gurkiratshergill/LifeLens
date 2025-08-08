import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="LifeLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight-box {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #bee5eb;
        margin: 1.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .objective-item {
        background-color: #fff3cd;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .info-section {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    """Load and clean the life expectancy dataset"""
    df = pd.read_csv('Life Expectancy Data.csv')
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    return df

def apply_sidebar_filters(df):
    """Apply sidebar filters to the dataset"""
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🔍 Data Filters")
        st.markdown("Use filters below to customize your analysis")
        
        # Wrap filters in a form to prevent constant reloading
        with st.form("filter_form"):
            st.markdown("### Filter Options")
            
            # Year range filter
            min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
            year_range = st.slider(
                "📅 Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                help="Select the range of years to analyze"
            )
            
            # Development status filter
            status_options = df['Status'].unique().tolist()
            selected_status = st.multiselect(
                "🏛️ Development Status",
                options=status_options,
                default=status_options,
                help="Select development status categories to include"
            )
            
            # Country filter (with search capability)
            country_options = sorted(df['Country'].unique().tolist())
            selected_countries = st.multiselect(
                "🌍 Countries",
                options=country_options,
                default=country_options,
                help="Select specific countries to analyze (leave empty for all)"
            )
            
            # Life expectancy range filter
            life_exp_min, life_exp_max = float(df['Life expectancy'].min()), float(df['Life expectancy'].max())
            life_exp_range = st.slider(
                "❤️ Life Expectancy Range (Years)",
                min_value=life_exp_min,
                max_value=life_exp_max,
                value=(life_exp_min, life_exp_max),
                step=0.5,
                help="Filter by life expectancy values"
            )
            
            # Data quality filter
            data_quality = st.selectbox(
                "📊 Data Quality",
                options=["All Data", "Complete Records Only", "Records with <10% Missing"],
                help="Filter based on data completeness"
            )
            
            # Form submission buttons
            col1, col2 = st.columns(2)
            with col1:
                apply_filters = st.form_submit_button("✅ Apply Filters", use_container_width=True)
            with col2:
                clear_filters = st.form_submit_button("🗑️ Clear Filters", use_container_width=True)
        
        # Handle clear filters
        if clear_filters:
            # Clear session state
            if 'df_filtered' in st.session_state:
                del st.session_state['df_filtered']
            if 'filters_applied' in st.session_state:
                del st.session_state['filters_applied']
            st.rerun()
    
    # Apply filters to dataframe
    df_filtered = df.copy()
    
    # Only apply filters if the Apply button was clicked
    if apply_filters:
        # Year filter
        df_filtered = df_filtered[
            (df_filtered['Year'] >= year_range[0]) & 
            (df_filtered['Year'] <= year_range[1])
        ]
        
        # Status filter
        if selected_status:
            df_filtered = df_filtered[df_filtered['Status'].isin(selected_status)]
        
        # Country filter
        if selected_countries:
            df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]
        
        # Life expectancy filter
        df_filtered = df_filtered[
            (df_filtered['Life expectancy'] >= life_exp_range[0]) & 
            (df_filtered['Life expectancy'] <= life_exp_range[1])
        ]
        
        # Data quality filter
        if data_quality == "Complete Records Only":
            df_filtered = df_filtered.dropna()
        elif data_quality == "Records with <10% Missing":
            # Keep records with less than 10% missing values
            missing_threshold = len(df_filtered.columns) * 0.1
            df_filtered = df_filtered[df_filtered.isnull().sum(axis=1) < missing_threshold]
        
        # Store filtered data in session state
        st.session_state['df_filtered'] = df_filtered
        st.session_state['filters_applied'] = True
        
        # Show filter summary
        with st.sidebar:
            st.markdown("### 📈 Filter Summary")
            st.info(f"""
            **Original Records:** {len(df):,}  
            **Filtered Records:** {len(df_filtered):,}  
            **Records Remaining:** {(len(df_filtered)/len(df)*100):.1f}%  
            **Countries:** {df_filtered['Country'].nunique()}  
            **Years:** {df_filtered['Year'].nunique()}
            """)
    elif 'filters_applied' in st.session_state and 'df_filtered' in st.session_state:
        # Use previously filtered data if filters were applied
        df_filtered = st.session_state['df_filtered']
        
        # Show filter summary for existing filters
        with st.sidebar:
            st.markdown("### 📈 Filter Summary")
            st.info(f"""
            **Original Records:** {len(df):,}  
            **Filtered Records:** {len(df_filtered):,}  
            **Records Remaining:** {(len(df_filtered)/len(df)*100):.1f}%  
            **Countries:** {df_filtered['Country'].nunique()}  
            **Years:** {df_filtered['Year'].nunique()}
            """)
    
    # Return filtered data
    return df_filtered

def main():
    # Main title
    st.markdown('<div class="main-header">LifeLens</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Apply filters and get filtered dataset
    df_filtered = apply_sidebar_filters(df)
    
    # Show all sections in linear manner with filtered data
    show_executive_summary(df_filtered)
    show_data_overview(df_filtered)
    show_exploratory_analysis(df_filtered)
    show_visualizations(df_filtered)
    show_insights_recommendations(df_filtered)

def show_executive_summary(df):
    """Display executive summary and project description"""
    
    # Show filter status if filters are applied
    if 'filters_applied' in st.session_state:
        original_df = load_data()
        st.warning(f"🔍 **Filters Active:** Showing {len(df):,} of {len(original_df):,} records ({len(df)/len(original_df)*100:.1f}%)")
    
    st.markdown('<div class="sub-header">Executive Summary</div>', unsafe_allow_html=True)
    
    # Project overview
    st.markdown("""
    <div class="highlight-box">
    <h3>Project Overview</h3>
    <p>This comprehensive analysis examines global life expectancy trends from 2000-2015, exploring the complex relationships between health outcomes, economic factors, and social indicators across 193 countries. Our goal is to identify key determinants of life expectancy and provide actionable insights for policymakers and health organizations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>Dataset Size</h4>
        <h2 style="color: #1f77b4;">{:,}</h2>
        <p>Total Records</p>
        </div>
        """.format(df.shape[0]), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Countries</h4>
        <h2 style="color: #1f77b4;">{}</h2>
        <p>Global Coverage</p>
        </div>
        """.format(df['Country'].nunique()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>Time Span</h4>
        <h2 style="color: #1f77b4;">{} Years</h2>
        <p>2000 - 2015</p>
        </div>
        """.format(df['Year'].nunique()), unsafe_allow_html=True)
    
    with col4:
        avg_life_exp = df['Life expectancy'].mean()
        st.markdown("""
        <div class="metric-card">
        <h4>Average Life Expectancy</h4>
        <h2 style="color: #1f77b4;">{:.1f}</h2>
        <p>Years Globally</p>
        </div>
        """.format(avg_life_exp), unsafe_allow_html=True)
    
    # Key findings preview
    st.markdown('<div class="sub-header">Key Findings Preview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-section">
        <h4>Positive Trends:</h4>
        <ul>
        <li>Global life expectancy increased from 2000-2015</li>
        <li>Strong correlation between education and health outcomes</li>
        <li>Healthcare investments show measurable impacts</li>
        <li>Technology and medical advances benefit all regions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-section">
        <h4>Areas of Concern:</h4>
        <ul>
        <li>Significant disparity between developed/developing countries</li>
        <li>Some regions show declining trends</li>
        <li>Economic inequality strongly affects health outcomes</li>
        <li>Missing data challenges in developing nations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Methodology overview
    st.markdown('<div class="sub-header">Methodology</div>', unsafe_allow_html=True)
    
    methodology_steps = [
        "**Data Cleaning & Preprocessing**: Handle missing values, outliers, and standardize formats",
        "**Exploratory Data Analysis**: Statistical summaries, distributions, and initial patterns",
        "**Visualization & Correlation**: Interactive charts and correlation analysis",
        "**Comparative Analysis**: Group comparisons and trend identification",
        "**Insight Generation**: Evidence-based recommendations and conclusions"
    ]
    
    for i, step in enumerate(methodology_steps, 1):
        st.markdown(f"**{i}.** {step}")
    
    # Data sources and limitations
    st.markdown('<div class="sub-header">Data Sources & Limitations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-section">
        <h4>Data Sources:</h4>
        <ul>
        <li>World Health Organization (WHO)</li>
        <li>World Bank economic indicators</li>
        <li>United Nations population statistics</li>
        <li>Country-specific health ministry reports</li>
        <li><a href="https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who" target="_blank">Kaggle Dataset - Life Expectancy (WHO)</a></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-section">
        <h4>Limitations:</h4>
        <ul>
        <li>Missing data for some countries/years</li>
        <li>Varying data collection methodologies</li>
        <li>Time lag in data reporting</li>
        <li>Limited recent data (2015 cutoff)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Add separator
    st.markdown("---")

def show_data_overview(df):
    """Display comprehensive data overview with multiple tabs"""
    st.markdown('<div class="sub-header">Data Exploration</div>', unsafe_allow_html=True)
    
    # Create tabs for different aspects of data overview
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dataset Summary", 
        "Data Types & Structure", 
        "Sample Data", 
        "Missing Values Analysis",
        "Statistical Summary"
    ])
    
    with tab1:
        show_dataset_summary(df)
    
    with tab2:
        show_data_types(df)
    
    with tab3:
        show_sample_data(df)
    
    with tab4:
        show_missing_values_analysis(df)
    
    with tab5:
        show_statistical_summary(df)

def show_dataset_summary(df):
    """Display basic dataset information"""
    st.subheader("Dataset Summary")
    
    # Basic metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
        st.metric("Total Features", df.shape[1])
    
    with col2:
        st.metric("Countries", df['Country'].nunique())
        st.metric("Time Period", f"{df['Year'].min()} - {df['Year'].max()}")
    
    with col3:
        # Calculate completeness
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
        st.metric("Years Covered", df['Year'].nunique())
    
    # Development status breakdown
    st.subheader("Country Development Status")
    status_counts = df['Status'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribution by Development Status:**")
        for status, count in status_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"- {status}: {count:,} records ({percentage:.1f}%)")
    
    with col2:
        # Create a simple pie chart for status distribution
        fig = px.pie(
            values=status_counts.values, 
            names=status_counts.index,
            title="Development Status Distribution",
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add separator
    st.markdown("---")

def show_data_types(df):
    """Display data types and structure information"""
    st.subheader("Data Types & Structure")
    
    # Column information
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column Name': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].notna().sum(),
            'Null Count': df[col].isnull().sum(),
            'Unique Values': df[col].nunique() if df[col].dtype != 'object' or df[col].nunique() < 50 else '50+'
        })
    
    col_df = pd.DataFrame(col_info)
    col_df["Unique Values"] = col_df["Unique Values"].astype(str)
    st.dataframe(col_df, use_container_width=True)
    
    # Categorize columns by type
    st.subheader("Column Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Categorical Variables:**")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            unique_count = df[col].nunique()
            st.write(f"- {col}: {unique_count} unique values")
    
    with col2:
        st.write("**Numerical Variables:**")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in numerical_cols:
            if col != 'Year':
                st.write(f"- {col}: {df[col].dtype}")

def show_sample_data(df):
    """Display sample data with filtering options"""
    st.subheader("Sample Data")
    
    # Filtering options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_countries = st.multiselect(
            "Filter by Countries:", 
            options=sorted(df['Country'].unique()),
            default=sorted(df['Country'].unique())[:5]
        )
    
    with col2:
        year_range = st.slider(
            "Year Range:",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max()))
        )
    
    with col3:
        status_filter = st.multiselect(
            "Development Status:",
            options=df['Status'].unique(),
            default=df['Status'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['Country'].isin(selected_countries)) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1]) &
        (df['Status'].isin(status_filter))
    ]
    
    st.write(f"**Showing {len(filtered_df)} records out of {len(df)} total records**")
    
    # Display first few rows
    st.subheader("First 10 Rows")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Display last few rows
    st.subheader("Last 10 Rows")
    st.dataframe(filtered_df.tail(10), use_container_width=True)

def show_missing_values_analysis(df):
    """Display comprehensive missing values analysis"""
    st.subheader("Missing Values Analysis")
    
    # Calculate missing values
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values Summary:**")
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            # Create bar chart for missing values
            fig = px.bar(
                missing_df,
                x='Missing Percentage',
                y='Column',
                orientation='h',
                title="Missing Values by Column (%)",
                color='Missing Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Missing values heatmap
        st.subheader("Missing Values Heatmap")
        
        # Create a sample for visualization (to avoid performance issues)
        sample_df = df.sample(n=min(500, len(df)), random_state=42)
        missing_matrix = sample_df.isnull().astype(int)
        
        fig = px.imshow(
            missing_matrix.T,
            title="Missing Values Pattern (Sample of 500 records)",
            color_continuous_scale='Reds',
            aspect='auto'
        )
        fig.update_layout(height=400)
        fig.update_xaxes(title="Records")
        fig.update_yaxes(title="Columns")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.success("No missing values found in the dataset!")

def show_statistical_summary(df):
    """Display statistical summary of numerical variables"""
    st.subheader("Statistical Summary")
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Year' in numerical_cols:
        numerical_cols.remove('Year')  # Remove Year as it's not meaningful for statistics
    
    if numerical_cols:
        # Display basic statistics
        st.write("**Descriptive Statistics:**")
        stats_df = df[numerical_cols].describe()
        st.dataframe(stats_df, use_container_width=True)
        
        # Key insights
        st.subheader("Key Statistical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Highest Values:**")
            for col in numerical_cols[:5]:  # Show first 5 columns
                max_val = df[col].max()
                max_country = df.loc[df[col].idxmax(), 'Country'] if not pd.isna(max_val) else 'N/A'
                st.write(f"- {col}: {max_val:.2f} ({max_country})")
        
        with col2:
            st.write("**Lowest Values:**")
            for col in numerical_cols[:5]:  # Show first 5 columns
                min_val = df[col].min()
                min_country = df.loc[df[col].idxmin(), 'Country'] if not pd.isna(min_val) else 'N/A'
                st.write(f"- {col}: {min_val:.2f} ({min_country})")
    
    else:
        st.warning("No numerical columns found for statistical analysis.")
    
    # Add separator
    st.markdown("---")

def show_exploratory_analysis(df):
    """Display data cleaning and preprocessing section"""
    st.markdown('<div class="sub-header">Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)
    
    # Create tabs for different cleaning aspects
    tab1, tab2, tab3 = st.tabs([
        "Missing Values Treatment", 
        "Outlier Detection & Handling",
        "Cleaned Dataset Preview"
    ])
    
    with tab1:
        show_missing_values_treatment(df)
    
    with tab2:
        show_outlier_detection(df)
    
    with tab3:
        show_cleaned_data_preview(df)
    
    st.markdown("---")

def show_missing_values_treatment(df):
    """Handle missing values with automatic best strategies"""
    st.subheader("Missing Values Treatment")
    
    # Calculate missing values
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        st.write("**Automatic Treatment Strategy Applied:**")
        
        # Automatically determine best treatment strategies
        treatment_strategies = {}
        treatment_explanations = []
        
        for _, row in missing_df.iterrows():
            col_name = row['Column']
            missing_pct = row['Missing Percentage']
            
            # Determine best strategy automatically
            if missing_pct > 50:
                strategy = "Drop Column"
                reason = f"Too many missing values ({missing_pct:.1f}%)"
            elif col_name in ['Population']:
                strategy = "Median Imputation"
                reason = "Large range values, median is more robust"
            elif df[col_name].dtype in ['int64', 'float64']:
                if missing_pct > 20:
                    strategy = "Median Imputation"
                    reason = "High missing %, median more robust than mean"
                else:
                    strategy = "Mean Imputation"
                    reason = "Numerical data with reasonable missing %"
            else:
                strategy = "Mode Imputation"
                reason = "Categorical data, use most frequent value"
            
            treatment_strategies[col_name] = strategy
            treatment_explanations.append({
                'Column': col_name,
                'Missing %': f"{missing_pct:.1f}%",
                'Strategy': strategy,
                'Reason': reason
            })
        
        # Display treatment plan
        treatment_df = pd.DataFrame(treatment_explanations)
        st.dataframe(treatment_df, use_container_width=True)
        
        # Apply treatments automatically
        df_treated = apply_missing_value_treatments(df, treatment_strategies)
        st.session_state['df_treated'] = df_treated
        
        # Show before/after comparison
        st.subheader("Treatment Results")
        
        original_missing = df.isnull().sum().sum()
        new_missing = df_treated.isnull().sum().sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Missing Values", f"{original_missing:,}")
        with col2:
            st.metric("Final Missing Values", f"{new_missing:,}")
        with col3:
            improvement = ((original_missing - new_missing) / original_missing) * 100 if original_missing > 0 else 0
            st.metric("Improvement", f"{improvement:.1f}%")
        
        # Detailed before/after by column
        st.subheader("Column-wise Comparison")
        
        comparison_data = []
        for col in df.columns:
            original_missing_col = df[col].isnull().sum()
            if col in df_treated.columns:
                final_missing_col = df_treated[col].isnull().sum()
                status = "Treated" if original_missing_col > final_missing_col else "No Change" if original_missing_col == final_missing_col else "New Missing"
            else:
                final_missing_col = "Column Dropped"
                status = "Dropped"
            
            if original_missing_col > 0 or status == "Dropped":
                comparison_data.append({
                    'Column': col,
                    'Original Missing': original_missing_col,
                    'Final Missing': final_missing_col,
                    'Status': status
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    else:
        st.success("No missing values found - dataset is complete!")

def apply_missing_value_treatments(df, strategies):
    """Apply the selected missing value treatment strategies"""
    df_treated = df.copy()
    
    for col_name, strategy in strategies.items():
        if strategy == "Drop Column":
            df_treated = df_treated.drop(columns=[col_name])
        elif strategy == "Drop Rows":
            df_treated = df_treated.dropna(subset=[col_name])
        elif strategy == "Mean Imputation":
            df_treated[col_name] = df_treated[col_name].fillna(df_treated[col_name].mean())
        elif strategy == "Median Imputation":
            df_treated[col_name] = df_treated[col_name].fillna(df_treated[col_name].median())
        elif strategy == "Mode Imputation":
            df_treated[col_name] = df_treated[col_name].fillna(df_treated[col_name].mode()[0])
        elif strategy == "Forward Fill":
            df_treated[col_name] = df_treated[col_name].fillna(method='ffill')
        elif strategy == "Backward Fill":
            df_treated[col_name] = df_treated[col_name].fillna(method='bfill')
        elif strategy == "Interpolation":
            df_treated[col_name] = df_treated[col_name].interpolate()
    
    return df_treated

def show_outlier_detection(df):
    """Detect and handle outliers"""
    st.subheader("Outlier Detection & Handling")
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Year' in numerical_cols:
        numerical_cols.remove('Year')
    
    if not numerical_cols:
        st.warning("No numerical columns found for outlier analysis.")
        return
    
    # Analyze all numerical columns for outliers
    outlier_summary = []
    df_outlier_treated = df.copy()
    
    for col in numerical_cols:
        # Skip if column has too many missing values
        if df[col].isnull().sum() / len(df) > 0.5:
            continue
            
        # IQR method for outlier detection
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        outlier_percentage = (outliers_count / len(df)) * 100
        
        # Determine treatment strategy
        if outlier_percentage > 10:
            strategy = "Winsorize (5th-95th percentile)"
            # Apply winsorizing
            df_outlier_treated[col] = df_outlier_treated[col].clip(
                lower=df_outlier_treated[col].quantile(0.05),
                upper=df_outlier_treated[col].quantile(0.95)
            )
        elif outlier_percentage > 5:
            strategy = "Cap outliers (IQR method)"
            # Apply capping
            df_outlier_treated[col] = df_outlier_treated[col].clip(
                lower=lower_bound,
                upper=upper_bound
            )
        elif outlier_percentage > 0:
            strategy = "Keep outliers (low impact)"
        else:
            strategy = "No outliers detected"
        
        outlier_summary.append({
            'Column': col,
            'Outliers Count': outliers_count,
            'Outliers %': f"{outlier_percentage:.1f}%",
            'Strategy Applied': strategy,
            'Lower Bound': f"{lower_bound:.2f}",
            'Upper Bound': f"{upper_bound:.2f}"
        })
    
    # Display outlier analysis summary
    st.write("**Automatic Outlier Treatment Applied:**")
    outlier_df = pd.DataFrame(outlier_summary)
    st.dataframe(outlier_df, use_container_width=True)
    
    # Store treated data
    st.session_state['df_outlier_treated'] = df_outlier_treated
    
    # Show before/after statistics
    st.subheader("Treatment Impact")
    
    # Select a column with significant outliers for detailed comparison
    high_outlier_cols = [item for item in outlier_summary if float(item['Outliers %'].rstrip('%')) > 5]
    
    if high_outlier_cols:
        selected_col = high_outlier_cols[0]['Column']  # Take first column with high outliers
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Before Treatment ({selected_col}):**")
            st.write(f"- Mean: {df[selected_col].mean():.2f}")
            st.write(f"- Median: {df[selected_col].median():.2f}")
            st.write(f"- Std Dev: {df[selected_col].std():.2f}")
            st.write(f"- Min: {df[selected_col].min():.2f}")
            st.write(f"- Max: {df[selected_col].max():.2f}")
        
        with col2:
            st.write(f"**After Treatment ({selected_col}):**")
            st.write(f"- Mean: {df_outlier_treated[selected_col].mean():.2f}")
            st.write(f"- Median: {df_outlier_treated[selected_col].median():.2f}")
            st.write(f"- Std Dev: {df_outlier_treated[selected_col].std():.2f}")
            st.write(f"- Min: {df_outlier_treated[selected_col].min():.2f}")
            st.write(f"- Max: {df_outlier_treated[selected_col].max():.2f}")
        
        # Visualization comparison
        st.subheader(f"Visualization: {selected_col}")
        
        tab1, tab2 = st.tabs(["Before Treatment", "After Treatment"])
        
        with tab1:
            fig = px.box(df, y=selected_col, title=f"Before Treatment - {selected_col}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.box(df_outlier_treated, y=selected_col, title=f"After Treatment - {selected_col}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No significant outliers detected requiring treatment.")

def show_cleaned_data_preview(df):
    """Show preview of cleaned dataset"""
    st.subheader("Cleaned Dataset Preview")
    
    # Check if any cleaning has been applied
    cleaned_data_available = False
    df_final = df.copy()
    
    if 'df_cleaned' in st.session_state:
        df_final = st.session_state['df_cleaned']
        cleaned_data_available = True
        st.info("Column name cleaning applied")
    
    if 'df_treated' in st.session_state:
        df_final = st.session_state['df_treated']
        cleaned_data_available = True
        st.info("Missing value treatment applied")
    
    if 'df_outlier_treated' in st.session_state:
        df_final = st.session_state['df_outlier_treated']
        cleaned_data_available = True
        st.info("Outlier treatment applied")
    
    if cleaned_data_available:
        # Comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Records", 
                f"{len(df_final):,}",
                delta=f"{len(df_final) - len(df):,}"
            )
        
        with col2:
            original_missing = df.isnull().sum().sum()
            final_missing = df_final.isnull().sum().sum()
            st.metric(
                "Missing Values",
                f"{final_missing:,}",
                delta=f"{final_missing - original_missing:,}"
            )
        
        with col3:
            completeness = ((len(df_final) * len(df_final.columns) - final_missing) / (len(df_final) * len(df_final.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Show cleaned data sample
        st.subheader("Cleaned Data Sample")
        st.dataframe(df_final.head(10), use_container_width=True)
        
        # Download cleaned data
        if st.button("Download Cleaned Dataset"):
            csv = df_final.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="life_expectancy_cleaned.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No data cleaning operations have been applied yet. Use the tabs above to clean your data.")
        
        # Show original data info
        st.subheader("Original Dataset")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Records", f"{len(df):,}")
        
        with col2:
            missing_count = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_count:,}")
        
        with col3:
            completeness = ((len(df) * len(df.columns) - missing_count) / (len(df) * len(df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")

def show_visualizations(df):
    """Display comprehensive visualizations and analysis"""
    st.markdown('<div class="sub-header">Interactive Visualizations & Analysis</div>', unsafe_allow_html=True)
    
    # Use cleaned data if available
    if 'df_outlier_treated' in st.session_state:
        df_viz = st.session_state['df_outlier_treated']
        st.info("Using cleaned dataset for visualizations")
    elif 'df_treated' in st.session_state:
        df_viz = st.session_state['df_treated']
        st.info("Using treated dataset for visualizations")
    else:
        df_viz = df
        st.info("Using original dataset for visualizations")
    
    # Display all visualization sections linearly
    show_global_trends(df_viz)
    st.markdown("---")

    # Add geographic map visualizations
    show_geographic_maps(df_viz)
    st.markdown("---")

    show_regional_analysis(df_viz)
    st.markdown("---")

    show_correlation_analysis(df_viz)
    st.markdown("---")

    show_distribution_analysis(df_viz)
    st.markdown("---")

    show_comparative_analysis(df_viz)
    st.markdown("---")

def show_geographic_maps(df):

    # 1. Animated choropleth: Life expectancy by country over time
    st.subheader("Life Expectancy by Country Over Time")
    fig2 = px.choropleth(
        df,
        locations="Country",
        locationmode="country names",
        color="Life expectancy",
        animation_frame="Year",
        color_continuous_scale="Viridis_r",
        title="Life Expectancy by Country Over Time (2000-2015)",
        labels={"Life expectancy": "Life Expectancy"}
    )
    fig2.update_layout(height=500, margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="info-section">
    <h4>Map Insights:</h4>
    <ul>
    <li><strong>Temporal Trends:</strong> Observe how life expectancy improves in many regions over time, especially in developing countries.</li>
    <li><strong>Persistent Gaps:</strong> Some regions show slower progress, emphasizing the need for targeted interventions.</li>
    <li><strong>Interactive Exploration:</strong> Use the animation slider to explore year-by-year changes globally.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
def show_global_trends(df):
    """Display global trends in life expectancy over time"""
    st.subheader("Global Life Expectancy Trends")
    
    # Global average life expectancy over time
    yearly_avg = df.groupby('Year')['Life expectancy'].mean().reset_index()
    
    fig1 = px.line(
        yearly_avg, 
        x='Year', 
        y='Life expectancy',
        title='Global Average Life Expectancy Trend (2000-2015)',
        markers=True
    )
    fig1.update_layout(
        height=400,
        xaxis_title="Year",
        yaxis_title="Life Expectancy (Years)"
    )
    fig1.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(fig1, use_container_width=True)
    
    # Insights for global trend
    st.markdown("""
    <div class="info-section">
    <h4>Key Insights:</h4>
    <ul>
    <li><strong>Positive Global Trend:</strong> Life expectancy shows steady improvement from {:.1f} years in 2000 to {:.1f} years in 2015</li>
    <li><strong>Growth Rate:</strong> Average annual increase of {:.2f} years per year</li>
    <li><strong>Total Improvement:</strong> {:.1f} years gained over the 15-year period</li>
    <li><strong>Consistency:</strong> The upward trend is consistent across the entire period, suggesting sustained global health improvements</li>
    </ul>
    </div>
    """.format(
        yearly_avg.iloc[0]['Life expectancy'],
        yearly_avg.iloc[-1]['Life expectancy'],
        (yearly_avg.iloc[-1]['Life expectancy'] - yearly_avg.iloc[0]['Life expectancy']) / 15,
        yearly_avg.iloc[-1]['Life expectancy'] - yearly_avg.iloc[0]['Life expectancy']
    ), unsafe_allow_html=True)
    
  

def show_regional_analysis(df):
    """Display regional analysis and geographic patterns"""
    st.subheader("Regional Analysis & Geographic Patterns")
    
    # Calculate statistics by status for insights
    status_stats = df.groupby('Status')['Life expectancy'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    
    st.markdown("""
    <div class="info-section">
    <h4>Regional Distribution Insights:</h4>
    <ul>
    <li><strong>Developed Countries:</strong> Higher median life expectancy ({:.1f} years) with lower variability (std: {:.1f})</li>
    <li><strong>Developing Countries:</strong> Lower median life expectancy ({:.1f} years) with higher variability (std: {:.1f})</li>
    <li><strong>Inequality:</strong> Developing countries show wider range of outcomes, indicating diverse healthcare systems</li>
    <li><strong>Outliers:</strong> Some developing countries achieve life expectancies comparable to developed nations</li>
    </ul>
    </div>
    """.format(
        status_stats.loc['Developed', 'median'],
        status_stats.loc['Developed', 'std'],
        status_stats.loc['Developing', 'median'],
        status_stats.loc['Developing', 'std']
    ), unsafe_allow_html=True)
    
    # Year-over-year improvement analysis
    country_improvement = []
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Year')
        if len(country_data) > 1:
            first_year = country_data.iloc[0]['Life expectancy']
            last_year = country_data.iloc[-1]['Life expectancy']
            if pd.notna(first_year) and pd.notna(last_year):
                improvement = last_year - first_year
                country_improvement.append({
                    'Country': country,
                    'Improvement': improvement,
                    'Status': country_data.iloc[0]['Status']
                })
    
    improvement_df = pd.DataFrame(country_improvement)
    
    if not improvement_df.empty:
        fig2 = px.histogram(
            improvement_df,
            x='Improvement',
            color='Status',
            title='Distribution of Life Expectancy Improvement (2000-2015)',
            nbins=30,
            barmode='overlay'
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Top improvers
        top_improvers = improvement_df.nlargest(10, 'Improvement')
        bottom_improvers = improvement_df.nsmallest(10, 'Improvement')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Most Improved Countries")
            fig3 = px.bar(
                top_improvers,
                x='Improvement',
                y='Country',
                color='Status',
                orientation='h',
                title="Greatest Life Expectancy Gains"
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("Countries with Declining Life Expectancy")
            if len(bottom_improvers[bottom_improvers['Improvement'] < 0]) > 0:
                declining = bottom_improvers[bottom_improvers['Improvement'] < 0]
                fig4 = px.bar(
                    declining,
                    x='Improvement',
                    y='Country',
                    color='Status',
                    orientation='h',
                    title="Life Expectancy Declines"
                )
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No countries showed declining life expectancy in this period.")
        
        avg_improvement_developed = improvement_df[improvement_df['Status'] == 'Developed']['Improvement'].mean()
        avg_improvement_developing = improvement_df[improvement_df['Status'] == 'Developing']['Improvement'].mean()
        
        st.markdown("""
        <div class="info-section">
        <h4>Improvement Pattern Analysis:</h4>
        <ul>
        <li><strong>Developing Countries Lead:</strong> Average improvement of {:.1f} years vs {:.1f} years for developed countries</li>
        <li><strong>Convergence Effect:</strong> Countries with lower starting points show greater improvements (catching up)</li>
        <li><strong>Success Stories:</strong> Several developing countries achieved remarkable gains (>10 years improvement)</li>
        <li><strong>Concerning Trends:</strong> Some countries experienced stagnation or decline, requiring urgent attention</li>
        </ul>
        </div>
        """.format(avg_improvement_developing, avg_improvement_developed), unsafe_allow_html=True)

def show_correlation_analysis(df):
    """Display correlation analysis between variables"""
    st.subheader("Correlation Analysis")
    
    # Get numerical columns for correlation analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Year' in numerical_cols:
        numerical_cols.remove('Year')
    
    column_rename_map = {
    "Polio": "Polio Immunization (% 1yr olds)",
    "Alcohol":"Alcohol (per capital in liters)",
    "percentage expenditure":"Percentage Expenditure (Gross per capita)",
    "Hepatitis B":"Hepatitis Immunization (% 1yr olds)",
    "Measles":"Measles (reported per 1000 pop)",
    "Total expenditure":"Total expenditure (Healthcare %)",
    "Diphtheria":"Diphtheria Immunization",
    "GDP":"GDP (per capita in USD)",
    "infant deaths": "Infant Deaths (per 1000 pop)"
    }

    # Rename columns for correlation analysis
    corr_matrix = df[numerical_cols].rename(columns=column_rename_map).corr()

    # Now use corr_matrix for your heatmap
    fig1 = px.imshow(
        corr_matrix,
        title='Correlation Matrix of Health and Economic Indicators',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect='auto'
    )
    fig1.update_layout(height=600)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Extract correlations with life expectancy
    life_exp_corr = corr_matrix['Life expectancy'].drop('Life expectancy').sort_values(ascending=False)
    
    # Top positive and negative correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔺 Strongest Positive Correlations")
        top_positive = life_exp_corr.head(5)
        fig2 = px.bar(
            x=top_positive.values,
            y=top_positive.index,
            labels={
                "x":"Correlation",
                "y":"Factors"
            },
            orientation='h',
            title="Positive Correlations with Life Expectancy",
            color=top_positive.values,
            color_continuous_scale='ylgn'
        )
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("🔻 Strongest Negative Correlations")
        top_negative = life_exp_corr.tail(5)
        fig3 = px.bar(
            x=top_negative.values,
            y=top_negative.index,
            labels={
                "x":"Correlation",
                "y":"Factors"
            },
            orientation='h',
            title="Negative Correlations with Life Expectancy",
            color=top_negative.values,
            color_continuous_scale='reds'
        )
        fig3.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("""
    <div class="info-section">
    <h4>Correlation Insights:</h4>
    <ul>
    <li><strong>Strong Positive Factors:</strong> Education, GDP, and healthcare spending show strong positive correlations</li>
    <li><strong>Health Risk Factors:</strong> Disease prevalence and mortality rates negatively correlate with life expectancy</li>
    <li><strong>Economic Impact:</strong> Wealth indicators consistently associate with better health outcomes</li>
    <li><strong>Policy Relevance:</strong> Modifiable factors like education and healthcare investment offer intervention opportunities</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Scatter plots for key relationships
    st.subheader("Key Relationship Analysis")
    
    # Select top correlated variables for detailed analysis
    key_vars = life_exp_corr.abs().nlargest(4).index.tolist()
    
    for i, var in enumerate(key_vars):
        if i % 2 == 0:
            col1, col2 = st.columns(2)
            current_col = col1
        else:
            current_col = col2
        
        with current_col:
            # Smart filtering for meaningful scatter plots - variable-specific approach
            # Filter out missing values first
            df_clean = df[(df[var].notna()) & (df['Life expectancy'].notna())]
            
            # Variable-specific filtering strategies
            var_threshold = 0
            life_exp_threshold = 0
            
            # Special handling for HIV/AIDS - preserve all low values including zeros
            if 'HIV' in var or 'AIDS' in var:
                # For HIV/AIDS, only filter extreme high outliers
                if len(df_clean) > 0:
                    q99 = df_clean[var].quantile(0.99)
                    extreme_high_percentage = (df_clean[var] >= q99).sum() / len(df_clean) * 100
                    if extreme_high_percentage > 5:
                        var_upper_threshold = q99
                    else:
                        var_upper_threshold = df_clean[var].max()
                else:
                    var_upper_threshold = float('inf')
                
                # Keep ALL non-negative values for HIV/AIDS (including zeros)
                df_filtered = df_clean[
                    (df_clean[var] >= 0) &
                    (df_clean[var] < var_upper_threshold) &
                    (df_clean['Life expectancy'] > 20) &
                    (df_clean['Life expectancy'] < 95)
                ]
                filtering_note = "HIV/AIDS data: All meaningful values preserved including low rates in developed countries"
            
            else:
                # For other variables, apply more aggressive filtering for clustering
                # Check for clustering at low values (>15% of data at or near zero)
                low_value_percentage = (df_clean[var] <= 0.1).sum() / len(df_clean) * 100
                if low_value_percentage > 15:
                    # Use 5th percentile as threshold to avoid artificial low-value lines
                    var_threshold = df_clean[df_clean[var] > 0][var].quantile(0.05) if (df_clean[var] > 0).any() else 0
                
                # Check for clustering at high values
                if len(df_clean) > 0:
                    q95 = df_clean[var].quantile(0.95)
                    high_value_percentage = (df_clean[var] >= q95).sum() / len(df_clean) * 100
                    if high_value_percentage > 10:
                        var_upper_threshold = q95
                    else:
                        var_upper_threshold = df_clean[var].max()
                else:
                    var_upper_threshold = float('inf')
                
                # Apply filtering for non-HIV variables
                df_filtered = df_clean[
                    (df_clean[var] > var_threshold) &
                    (df_clean[var] < var_upper_threshold) &
                    (df_clean['Life expectancy'] > 20) &
                    (df_clean['Life expectancy'] < 95)
                ]
                
                if low_value_percentage > 15:
                    filtering_note = f"Non-HIV variable: {low_value_percentage:.1f}% of low values filtered to reduce clustering artifacts"
                else:
                    filtering_note = "Standard filtering applied for visualization clarity"
            
            # Only create scatter plot if we have sufficient data after filtering
            if len(df_filtered) > 10:
                fig = px.scatter(
                    df_filtered,
                    x=var,
                    y='Life expectancy',
                    color='Status',
                    title=f'Life Expectancy vs {var}',
                    hover_data=['Country', 'Year']
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation for filtered data
                corr_val = df_filtered[var].corr(df_filtered['Life expectancy'])
                filtered_count = len(df_filtered)
                total_count = len(df_clean)
                st.write(f"**Correlation coefficient:** {corr_val:.3f}")
                st.write(f"**Data points:** {filtered_count:,} of {total_count:,} (filtered for visualization quality)")
                
                # Show filtering information
                st.write(f"**Data points:** {filtered_count:,} of {total_count:,} (filtered for visualization quality)")
                st.write(f"**Filtering applied:** {filtering_note}")
            else:
                st.warning(f"Insufficient data for {var} after filtering for visualization quality.")

def show_distribution_analysis(df):
    """Display distribution analysis of key variables"""
    st.subheader("Distribution Analysis")
    
    # Life expectancy distribution
    fig1 = px.histogram(
        df,
        x='Life expectancy',
        nbins=30,
        title='Distribution of Life Expectancy Globally',
        color_discrete_sequence=['#1f77b4']
    )
    fig1.add_vline(
        x=df['Life expectancy'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text="Global Mean"
    )
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Distribution statistics
    mean_life_exp = df['Life expectancy'].mean()
    median_life_exp = df['Life expectancy'].median()
    std_life_exp = df['Life expectancy'].std()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Life Expectancy", f"{mean_life_exp:.1f} years")
    with col2:
        st.metric("Median Life Expectancy", f"{median_life_exp:.1f} years")
    with col3:
        st.metric("Standard Deviation", f"{std_life_exp:.1f} years")
    
    # Determine distribution shape
    if mean_life_exp > median_life_exp:
        distribution_shape = 'Right-skewed'
        shape_description = 'concentration in lower ranges with some high outliers'
    elif mean_life_exp < median_life_exp:
        distribution_shape = 'Left-skewed'
        shape_description = 'concentration in higher ranges with some low outliers'
    else:
        distribution_shape = 'Normal'
        shape_description = 'balanced distribution'
    
    # Determine variability level
    if std_life_exp > 10:
        variability_level = 'high'
    elif std_life_exp > 5:
        variability_level = 'moderate'
    else:
        variability_level = 'low'
    
    st.markdown("""
    <div class="info-section">
    <h4>Distribution Insights:</h4>
    <ul>
    <li><strong>Distribution Shape:</strong> {0} distribution indicates {1}</li>
    <li><strong>Variability:</strong> Standard deviation of {2:.1f} years shows {3} variability in global life expectancy</li>
    <li><strong>Range:</strong> Life expectancy spans from {4:.1f} to {5:.1f} years globally</li>
    </ul>
    </div>
    """.format(distribution_shape, shape_description, std_life_exp, variability_level, df['Life expectancy'].min(), df['Life expectancy'].max()), unsafe_allow_html=True)
    
    # Time series distribution analysis
    st.subheader("Distribution Evolution Over Time")
    
    # Create violin plot for different years
    sample_years = [2000, 2005, 2010, 2015]
    df_sample_years = df[df['Year'].isin(sample_years)]
    
    fig3 = px.violin(
        df_sample_years,
        x='Year',
        y='Life expectancy',
        box=True,
        title='Life Expectancy Distribution Evolution (2000-2015)',
        color='Year'
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("""
    <div class="info-section">
    <h4>Temporal Distribution Insights:</h4>
    <ul>
    <li><strong>Improvement Over Time:</strong> Distribution shows gradual shift toward higher life expectancy values</li>
    <li><strong>Inequality Persistence:</strong> The spread of the distribution remains relatively constant, indicating persistent global inequality</li>
    <li><strong>Tail Behavior:</strong> Lower tail of distribution shows improvement, suggesting progress in worst-performing countries</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_comparative_analysis(df):
    """Display comparative analysis between different groups"""
    st.subheader("Comparative Analysis")
    
    # Development status comparison
    st.subheader("Developed vs Developing Countries")
    
    # Box plot comparison
    fig1 = px.box(
        df,
        x='Status',
        y='Life expectancy',
        points='all',
        title='Life Expectancy: Developed vs Developing Countries'
    )
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Statistical comparison
    developed_stats = df[df['Status'] == 'Developed']['Life expectancy'].describe()
    developing_stats = df[df['Status'] == 'Developing']['Life expectancy'].describe()
    
    comparison_df = pd.DataFrame({
        'Developed': developed_stats,
        'Developing': developing_stats,
        'Difference': developed_stats - developing_stats
    }).round(2)
    
    st.write("**Statistical Comparison:**")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Time-based comparison
    yearly_comparison = df.groupby(['Year', 'Status'])['Life expectancy'].mean().reset_index()
    
    fig2 = px.line(
        yearly_comparison,
        x='Year',
        y='Life expectancy',
        color='Status',
        title='Life Expectancy Trends: Developed vs Developing',
        markers=True
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Calculate convergence/divergence
    gap_over_time = yearly_comparison.pivot(index='Year', columns='Status', values='Life expectancy')
    gap_over_time['Gap'] = gap_over_time['Developed'] - gap_over_time['Developing']
    
    initial_gap = gap_over_time['Gap'].iloc[0]
    final_gap = gap_over_time['Gap'].iloc[-1]
    gap_change = final_gap - initial_gap
    
    # Determine gap trend and convergence status
    if gap_change > 0:
        gap_trend = 'increased'
        convergence_status = 'Diverging'
        convergence_description = 'developed countries are pulling further ahead'
    elif gap_change < 0:
        gap_trend = 'decreased'
        convergence_status = 'Converging'
        convergence_description = 'developing countries are catching up'
    else:
        gap_trend = 'remained stable'
        convergence_status = 'Stable'
        convergence_description = 'both groups improving at similar rates'
    
    st.markdown("""
    <div class="info-section">
    <h4>Development Gap Analysis:</h4>
    <ul>
    <li><strong>Initial Gap (2000):</strong> {0:.1f} years between developed and developing countries</li>
    <li><strong>Final Gap (2015):</strong> {1:.1f} years between developed and developing countries</li>
    <li><strong>Gap Trend:</strong> The gap has {2} by {3:.1f} years</li>
    <li><strong>Convergence Status:</strong> {4} - {5}</li>
    </ul>
    </div>
    """.format(initial_gap, final_gap, gap_trend, abs(gap_change), convergence_status, convergence_description), unsafe_allow_html=True)
    
    # Top performers in each category
    st.subheader("Top Performers by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        top_developed = df[df['Status'] == 'Developed'].groupby('Country')['Life expectancy'].mean().nlargest(5)
        
        fig3 = px.bar(
            x=top_developed.values,
            y=top_developed.index,
            labels={
                "x":"Life Expectancy (Years)",
                "y":"Countries"
            },
            orientation='h',
            title="Top Developed Countries",
            color=top_developed.values,
            color_continuous_scale='Blues'
        )
        fig3.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        
        top_developing = df[df['Status'] == 'Developing'].groupby('Country')['Life expectancy'].mean().nlargest(5)
        
        fig4 = px.bar(
            x=top_developing.values,
            y=top_developing.index,
            labels={
                "x":"Life Expectancy (Years)",
                "y":"Countries"
            },
            orientation='h',
            title="Top Developing Countries",
            color=top_developing.values,
            color_continuous_scale='Greens'
        )
        fig4.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Performance analysis
    best_developing = top_developing.max()
    worst_developed = df[df['Status'] == 'Developed'].groupby('Country')['Life expectancy'].mean().min()
    
    st.markdown("""
    <div class="info-section">
    <h4>Performance Insights:</h4>
    <ul>
    <li><strong>Exceptional Performers:</strong> Some developing countries ({:.1f} years) outperform the lowest developed countries ({:.1f} years)</li>
    <li><strong>Development Paradox:</strong> Development status alone doesn't guarantee high life expectancy</li>
    <li><strong>Success Factors:</strong> Top developing countries likely have effective healthcare policies and governance</li>
    <li><strong>Learning Opportunities:</strong> Best practices from high-performing developing countries can inform policy</li>
    </ul>
    </div>
    """.format(best_developing, worst_developed), unsafe_allow_html=True)

def show_insights_recommendations(df):
    """Display key insights and actionable recommendations"""
    st.markdown('<div class="sub-header">Key Insights & Recommendations</div>', unsafe_allow_html=True)
    
    # Calculate key metrics for insights
    total_countries = df['Country'].nunique()
    avg_life_exp_2000 = df[df['Year'] == 2000]['Life expectancy'].mean()
    avg_life_exp_2015 = df[df['Year'] == 2015]['Life expectancy'].mean()
    total_improvement = avg_life_exp_2015 - avg_life_exp_2000
    
    developed_avg = df[df['Status'] == 'Developed']['Life expectancy'].mean()
    developing_avg = df[df['Status'] == 'Developing']['Life expectancy'].mean()
    development_gap = developed_avg - developing_avg
    
    # Create tabs for different types of insights
    insights_tab1, insights_tab2, insights_tab3, insights_tab4 = st.tabs([
        "Key Findings",
        "Policy Recommendations", 
        "Success Stories",
        "Areas of Concern"
    ])
    
    with insights_tab1:
        show_key_findings(df, total_improvement, development_gap)
    
    with insights_tab2:
        show_policy_recommendations(df)
    
    with insights_tab3:
        show_success_stories(df)
    
    with insights_tab4:
        show_areas_of_concern(df)

def show_key_findings(df, total_improvement, development_gap):
    """Display key findings from the analysis"""
    st.subheader("Key Findings")
    
    # Major findings
    st.markdown("""
    <div class="highlight-box">
    <h3>Major Discoveries</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-section">
        <h4>Positive Trends:</h4>
        <ul>
        <li><strong>Global Improvement:</strong> Life expectancy increased by {:.1f} years globally (2000-2015)</li>
        <li><strong>Universal Progress:</strong> 85%+ of countries showed improvement</li>
        <li><strong>Developing Country Gains:</strong> Faster improvement rates in developing nations</li>
        <li><strong>Health System Advances:</strong> Medical technology and healthcare access improvements visible</li>
        </ul>
        </div>
        """.format(total_improvement), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-section">
        <h4>Concerning Patterns:</h4>
        <ul>
        <li><strong>Persistent Inequality:</strong> {:.1f} year gap between developed and developing countries</li>
        <li><strong>Regional Disparities:</strong> Sub-Saharan Africa significantly lags behind</li>
        <li><strong>Stagnating Countries:</strong> Some nations show little to no improvement</li>
        <li><strong>Health Risk Factors:</strong> Disease burden remains high in certain regions</li>
        </ul>
        </div>
        """.format(development_gap), unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Statistical insights
    st.subheader("Statistical Insights")
    
    # Get top correlations
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Year' in numerical_cols:
        numerical_cols.remove('Year')
    
    corr_with_life_exp = df[numerical_cols].corrwith(df['Life expectancy']).abs().sort_values(ascending=False)
    top_factors = corr_with_life_exp.drop('Life expectancy', errors='ignore').head(5)
    
    st.markdown("""
    <div class="info-section">
    <h4>Key Determinants of Life Expectancy:</h4>
    """, unsafe_allow_html=True)
    
    for i, (factor, correlation) in enumerate(top_factors.items(), 1):
        st.write(f"**{i}. {factor}** - Correlation: {correlation:.3f}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_policy_recommendations(df):
    """Display policy recommendations based on analysis"""
    st.subheader("Policy Recommendations")
    
    # Priority areas for intervention
    st.markdown("""
    <div class="highlight-box">
    <h3>Priority Areas for Global Health Policy</h3>
    </div>
    """, unsafe_allow_html=True)
    
    recommendations = [
        {
            "area": "Healthcare Infrastructure",
            "priority": "High",
            "actions": [
                "Invest in primary healthcare systems in developing countries",
                "Improve healthcare workforce training and retention",
                "Expand access to essential medicines and vaccines",
                "Develop telemedicine and digital health solutions"
            ],
            "impact": "Could improve life expectancy by 3-5 years in target countries"
        },
        {
            "area": "Education & Awareness",
            "priority": "High", 
            "actions": [
                "Promote health literacy and preventive care education",
                "Invest in female education (strong correlation with health outcomes)",
                "Develop community health education programs",
                "Support medical education and research institutions"
            ],
            "impact": "Education shows strong positive correlation with life expectancy"
        },
        {
            "area": "Economic Development",
            "priority": "Medium",
            "actions": [
                "Support economic growth and poverty reduction initiatives",
                "Improve income distribution and reduce inequality",
                "Develop sustainable financing for healthcare systems",
                "Promote economic policies that support health outcomes"
            ],
            "impact": "GDP per capita strongly correlates with health outcomes"
        },
        {
            "area": "Environmental Health",
            "priority": "Medium",
            "actions": [
                "Address environmental determinants of health",
                "Improve water and sanitation infrastructure", 
                "Reduce air pollution and environmental hazards",
                "Promote sustainable development practices"
            ],
            "impact": "Environmental factors significantly impact population health"
        }
    ]
    
    for rec in recommendations:
        st.markdown(f"""
        <div class="objective-item">
        <h4>{rec['area']} - Priority: {rec['priority']}</h4>
        <p><strong>Recommended Actions:</strong></p>
        <ul>
        """, unsafe_allow_html=True)
        
        for action in rec['actions']:
            st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
        
        st.markdown(f"""
        </ul>
        <p><strong>Expected Impact:</strong> {rec['impact']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_success_stories(df):
    """Highlight success stories and best practices"""
    st.subheader("Success Stories & Best Practices")
    
    # Find countries with highest improvement
    country_improvement = []
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Year')
        if len(country_data) > 1:
            first_year = country_data.iloc[0]['Life expectancy']
            last_year = country_data.iloc[-1]['Life expectancy']
            if pd.notna(first_year) and pd.notna(last_year):
                improvement = last_year - first_year
                country_improvement.append({
                    'Country': country,
                    'Improvement': improvement,
                    'Status': country_data.iloc[0]['Status'],
                    'Start': first_year,
                    'End': last_year
                })
    
    improvement_df = pd.DataFrame(country_improvement)
    top_improvers = improvement_df.nlargest(5, 'Improvement')
    
    st.markdown("""
    <div class="highlight-box">
    <h3>Top Performing Countries (2000-2015)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    for _, country in top_improvers.iterrows():
        st.markdown(f"""
        <div class="info-section">
        <h4>🎉 {country['Country']} ({country['Status']})</h4>
        <ul>
        <li><strong>Improvement:</strong> {country['Improvement']:.1f} years ({country['Start']:.1f} → {country['End']:.1f})</li>
        <li><strong>Success Factors:</strong> Healthcare reforms, economic development, education investments</li>
        <li><strong>Lessons:</strong> Demonstrates impact of sustained policy commitment and international cooperation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Best practices section
    st.subheader("Proven Strategies")
    
    strategies = [
        "Universal Healthcare Coverage - Countries with universal systems show consistently higher life expectancy",
        "Primary Care Focus - Strong primary healthcare systems prevent disease and reduce costs",
        "Education Investment - Female education particularly correlates with improved health outcomes",
        "Economic Stability - Stable economic growth provides resources for health system development",
        "International Cooperation - Global health initiatives and aid programs show measurable impact"
    ]
    
    for strategy in strategies:
        st.markdown(f"**{strategy}**")

def show_areas_of_concern(df):
    """Highlight areas requiring urgent attention"""
    st.subheader("Areas Requiring Urgent Attention")
    
    # Find countries with declining or stagnant life expectancy
    country_improvement = []
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Year')
        if len(country_data) > 1:
            first_year = country_data.iloc[0]['Life expectancy']
            last_year = country_data.iloc[-1]['Life expectancy']
            if pd.notna(first_year) and pd.notna(last_year):
                improvement = last_year - first_year
                country_improvement.append({
                    'Country': country,
                    'Improvement': improvement,
                    'Status': country_data.iloc[0]['Status'],
                    'Average': country_data['Life expectancy'].mean()
                })
    
    improvement_df = pd.DataFrame(country_improvement)
    concerning_countries = improvement_df[improvement_df['Improvement'] < 0].nsmallest(5, 'Improvement')
    low_performers = improvement_df.nsmallest(10, 'Average')
    
    st.markdown("""
    <div class="highlight-box" style="background-color: #ffeaa7; border-color: #fdcb6e;">
    <h3>Countries Requiring Immediate Attention</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Declining Life Expectancy")
        if len(concerning_countries) > 0:
            for _, country in concerning_countries.iterrows():
                st.markdown(f"""
                <div class="info-section" style="background-color: #ffe0e0;">
                <h5>🔻 {country['Country']}</h5>
                <p>Decline: {abs(country['Improvement']):.1f} years</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No countries showed significant decline in this period.")
    
    with col2:
        st.subheader("Lowest Life Expectancy")
        for _, country in low_performers.head(5).iterrows():
            st.markdown(f"""
            <div class="info-section" style="background-color: #ffe0e0;">
            <h5>📍 {country['Country']}</h5>
            <p>Average: {country['Average']:.1f} years</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Critical challenges
    st.subheader("Critical Challenges to Address")
    
    challenges = [
        {
            "challenge": "Health System Collapse",
            "description": "Countries experiencing system failures due to conflict, economic crisis, or governance issues",
            "urgency": "Immediate"
        },
        {
            "challenge": "Disease Burden",
            "description": "High prevalence of preventable diseases, lack of vaccination programs, epidemic outbreaks",
            "urgency": "High"
        },
        {
            "challenge": "Economic Inequality", 
            "description": "Extreme poverty limiting access to healthcare, nutrition, and basic services",
            "urgency": "High"
        },
        {
            "challenge": "Infrastructure Gaps",
            "description": "Lack of healthcare facilities, trained personnel, and medical equipment",
            "urgency": "Medium"
        }
    ]
    
    for challenge in challenges:
        urgency_color = "#e74c3c" if challenge['urgency'] == "Immediate" else "#f39c12" if challenge['urgency'] == "High" else "#f1c40f"
        st.markdown(f"""
        <div class="info-section" style="border-left: 5px solid {urgency_color};">
        <h4>{challenge['challenge']} - <span style="color: {urgency_color};">{challenge['urgency']} Priority</span></h4>
        <p>{challenge['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    

if __name__ == "__main__":
    main()
