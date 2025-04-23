import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
import os

# Create a results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Set page config
st.set_page_config(
    page_title="PRL Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Probabilistic Reward Learning (PRL) Analysis Dashboard")
st.markdown("""
This interactive dashboard allows you to analyze data from a Probabilistic Reward Learning experiment.
The experiment includes different blocks with varying reward probabilities (80/20 and 50/50).
""")

@st.cache_data
def load_and_preprocess_data():
    """
    Load data from master.csv and preprocess it:
    1. Filter out practice trials
    2. Handle missing data
    3. Identify optimal choices in 80/20 block
    """
    # Load data
    data_path = 'master.csv'
    df = pd.read_csv(data_path)
    
    # Filter out practice trials
    df_filtered = df[(df['is_practice'] == False) & (df['block'] != 'Practice')]
    
    # Handle missing data
    missing_selected = df_filtered['selected_option'].isna().sum()
    missing_rt = df_filtered['response_time_ms'].isna().sum()
    
    # Remove rows with missing selected_option or response_time_ms
    df_clean = df_filtered.dropna(subset=['selected_option', 'response_time_ms'])
    
    # Identify optimal choice for 80/20 block
    def get_optimal_choice(row):
        if row['block'] != '80/20':
            return np.nan
        
        # Determine which option has the 0.8 probability
        if row['left_option_prob'] == 0.8:
            return 0  # Left is optimal (0)
        elif row['right_option_prob'] == 0.8:
            return 1  # Right is optimal (1)
        else:
            return np.nan
    
    df_clean.loc[:, 'optimal_choice'] = df_clean.apply(get_optimal_choice, axis=1)
    
    # Mark correct choices in 80/20 block
    df_clean.loc[:, 'chose_optimal'] = (df_clean['selected_option'] == df_clean['optimal_choice'])
    
    # For 50/50 block, track if choice matches first choice in the block
    df_5050 = df_clean[df_clean['block'] == '50/50'].copy()
    
    # Group by participant to find first choice in block
    first_choices = df_5050.groupby('participant_id').first()['selected_option']
    
    # Create a dictionary of participant_id -> first_choice
    first_choice_dict = first_choices.to_dict()
    
    # Function to check if the current choice matches first choice
    def matches_first_choice(row):
        if row['block'] != '50/50':
            return np.nan
        return 1 if row['selected_option'] == first_choice_dict.get(row['participant_id']) else 0
    
    df_clean.loc[:, 'matches_first_choice'] = df_clean.apply(matches_first_choice, axis=1)
    
    # Add a column for "switched from previous trial"
    df_clean.loc[:, 'prev_choice'] = df_clean.groupby(['participant_id', 'block'])['selected_option'].shift(1)
    df_clean.loc[:, 'switched'] = (df_clean['selected_option'] != df_clean['prev_choice']).astype(int)
    
    return df_clean, missing_selected, missing_rt

# Load data
with st.spinner("Loading and preprocessing data..."):
    df, missing_selected, missing_rt = load_and_preprocess_data()

# Display data info in sidebar
st.sidebar.header("Dataset Information")
total_participants = df['participant_id'].nunique()
st.sidebar.metric("Number of Participants", total_participants)

# Preprocessing summary
with st.sidebar.expander("Preprocessing Details"):
    st.write(f"Original rows after excluding practice: {len(df) + missing_selected}")
    st.write(f"Missing data in selected_option: {missing_selected} ({missing_selected/(len(df) + missing_selected)*100:.2f}%)")
    st.write(f"Missing data in response_time_ms: {missing_rt} ({missing_rt/(len(df) + missing_rt)*100:.2f}%)")
    st.write(f"Final rows after preprocessing: {len(df)}")

# Global controls in sidebar
st.sidebar.header("Analysis Controls")

# Trial bin size selection
bin_size = st.sidebar.slider("Trial Bin Size", min_value=1, max_value=20, value=10, 
                            help="Number of trials to group in each bin for analysis")

# Add trial bin to DataFrame based on selected bin size
df['trial_bin'] = ((df['trial_number'] - 1) // bin_size) + 1

# Participant selection
all_participants = sorted(df['participant_id'].unique())
selected_participants = st.sidebar.multiselect(
    "Select Participants for Individual Analysis",
    options=["All Participants"] + list(all_participants),
    default=["All Participants"],
    help="Select specific participants to analyze individually"
)

# If 'All Participants' is selected along with other options, keep only 'All Participants'
if "All Participants" in selected_participants and len(selected_participants) > 1:
    selected_participants = ["All Participants"]

# Create filtered dataframe for selected participants
if "All Participants" in selected_participants:
    df_selected = df.copy()
    label_for_all = "Group Average"
else:
    df_selected = df[df['participant_id'].isin(selected_participants)]
    label_for_all = None

# Main tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Learning Curves", 
    "Response Times", 
    "Exploration/Exploitation", 
    "Score Dynamics",
    "Individual Differences"
])

# ===== 1. Learning Curves Analysis =====
with tab1:
    st.header("Learning Curves Analysis")
    st.markdown("""
    This analysis shows how participants learn to choose the optimal option (in 80/20 block)
    and how consistent their choices are (in 50/50 block) over time.
    """)
    
    # Create figures for learning curves
    fig_learning = make_subplots(rows=1, cols=2, 
                                subplot_titles=("Learning Curve: 80/20 Block", "Choice Stability: 50/50 Block"))
    
    # Function to add data to plots
    def add_learning_data(data, name, color):
        # 80/20 block data
        data_8020 = data[data['block'] == '80/20']
        optimal_by_bin = data_8020.groupby('trial_bin')['chose_optimal'].mean().reset_index()
        
        # Add to plot
        fig_learning.add_trace(
            go.Scatter(
                x=optimal_by_bin['trial_bin'], 
                y=optimal_by_bin['chose_optimal'],
                mode='lines+markers',
                name=f"{name} (80/20)",
                line=dict(color=color),
                legendgroup=name
            ),
            row=1, col=1
        )
        
        # 50/50 block data
        data_5050 = data[data['block'] == '50/50']
        matches_by_bin = data_5050.groupby('trial_bin')['matches_first_choice'].mean().reset_index()
        
        # Add to plot
        fig_learning.add_trace(
            go.Scatter(
                x=matches_by_bin['trial_bin'], 
                y=matches_by_bin['matches_first_choice'],
                mode='lines+markers',
                name=f"{name} (50/50)",
                line=dict(color=color, dash='dash'),
                legendgroup=name
            ),
            row=1, col=2
        )
    
    # Add group average
    add_learning_data(df, label_for_all or "Group Average", 'royalblue')
    
    # Add individual participants if selected
    colors = px.colors.qualitative.Set2
    if "All Participants" not in selected_participants:
        for i, participant in enumerate(selected_participants):
            participant_data = df[df['participant_id'] == participant]
            add_learning_data(participant_data, f"Participant {participant}", colors[i % len(colors)])
    
    # Update figure layout
    fig_learning.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Add horizontal reference line at 0.5
    fig_learning.add_shape(
        type="line", line=dict(dash="dash", color="gray"),
        x0=0, y0=0.5, x1=df['trial_bin'].max(), y1=0.5, row=1, col=1
    )
    fig_learning.add_shape(
        type="line", line=dict(dash="dash", color="gray"),
        x0=0, y0=0.5, x1=df['trial_bin'].max(), y1=0.5, row=1, col=2
    )
    
    fig_learning.update_xaxes(title_text="Trial Bin", row=1, col=1)
    fig_learning.update_xaxes(title_text="Trial Bin", row=1, col=2)
    fig_learning.update_yaxes(title_text="Proportion of Optimal Choices", row=1, col=1)
    fig_learning.update_yaxes(title_text="Proportion Matching First Choice", row=1, col=2)
    fig_learning.update_yaxes(range=[0.4, 1.0], row=1, col=1)
    fig_learning.update_yaxes(range=[0.4, 1.0], row=1, col=2)
    
    st.plotly_chart(fig_learning, use_container_width=True)
    
    # Summary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("80/20 Block Learning Summary")
        df_8020 = df_selected[df_selected['block'] == '80/20']
        
        # First bin vs last bin
        first_bin = df_8020[df_8020['trial_bin'] == 1]['chose_optimal'].mean()
        last_bin = df_8020[df_8020['trial_bin'] == df_8020['trial_bin'].max()]['chose_optimal'].mean()
        
        # Calculate learning slope (linear regression)
        optimal_by_bin = df_8020.groupby('trial_bin')['chose_optimal'].mean().reset_index()
        
        if len(optimal_by_bin) > 1:  # Need at least 2 points for regression
            x = optimal_by_bin['trial_bin'].values
            y = optimal_by_bin['chose_optimal'].values
            slope_8020 = np.polyfit(x, y, 1)[0]
        else:
            slope_8020 = 0
        
        metrics_df = pd.DataFrame({
            'Metric': ['First Bin Optimal Rate', 'Last Bin Optimal Rate', 'Learning Slope'],
            'Value': [f"{first_bin:.2f}", f"{last_bin:.2f}", f"{slope_8020:.4f}"]
        })
        
        st.dataframe(metrics_df, hide_index=True)
    
    with col2:
        st.subheader("50/50 Block Choice Stability")
        df_5050 = df_selected[df_selected['block'] == '50/50']
        
        # First bin vs last bin
        first_bin = df_5050[df_5050['trial_bin'] == 1]['matches_first_choice'].mean()
        last_bin = df_5050[df_5050['trial_bin'] == df_5050['trial_bin'].max()]['matches_first_choice'].mean()
        
        # Calculate stability slope
        stability_by_bin = df_5050.groupby('trial_bin')['matches_first_choice'].mean().reset_index()
        
        if len(stability_by_bin) > 1:  # Need at least 2 points for regression
            x = stability_by_bin['trial_bin'].values
            y = stability_by_bin['matches_first_choice'].values
            slope_5050 = np.polyfit(x, y, 1)[0]
        else:
            slope_5050 = 0
        
        metrics_df = pd.DataFrame({
            'Metric': ['First Bin Stability', 'Last Bin Stability', 'Stability Slope'],
            'Value': [f"{first_bin:.2f}", f"{last_bin:.2f}", f"{slope_5050:.4f}"]
        })
        
        st.dataframe(metrics_df, hide_index=True)

# ===== 2. Response Times Analysis =====
with tab2:
    st.header("Response Times Analysis")
    st.markdown("""
    This analysis examines response times (RT) across different conditions and over time.
    """)
    
    # Overall RT metrics by block
    col1, col2 = st.columns(2)
    
    rt_by_block = df_selected.groupby('block')['response_time_ms'].mean()
    rt_std_by_block = df_selected.groupby('block')['response_time_ms'].std() / np.sqrt(df_selected.groupby('block')['response_time_ms'].count())
    
    with col1:
        st.metric("Average RT in 80/20 Block", f"{rt_by_block.get('80/20', 0):.2f} ms")
    
    with col2:
        st.metric("Average RT in 50/50 Block", f"{rt_by_block.get('50/50', 0):.2f} ms")
    
    # RT Dynamics over time
    st.subheader("Response Time Dynamics")
    
    fig_rt = make_subplots(rows=1, cols=2, 
                          subplot_titles=("RT Dynamics: 80/20 Block", "RT Dynamics: 50/50 Block"))
    
    # Function to add RT data to plots
    def add_rt_data(data, name, color):
        # 80/20 block data
        data_8020 = data[data['block'] == '80/20']
        rt_by_bin = data_8020.groupby('trial_bin')['response_time_ms'].mean().reset_index()
        
        fig_rt.add_trace(
            go.Scatter(
                x=rt_by_bin['trial_bin'], 
                y=rt_by_bin['response_time_ms'],
                mode='lines+markers',
                name=f"{name} (80/20)",
                line=dict(color=color),
                legendgroup=name
            ),
            row=1, col=1
        )
        
        # 50/50 block data
        data_5050 = data[data['block'] == '50/50']
        rt_by_bin = data_5050.groupby('trial_bin')['response_time_ms'].mean().reset_index()
        
        fig_rt.add_trace(
            go.Scatter(
                x=rt_by_bin['trial_bin'], 
                y=rt_by_bin['response_time_ms'],
                mode='lines+markers',
                name=f"{name} (50/50)",
                line=dict(color=color, dash='dash'),
                legendgroup=name
            ),
            row=1, col=2
        )
    
    # Add group average
    add_rt_data(df_selected, label_for_all or "Group Average", 'royalblue')
    
    # Add individual participants if selected
    if "All Participants" not in selected_participants:
        for i, participant in enumerate(selected_participants):
            participant_data = df[df['participant_id'] == participant]
            add_rt_data(participant_data, f"Participant {participant}", colors[i % len(colors)])
    
    fig_rt.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig_rt.update_xaxes(title_text="Trial Bin", row=1, col=1)
    fig_rt.update_xaxes(title_text="Trial Bin", row=1, col=2)
    fig_rt.update_yaxes(title_text="Response Time (ms)", row=1, col=1)
    fig_rt.update_yaxes(title_text="Response Time (ms)", row=1, col=2)
    
    st.plotly_chart(fig_rt, use_container_width=True)
    
    # RT analysis by choice type and previous feedback
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RT by Choice Type (80/20 Block)")
        
        df_8020 = df_selected[df_selected['block'] == '80/20']
        
        if not df_8020.empty:
            rt_by_choice = df_8020.groupby('chose_optimal')['response_time_ms'].mean().reset_index()
            rt_by_choice['Choice Type'] = rt_by_choice['chose_optimal'].map({True: 'Optimal', False: 'Suboptimal'})
            
            fig_choice = px.bar(
                rt_by_choice, 
                x='Choice Type', 
                y='response_time_ms',
                color='Choice Type',
                color_discrete_map={'Optimal': 'green', 'Suboptimal': 'red'},
                title="RT by Choice Type",
                labels={'response_time_ms': 'Response Time (ms)'}
            )
            
            st.plotly_chart(fig_choice, use_container_width=True)
            
            # Print values
            optimal_rt = rt_by_choice[rt_by_choice['chose_optimal'] == True]['response_time_ms'].values[0] if True in rt_by_choice['chose_optimal'].values else 0
            suboptimal_rt = rt_by_choice[rt_by_choice['chose_optimal'] == False]['response_time_ms'].values[0] if False in rt_by_choice['chose_optimal'].values else 0
            
            st.markdown(f"""
            - Optimal choice RT: **{optimal_rt:.2f}** ms
            - Suboptimal choice RT: **{suboptimal_rt:.2f}** ms
            - Difference: **{optimal_rt - suboptimal_rt:.2f}** ms
            """)
        else:
            st.info("No data available for the selected participants in the 80/20 block.")
    
    with col2:
        st.subheader("RT by Previous Feedback")
        
        # Add previous trial feedback
        df_selected_copy = df_selected.copy()
        df_selected_copy.loc[:, 'prev_feedback'] = df_selected_copy.groupby(['participant_id', 'block'])['feedback'].shift(1)
        
        # Filter out trials without previous feedback
        df_with_prev = df_selected_copy.dropna(subset=['prev_feedback'])
        
        if not df_with_prev.empty:
            rt_by_feedback = df_with_prev.groupby('prev_feedback')['response_time_ms'].mean().reset_index()
            
            fig_feedback = px.bar(
                rt_by_feedback, 
                x='prev_feedback', 
                y='response_time_ms',
                color='prev_feedback',
                color_discrete_map={'positive': 'green', 'negative': 'red'},
                title="RT by Previous Feedback",
                labels={'response_time_ms': 'Response Time (ms)', 'prev_feedback': 'Previous Feedback'}
            )
            
            st.plotly_chart(fig_feedback, use_container_width=True)
            
            # Print values
            positive_rt = rt_by_feedback[rt_by_feedback['prev_feedback'] == 'positive']['response_time_ms'].values[0] if 'positive' in rt_by_feedback['prev_feedback'].values else 0
            negative_rt = rt_by_feedback[rt_by_feedback['prev_feedback'] == 'negative']['response_time_ms'].values[0] if 'negative' in rt_by_feedback['prev_feedback'].values else 0
            
            st.markdown(f"""
            - After positive feedback: **{positive_rt:.2f}** ms
            - After negative feedback: **{negative_rt:.2f}** ms
            - Difference: **{positive_rt - negative_rt:.2f}** ms
            """)
        else:
            st.info("No data available for RT analysis by previous feedback.")

# ===== 3. Exploration/Exploitation Analysis =====
with tab3:
    st.header("Exploration vs. Exploitation Analysis")
    st.markdown("""
    This analysis examines how participants balance exploration (trying different options) 
    versus exploitation (sticking with a known option).
    """)
    
    # Overall switch rate by block
    switch_rate_by_block = df_selected.groupby('block')['switched'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Switch Rate in 80/20 Block", f"{switch_rate_by_block.get('80/20', 0):.2f}")
    
    with col2:
        st.metric("Average Switch Rate in 50/50 Block", f"{switch_rate_by_block.get('50/50', 0):.2f}")
    
    # Switch Rate Dynamics over time
    st.subheader("Exploration Dynamics (Switch Rate Over Time)")
    
    fig_switch = make_subplots(rows=1, cols=2, 
                              subplot_titles=("Exploration Dynamics: 80/20 Block", 
                                             "Exploration Dynamics: 50/50 Block"))
    
    # Function to add switch rate data to plots
    def add_switch_data(data, name, color):
        # 80/20 block data
        data_8020 = data[data['block'] == '80/20']
        switch_by_bin = data_8020.groupby('trial_bin')['switched'].mean().reset_index()
        
        fig_switch.add_trace(
            go.Scatter(
                x=switch_by_bin['trial_bin'], 
                y=switch_by_bin['switched'],
                mode='lines+markers',
                name=f"{name} (80/20)",
                line=dict(color=color),
                legendgroup=name
            ),
            row=1, col=1
        )
        
        # 50/50 block data
        data_5050 = data[data['block'] == '50/50']
        switch_by_bin = data_5050.groupby('trial_bin')['switched'].mean().reset_index()
        
        fig_switch.add_trace(
            go.Scatter(
                x=switch_by_bin['trial_bin'], 
                y=switch_by_bin['switched'],
                mode='lines+markers',
                name=f"{name} (50/50)",
                line=dict(color=color, dash='dash'),
                legendgroup=name
            ),
            row=1, col=2
        )
    
    # Add group average
    add_switch_data(df_selected, label_for_all or "Group Average", 'royalblue')
    
    # Add individual participants if selected
    if "All Participants" not in selected_participants:
        for i, participant in enumerate(selected_participants):
            participant_data = df[df['participant_id'] == participant]
            add_switch_data(participant_data, f"Participant {participant}", colors[i % len(colors)])
    
    fig_switch.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig_switch.update_xaxes(title_text="Trial Bin", row=1, col=1)
    fig_switch.update_xaxes(title_text="Trial Bin", row=1, col=2)
    fig_switch.update_yaxes(title_text="Switch Rate", row=1, col=1)
    fig_switch.update_yaxes(title_text="Switch Rate", row=1, col=2)
    fig_switch.update_yaxes(range=[0, 0.6], row=1, col=1)
    fig_switch.update_yaxes(range=[0, 0.6], row=1, col=2)
    
    st.plotly_chart(fig_switch, use_container_width=True)
    
    # Calculate choice entropy as another measure of exploration
    st.subheader("Choice Entropy Analysis")
    st.markdown("""
    Shannon entropy measures the unpredictability of choices. Higher entropy indicates more exploration.
    """)
    
    def calculate_entropy(choices):
        """Calculate Shannon entropy of choices"""
        unique, counts = np.unique(choices, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))
    
    # Calculate entropy for each participant, block, and trial bin
    if not df_selected.empty:
        entropy_data = []
        
        for (participant, block, bin_num), group in df_selected.groupby(['participant_id', 'block', 'trial_bin']):
            choices = group['selected_option'].values
            if len(choices) > 1:  # Need at least 2 choices to calculate meaningful entropy
                entropy = calculate_entropy(choices)
                entropy_data.append({
                    'participant_id': participant,
                    'block': block,
                    'trial_bin': bin_num,
                    'entropy': entropy
                })
        
        if entropy_data:
            entropy_df = pd.DataFrame(entropy_data)
            
            # Calculate mean entropy by block and trial bin
            if "All Participants" in selected_participants:
                entropy_by_block_bin = entropy_df.groupby(['block', 'trial_bin'])['entropy'].mean().reset_index()
            else:
                entropy_by_block_bin = entropy_df
            
            # Plot entropy over time
            fig_entropy = px.line(
                entropy_by_block_bin, 
                x='trial_bin', 
                y='entropy', 
                color='block',
                color_discrete_map={'80/20': 'blue', '50/50': 'orange'},
                markers=True,
                title="Choice Entropy Over Time",
                labels={'trial_bin': 'Trial Bin', 'entropy': 'Shannon Entropy', 'block': 'Block'}
            )
            
            fig_entropy.update_layout(height=400)
            fig_entropy.update_yaxes(range=[0, 1])
            
            st.plotly_chart(fig_entropy, use_container_width=True)
            
            # Display average entropy by block
            entropy_by_block = entropy_df.groupby('block')['entropy'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Entropy in 80/20 Block", f"{entropy_by_block.get('80/20', 0):.2f}")
            
            with col2:
                st.metric("Average Entropy in 50/50 Block", f"{entropy_by_block.get('50/50', 0):.2f}")
            
            st.info("Higher entropy (closer to 1) indicates more exploration/randomness in choices.")
        else:
            st.info("Not enough data to calculate entropy for the selected participants or trial bins.")
    else:
        st.info("No data available for entropy analysis.")

# ===== 4. Score Dynamics Analysis =====
with tab4:
    st.header("Score Dynamics Analysis")
    st.markdown("""
    This analysis examines how participants accumulate points over time in different blocks.
    """)
    
    # Function to get score data for plotting
    def get_score_data(data, block_type):
        block_data = data[data['block'] == block_type]
        
        if "All Participants" in selected_participants:
            # Group average
            avg_scores = block_data.groupby('trial_number')['total_points'].mean().reset_index()
            return avg_scores
        else:
            # Individual participants
            return block_data[['participant_id', 'trial_number', 'total_points']]
    
    # Get data for both blocks
    data_8020 = get_score_data(df_selected, '80/20')
    data_5050 = get_score_data(df_selected, '50/50')
    
    # Create plot
    fig_score = go.Figure()
    
    # Add traces based on selection
    if "All Participants" in selected_participants:
        # Add average traces
        fig_score.add_trace(
            go.Scatter(
                x=data_8020['trial_number'],
                y=data_8020['total_points'],
                mode='lines',
                name='80/20 Block (Avg)',
                line=dict(color='blue')
            )
        )
        
        fig_score.add_trace(
            go.Scatter(
                x=data_5050['trial_number'],
                y=data_5050['total_points'],
                mode='lines',
                name='50/50 Block (Avg)',
                line=dict(color='orange')
            )
        )
    else:
        # Add individual participant traces
        for participant in selected_participants:
            part_data_8020 = data_8020[data_8020['participant_id'] == participant]
            part_data_5050 = data_5050[data_5050['participant_id'] == participant]
            
            fig_score.add_trace(
                go.Scatter(
                    x=part_data_8020['trial_number'],
                    y=part_data_8020['total_points'],
                    mode='lines',
                    name=f'Participant {participant} (80/20)',
                    line=dict(color='blue')
                )
            )
            
            fig_score.add_trace(
                go.Scatter(
                    x=part_data_5050['trial_number'],
                    y=part_data_5050['total_points'],
                    mode='lines',
                    name=f'Participant {participant} (50/50)',
                    line=dict(color='orange')
                )
            )
    
    fig_score.update_layout(
        title="Cumulative Score Over Trials",
        xaxis_title="Trial Number",
        yaxis_title="Cumulative Score (points)",
        height=500
    )
    
    st.plotly_chart(fig_score, use_container_width=True)
    
    # Final scores comparison
    col1, col2 = st.columns(2)
    
    # Calculate final scores for selected participants
    final_score_8020 = df_selected[df_selected['block'] == '80/20'].groupby('participant_id')['total_points'].max().mean()
    final_score_5050 = df_selected[df_selected['block'] == '50/50'].groupby('participant_id')['total_points'].max().mean()
    
    with col1:
        st.metric("Average Final Score (80/20 Block)", f"{final_score_8020:.2f}")
    
    with col2:
        st.metric("Average Final Score (50/50 Block)", f"{final_score_5050:.2f}")
    
    # Score accumulation rate analysis
    st.subheader("Score Accumulation Rate")
    
    # Calculate linear regression for each block's scoring rate
    def calc_score_slope(data):
        if len(data) > 1:
            x = data['trial_number'].values
            y = data['total_points'].values
            slope = np.polyfit(x, y, 1)[0]
            return slope
        else:
            return 0
    
    if "All Participants" in selected_participants:
        slope_8020 = calc_score_slope(data_8020)
        slope_5050 = calc_score_slope(data_5050)
        
        score_slopes = pd.DataFrame({
            'Block': ['80/20 Block', '50/50 Block'],
            'Score Accumulation Rate (points/trial)': [slope_8020, slope_5050]
        })
        
        st.dataframe(score_slopes, hide_index=True)
    else:
        # Calculate for each selected participant
        score_slopes = []
        
        for participant in selected_participants:
            part_data_8020 = data_8020[data_8020['participant_id'] == participant]
            part_data_5050 = data_5050[data_5050['participant_id'] == participant]
            
            slope_8020 = calc_score_slope(part_data_8020)
            slope_5050 = calc_score_slope(part_data_5050)
            
            score_slopes.append({
                'Participant': participant,
                '80/20 Block Rate': slope_8020,
                '50/50 Block Rate': slope_5050
            })
        
        score_slopes_df = pd.DataFrame(score_slopes)
        st.dataframe(score_slopes_df, hide_index=True)

# ===== 5. Individual Differences Analysis =====
with tab5:
    st.header("Individual Differences Analysis")
    st.markdown("""
    This analysis examines how participants differ in their learning, exploration strategies,
    and performance in the 80/20 block.
    """)
    
    # Calculate metrics for each participant in 80/20 block
    if not df.empty:
        df_8020 = df[df['block'] == '80/20']
        
        if not df_8020.empty:
            participant_metrics = []
            
            for participant_id, group in df_8020.groupby('participant_id'):
                # Overall proportion of optimal choices
                optimal_proportion = group['chose_optimal'].mean()
                
                # Average RT
                avg_rt = group['response_time_ms'].mean()
                
                # Average switch rate (focusing on later trials, after learning)
                later_trials = group[group['trial_number'] > 30]  # Second half of trials
                switch_rate = later_trials['switched'].mean() if len(later_trials) > 0 else np.nan
                
                # Final cumulative score
                final_score = group['total_points'].max()
                
                participant_metrics.append({
                    'participant_id': participant_id,
                    'optimal_proportion': optimal_proportion,
                    'avg_rt': avg_rt,
                    'switch_rate': switch_rate,
                    'final_score': final_score
                })
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(participant_metrics)
            
            # Display metrics table
            st.subheader("Participant Metrics in 80/20 Block")
            
            # Format the dataframe for display
            display_df = metrics_df.copy()
            display_df.columns = [
                'Participant ID', 
                'Optimal Choice %', 
                'Avg RT (ms)',
                'Switch Rate (later trials)',
                'Final Score'
            ]
            
            # Format percentages
            display_df['Optimal Choice %'] = display_df['Optimal Choice %'].map(lambda x: f"{x:.1%}")
            display_df['Switch Rate (later trials)'] = display_df['Switch Rate (later trials)'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
            
            # Format as sortable dataframe
            st.dataframe(display_df, hide_index=True)
            
            # Scatter plot matrix
            st.subheader("Relationships Between Metrics")
            st.markdown("Select metrics to compare:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_metric = st.selectbox(
                    "X-axis metric",
                    options=[
                        'optimal_proportion', 
                        'avg_rt', 
                        'switch_rate', 
                        'final_score'
                    ],
                    format_func=lambda x: {
                        'optimal_proportion': 'Optimal Choice %',
                        'avg_rt': 'Average RT (ms)',
                        'switch_rate': 'Switch Rate',
                        'final_score': 'Final Score'
                    }[x],
                    index=0
                )
            
            with col2:
                y_metric = st.selectbox(
                    "Y-axis metric",
                    options=[
                        'optimal_proportion', 
                        'avg_rt', 
                        'switch_rate', 
                        'final_score'
                    ],
                    format_func=lambda x: {
                        'optimal_proportion': 'Optimal Choice %',
                        'avg_rt': 'Average RT (ms)',
                        'switch_rate': 'Switch Rate',
                        'final_score': 'Final Score'
                    }[x],
                    index=3
                )
            
            # Create scatter plot
            fig_scatter = px.scatter(
                metrics_df,
                x=x_metric,
                y=y_metric,
                hover_data=['participant_id'],
                labels={
                    'optimal_proportion': 'Optimal Choice %',
                    'avg_rt': 'Average RT (ms)',
                    'switch_rate': 'Switch Rate',
                    'final_score': 'Final Score',
                    'participant_id': 'Participant ID'
                },
                title=f"Relationship between {x_metric.replace('_', ' ').title()} and {y_metric.replace('_', ' ').title()}"
            )
            
            # Highlight selected participants if applicable
            if "All Participants" not in selected_participants:
                selected_metrics = metrics_df[metrics_df['participant_id'].isin(selected_participants)]
                
                if not selected_metrics.empty:
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=selected_metrics[x_metric],
                            y=selected_metrics[y_metric],
                            mode='markers',
                            marker=dict(color='red', size=12, line=dict(width=2, color='DarkSlateGrey')),
                            name='Selected Participants',
                            text=selected_metrics['participant_id']
                        )
                    )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Try to perform clustering if scikit-learn is available
            st.subheader("Participant Clustering")
            
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # Allow user to select number of clusters
                n_clusters = st.slider("Number of clusters", min_value=2, max_value=5, value=3)
                
                # Prepare data for clustering
                cluster_cols = ['optimal_proportion', 'avg_rt', 'switch_rate', 'final_score']
                
                # Handle missing values
                cluster_data = metrics_df[cluster_cols].copy()
                cluster_data = cluster_data.fillna(cluster_data.mean())
                
                # Scale features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Add cluster assignment to metrics
                metrics_df['cluster'] = clusters
                
                # Add to scatter plot
                fig_cluster = px.scatter(
                    metrics_df,
                    x=x_metric,
                    y=y_metric,
                    color='cluster',
                    hover_data=['participant_id'],
                    labels={
                        'optimal_proportion': 'Optimal Choice %',
                        'avg_rt': 'Average RT (ms)',
                        'switch_rate': 'Switch Rate',
                        'final_score': 'Final Score',
                        'participant_id': 'Participant ID',
                        'cluster': 'Cluster'
                    },
                    title=f"Participant Clusters Based on Learning Metrics"
                )
                
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Display cluster statistics
                st.subheader("Cluster Characteristics")
                
                cluster_stats = metrics_df.groupby('cluster')[cluster_cols].mean()
                
                # Format for display
                display_stats = cluster_stats.copy()
                display_stats.columns = [
                    'Optimal Choice %', 
                    'Avg RT (ms)',
                    'Switch Rate',
                    'Final Score'
                ]
                
                # Format percentages
                display_stats['Optimal Choice %'] = display_stats['Optimal Choice %'].map(lambda x: f"{x:.1%}")
                display_stats['Switch Rate'] = display_stats['Switch Rate'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
                
                st.dataframe(display_stats)
                
                # Describe cluster characteristics
                for cluster in sorted(metrics_df['cluster'].unique()):
                    cluster_size = sum(metrics_df['cluster'] == cluster)
                    opt_rate = cluster_stats.loc[cluster, 'optimal_proportion']
                    switch_rate = cluster_stats.loc[cluster, 'switch_rate']
                    final_score = cluster_stats.loc[cluster, 'final_score']
                    
                    st.markdown(f"""
                    **Cluster {cluster}** ({cluster_size} participants):
                    - Optimal choice rate: {opt_rate:.1%}
                    - Switch rate (late trials): {switch_rate:.1%}
                    - Final score: {final_score:.1f}
                    """)
                
            except ImportError:
                st.info("""
                Clustering requires scikit-learn, which is not available in the current environment.
                To enable clustering, install scikit-learn using `pip install scikit-learn`.
                """)
                
                # Simple alternative based on quantiles of optimal choice rate
                st.subheader("Participant Groups")
                st.markdown("""
                Without scikit-learn, we can group participants based on their optimal choice rate.
                """)
                
                # Define the columns for statistics (same as would be used for clustering)
                analysis_cols = ['optimal_proportion', 'avg_rt', 'switch_rate', 'final_score']
                
                # Create groups based on quantiles
                metrics_df['optimal_group'] = pd.qcut(
                    metrics_df['optimal_proportion'], 
                    q=3, 
                    labels=['Low', 'Medium', 'High']
                )
                
                # Display groups
                fig_groups = px.scatter(
                    metrics_df,
                    x=x_metric,
                    y=y_metric,
                    color='optimal_group',
                    hover_data=['participant_id'],
                    labels={
                        'optimal_proportion': 'Optimal Choice %',
                        'avg_rt': 'Average RT (ms)',
                        'switch_rate': 'Switch Rate',
                        'final_score': 'Final Score',
                        'participant_id': 'Participant ID',
                        'optimal_group': 'Optimal Choice Group'
                    },
                    title=f"Participant Groups Based on Optimal Choice Rate"
                )
                
                st.plotly_chart(fig_groups, use_container_width=True)
                
                # Display group statistics
                st.subheader("Group Characteristics")
                
                group_stats = metrics_df.groupby('optimal_group')[analysis_cols].mean()
                
                # Format for display
                display_group_stats = group_stats.copy()
                display_group_stats.columns = [
                    'Optimal Choice %', 
                    'Avg RT (ms)',
                    'Switch Rate',
                    'Final Score'
                ]
                
                # Format percentages
                display_group_stats['Optimal Choice %'] = display_group_stats['Optimal Choice %'].map(lambda x: f"{x:.1%}")
                display_group_stats['Switch Rate'] = display_group_stats['Switch Rate'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else 'N/A')
                
                st.dataframe(display_group_stats)
        else:
            st.info("No data available for the 80/20 block.")
    else:
        st.info("No data available for individual differences analysis.")

# Add some resources and explanations at the bottom
st.markdown("---")
st.markdown("""
### About Probabilistic Reward Learning (PRL)

In PRL tasks, participants learn to make choices between options with different reward probabilities:
- In the 80/20 block, one option has an 80% chance of reward, while the other has only 20%.
- In the 50/50 block, both options have equal (50%) chance of reward.

Participants typically learn to choose the more rewarding option in the 80/20 block,
while they may show more varied behavior in the 50/50 block where no strategy is optimal.

### Metrics Explained

- **Optimal Choice Rate**: The proportion of trials where the participant chose the option with higher reward probability.
- **Switch Rate**: The proportion of trials where the participant chose a different option than on the previous trial.
- **Response Time (RT)**: The time taken to make a choice on each trial, measured in milliseconds.
- **Entropy**: A measure of choice randomness or exploration, with higher values indicating more exploration.

### Analysis Tips

- Use the sidebar to select specific participants to analyze.
- Adjust the trial bin size to control the granularity of time-based analyses.
- Compare blocks to understand how behavior changes with different reward probabilities.
""")