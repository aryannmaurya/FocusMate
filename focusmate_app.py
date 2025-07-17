import streamlit as st
import requests
import datetime as dt
import pandas as pd
import plotly.express as px
import os
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import calendar
import json
# from plotly_calplot import calplot as cp

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="FocusMate - AI Coach",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🎯"
)

# ---------------------------
# File Paths & Constants
# ---------------------------
DIARY_FILE = "diary_log.csv"
GOALS_FILE = "goals.csv"

# ---------------------------
# Helper Functions for Data Handling
# ---------------------------

def load_data(file_path, columns):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading {file_path}: {e}")
        return pd.DataFrame(columns=columns)

def load_goals():
    """Loads goals from a CSV file."""
    if not os.path.exists(GOALS_FILE) or os.path.getsize(GOALS_FILE) == 0:
        return pd.DataFrame(columns=['Goal', 'Status', 'DateAdded'])
    try:
        df = pd.read_csv(GOALS_FILE)
        df['DateAdded'] = pd.to_datetime(df['DateAdded'])
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading goals: {e}")
        return pd.DataFrame(columns=['Goal', 'Status', 'DateAdded'])

def save_goals(df_goals):
    """Saves the goals DataFrame to CSV."""
    df_goals.to_csv(GOALS_FILE, index=False)

# ---------------------------
# Main App Structure
# ---------------------------
st.markdown(
    "<h1 style='text-align: center; color: #6C63FF;'>🎯 FocusMate</h1>"
    "<h4 style='text-align: center; color: gray;'>Your AI-Powered Daily Reflection & Goal Tracker</h4><hr>",
    unsafe_allow_html=True
)

# Load data once
diary_df = load_data(DIARY_FILE, ['Date', 'Diary', 'Mood', 'Sentiment', 'AI Feedback', 'Keywords', 'Linked Goals'])
goals_df = load_goals()

# --- Create UI Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["✍️ New Entry", "📊 Analytics Dashboard", "🎯 Goals", "📜 History"])

# ---------------------------
# TAB 1: New Entry
# ---------------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🖊️ Today's Entry")
        diary_input = st.text_area("What's on your mind? Document your wins, challenges, and thoughts.", height=280, label_visibility="collapsed")

    with col2:
        st.markdown("### 🙂 Mood")
        mood_options = ["😄 Happy", "🥳 Excited", "😐 Neutral", "😴 Tired", "😢 Sad", "🤯 Stressed"]
        mood = st.radio("Select your mood:", mood_options, label_visibility="collapsed")

        st.markdown("### 🔗 Link to Goals")
        active_goals = goals_df[goals_df['Status'] == 'Active']['Goal'].tolist()
        if not active_goals:
            st.info("No active goals. Add some in the 'Goals' tab!")
            linked_goals = []
        else:
            linked_goals = st.multiselect("Select goals this entry relates to:", active_goals, label_visibility="collapsed")

    submit = st.button("🚀 Analyze & Save Entry", use_container_width=True, type="primary")

    if submit and diary_input.strip():
        with st.spinner("🤖 Your AI coach is reflecting..."):
            date = dt.datetime.now()
            sentiment = TextBlob(diary_input).sentiment.polarity

            prompt = f"""
            You are 'FocusMate', a positive and insightful AI progress coach. Your goal is to help me reflect and grow. Analyze my diary entry.

            Diary:
            \"\"\"{diary_input}\"\"\"

            Respond in this exact YAML-like format, with no preamble:
            Win: [Identify one positive achievement or success, even a small one.]
            Challenge: [Identify the main struggle or area for improvement in a constructive way.]
            Insight: [Provide a brief, actionable insight or a re-framing perspective based on the entry.]
            Question: [Ask one open-ended question to encourage deeper reflection.]
            Keywords: [A comma-separated list of 3-5 key topics.]
            """
            try:
                response = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.1:8b", "prompt": prompt}, stream=True)
                response.raise_for_status()
                full_reply = "".join(json.loads(line.decode("utf-8")).get("response", "") for line in response.iter_lines() if line)

                log_data = {
                    "Date": date.strftime("%Y-%m-%d %H:%M:%S"), "Diary": diary_input, "Mood": mood,
                    "Sentiment": sentiment, "AI Feedback": full_reply,
                    "Keywords": ','.join(line.split("Keywords:")[1].strip() for line in full_reply.split('\n') if "Keywords:" in line),
                    "Linked Goals": ", ".join(linked_goals)
                }
                new_df = pd.DataFrame([log_data])
                diary_df = pd.concat([diary_df, new_df], ignore_index=True)
                diary_df.to_csv(DIARY_FILE, index=False)
                st.success("✅ Your entry has been saved and analyzed!")

                with st.container(border=True):
                    st.markdown("### 🧠 Your AI Coach's Reflection")
                    for line in full_reply.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            st.markdown(f"**{key.strip()}:** {value.strip()}")

            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Connection Error: Could not connect to the LLaMA server. Is it running? Error: {e}")
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred: {e}")

    elif submit:
        st.warning("Please write something before submitting.")

# ---------------------------
# TAB 2: Analytics Dashboard
# ---------------------------
with tab2:
    st.header("Analytics Dashboard")
    if diary_df.empty:
        st.info("📭 No data yet. Your analytics will appear here once you've made some entries.")
    else:
        st.markdown("### 🗓️ Yearly Consistency & Mood")
        diary_df['DateOnly'] = diary_df['Date'].dt.date
        day_counts = diary_df.groupby('DateOnly').size().reset_index(name='counts')
        day_counts['DateOnly'] = pd.to_datetime(day_counts['DateOnly'])
        day_counts = day_counts.set_index('DateOnly')
        
        # Create a colormap for the heatmap
        mood_color_map = {'😄 Happy': '#2ECC71', '🥳 Excited': '#3498DB', '😐 Neutral': '#95A5A6', '😴 Tired': '#F39C12', '😢 Sad': '#E74C3C', '🤯 Stressed': '#8E44AD'}
        daily_mood = diary_df.groupby('DateOnly')['Mood'].first().map(mood_color_map).reindex(day_counts.index).fillna('#ecf0f1')

        # Create a matplotlib calendar plot instead of using plotly_calplot
        def plot_calendar_heatmap(data, year=None):
            # If year is not specified, use the current year
            if year is None:
                year = dt.datetime.now().year
                
            # Filter data for the specified year
            year_data = data[data.index.year == year]
            
            # Create a figure with a dark background if dark_theme is True
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('#0E1117')  # Dark background matching Streamlit's dark theme
            ax.set_facecolor('#0E1117')
            
            # Create a colormap
            cmap = plt.cm.viridis
            
            # Get the min and max values for normalization
            vmin = 0
            vmax = data['counts'].max() if not data.empty else 1
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
            # Calculate positions for each day
            cal = calendar.Calendar()
            
            # Track the current position
            x_pos = 0
            y_pos = 0
            
            # Define the size of each cell and the gap
            cell_size = 0.9
            gap = 0.3
            
            # Store month positions for labels
            month_positions = []
            
            # Plot each month
            for month in range(1, 13):
                # Get all days in the month
                month_days = cal.monthdayscalendar(year, month)
                
                # Store the center position of this month for the label
                month_positions.append((x_pos + (len(month_days[0]) * (cell_size + gap)) / 2, y_pos - 0.5))
                
                # Plot each day in the month
                for week_idx, week in enumerate(month_days):
                    for day_idx, day in enumerate(week):
                        if day != 0:  # Calendar returns 0 for days not in the month
                            date = pd.Timestamp(year=year, month=month, day=day)
                            # Get the count for this day, default to 0 if not found
                            count = year_data.loc[date]['counts'] if date in year_data.index else 0
                            
                            # Calculate the color based on the count
                            color = cmap(norm(count))
                            
                            # Plot a rectangle for this day
                            rect = plt.Rectangle(
                                (x_pos + day_idx * (cell_size + gap), y_pos - week_idx * (cell_size + gap)),
                                cell_size, cell_size,
                                facecolor=color,
                                edgecolor='none',
                                alpha=0.8 if count > 0 else 0.3  # Lower alpha for days with no entries
                            )
                            ax.add_patch(rect)
                            
                            # Add the day number as text
                            ax.text(
                                x_pos + day_idx * (cell_size + gap) + cell_size/2,
                                y_pos - week_idx * (cell_size + gap) + cell_size/2,
                                str(day),
                                ha='center',
                                va='center',
                                color='white' if count > 0 else '#888888',
                                fontsize=8
                            )
                
                # Move to the next month position
                x_pos += (7 + gap) * (cell_size + gap)
                
                # If we've plotted 4 months, move to the next row
                if month % 4 == 0:
                    y_pos -= 6 * (cell_size + gap)  # Move down for the next row
                    x_pos = 0  # Reset x position
            
            # Add month labels
            for idx, (x, y) in enumerate(month_positions):
                month_name = calendar.month_name[idx + 1]
                ax.text(x, y, month_name, ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            
            # Add a color bar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
            cbar.set_label('Number of Entries', color='white')
            cbar.ax.tick_params(colors='white')
            
            # Set the limits and remove the axes
            ax.set_xlim(-1, 30)
            ax.set_ylim(-25, 2)
            ax.axis('off')
            
            # Add a title
            ax.set_title(f'Daily Entries for {year}', color='white', fontsize=14, pad=20)
            
            plt.tight_layout()
            return fig
        
        # Create the calendar plot for the current year
        current_year = dt.datetime.now().year
        fig = plot_calendar_heatmap(day_counts, year=current_year)
        
        # Display the matplotlib figure in Streamlit
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        one_week_ago = dt.datetime.now() - dt.timedelta(days=7)
        week_data = diary_df[diary_df['Date'] > one_week_ago].copy()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("😌 Mood Breakdown (Last 7 Days)")
            if not week_data.empty:
                mood_counts = week_data['Mood'].value_counts()
                fig_mood = px.pie(values=mood_counts.values, names=mood_counts.index, hole=0.4, color_discrete_map=mood_color_map)
                st.plotly_chart(fig_mood, use_container_width=True)
            else:
                st.info("No entries in the last 7 days.")
        
        with col2:
            st.subheader("📈 Sentiment Trend (Last 7 Days)")
            if not week_data.empty and 'Sentiment' in week_data.columns and not week_data['Sentiment'].isnull().all():
                week_data.sort_values('Date', inplace=True)
                fig_sentiment = px.line(week_data, x='Date', y='Sentiment', markers=True)
                fig_sentiment.update_yaxes(range=[-1.1, 1.1])
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.info("No sentiment data to display for the last 7 days.")

        st.markdown("---")
        st.subheader("🎯 Goal Progress (All Time)")
        if not goals_df[goals_df['Status'] == 'Active'].empty:
            goal_progress = {}
            for goal in active_goals:
                count = diary_df['Linked Goals'].str.contains(goal, na=False).sum()
                goal_progress[goal] = count
            
            progress_df = pd.DataFrame(list(goal_progress.items()), columns=['Goal', 'Entries'])
            fig_goals = px.bar(progress_df, x='Entries', y='Goal', orientation='h', title="Number of Diary Entries per Goal")
            st.plotly_chart(fig_goals, use_container_width=True)
        else:
            st.info("No active goals to track progress against.")

# ---------------------------
# TAB 3: Goals
# ---------------------------
with tab3:
    st.header("🎯 Your Goals")
    st.markdown("Set, track, and accomplish your objectives.")

    with st.form("new_goal_form", clear_on_submit=True):
        new_goal = st.text_input("Enter a new goal:")
        submitted = st.form_submit_button("Add Goal")
        if submitted and new_goal:
            new_goal_data = {'Goal': new_goal, 'Status': 'Active', 'DateAdded': dt.datetime.now().strftime("%Y-%m-%d")}
            goals_df = pd.concat([goals_df, pd.DataFrame([new_goal_data])], ignore_index=True)
            save_goals(goals_df)
            st.success(f"Goal '{new_goal}' added!")
    
    st.markdown("---")
    
    active_goals_df = goals_df[goals_df['Status'] == 'Active']
    completed_goals_df = goals_df[goals_df['Status'] == 'Completed']

    if not active_goals_df.empty:
        st.subheader("🚀 Active Goals")
        for index, row in active_goals_df.iterrows():
            col1, col2, col3 = st.columns([6,1,1])
            with col1:
                st.markdown(f"**{row['Goal']}**")
                st.caption(f"Added on: {pd.to_datetime(row['DateAdded']).strftime('%B %d, %Y')}")
            with col2:
                if st.button("✅", key=f"complete_{index}", help="Mark as complete"):
                    goals_df.at[index, 'Status'] = 'Completed'
                    save_goals(goals_df)
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{index}", help="Delete goal"):
                    goals_df.drop(index, inplace=True)
                    save_goals(goals_df)
                    st.rerun()
    else:
        st.info("You have no active goals. Time to set some!")

    if not completed_goals_df.empty:
        with st.expander("🏆 View Completed Goals"):
            for index, row in completed_goals_df.iterrows():
                st.markdown(f"- ~{row['Goal']}~")

# ---------------------------
# TAB 4: History
# ---------------------------
with tab4:
    st.header("📜 Entry History")
    if diary_df.empty:
        st.info("📭 Your diary is empty. All your past entries will be stored here.")
    else:
        df_sorted = diary_df.sort_values(by="Date", ascending=False)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            search_query = st.text_input("🔍 Search entries by keyword")
        with col2:
            mood_filter = st.multiselect("Filter by mood:", options=df_sorted['Mood'].unique())

        filtered_df = df_sorted
        if search_query:
            filtered_df = filtered_df[filtered_df['Diary'].str.contains(search_query, case=False, na=False)]
        if mood_filter:
            filtered_df = filtered_df[filtered_df['Mood'].isin(mood_filter)]

        st.write(f"Showing {len(filtered_df)} of {len(df_sorted)} entries.")
        st.markdown("---")

        for index, row in filtered_df.iterrows():
            with st.expander(f"**{row['Date'].strftime('%B %d, %Y')}** | Mood: {row['Mood']}"):
                st.markdown(f"**Diary Entry:**")
                st.write(row['Diary'])
                if pd.notna(row['AI Feedback']) and row['AI Feedback']:
                    st.markdown("**🧠 AI Coach's Reflection:**")
                    for line in row['AI Feedback'].split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            st.markdown(f"**{key.strip()}:** {value.strip()}")
                if 'Linked Goals' in row and pd.notna(row['Linked Goals']) and row['Linked Goals']:
                    st.markdown(f"**🔗 Linked Goals:** {row['Linked Goals']}")