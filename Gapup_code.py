import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
import io
import base64
import xlsxwriter
from fpdf import FPDF

# Connect to MongoDB
client = MongoClient("mongodb+srv://my_algo_user:aUdNEfbaM2MDd3k8@cluster0.otr8r.mongodb.net/?retryWrites=true&w=majority")
db = client['Test1']

# Fetch data from MongoDB
trade_details = pd.DataFrame(list(db.Trade_Details.find()))

# Convert MongoDB ObjectId to string
trade_details['_id'] = trade_details['_id'].astype(str)

# Ensure the trade_date column is in datetime format
trade_details['trade_date'] = pd.to_datetime(trade_details['trade_date'])

# Add time of day column if not present
if 'time_of_day' not in trade_details.columns:
    trade_details['time_of_day'] = trade_details['trade_date'].dt.hour

# Ensure 'entry_date' and 'exit_date' columns exist
if 'entry_date' not in trade_details.columns:
    trade_details['entry_date'] = trade_details['trade_date']

if 'exit_date' not in trade_details.columns:
    # Assuming exit_date is some time after entry_date, here using trade_date + 1 hour for demonstration
    trade_details['exit_date'] = trade_details['trade_date'] + pd.Timedelta(hours=1)

# Add trade duration column
trade_details['trade_duration'] = (trade_details['exit_date'] - trade_details['entry_date']).dt.total_seconds() / 3600  # Duration in hours

# Streamlit app layout
st.title("Trading Strategy Performance Dashboard")

# Sidebar for filters
st.sidebar.header("Filters")

# Load saved filters from session state
if "filters" not in st.session_state:
    st.session_state["filters"] = {
        "start_date": trade_details['trade_date'].min().date(),
        "end_date": trade_details['trade_date'].max().date(),
        "selected_symbols": ["All"],
        "selected_trade_types": ["All"],
        "profit_loss_filter": (float(trade_details['pnl'].min()), float(trade_details['pnl'].max())),
        "time_of_day_filter": (0, 23),
        "trade_duration_filter": (0, trade_details['trade_duration'].max())
    }

# Date range filter
start_date = st.sidebar.date_input("Start date", st.session_state["filters"]["start_date"])
end_date = st.sidebar.date_input("End date", st.session_state["filters"]["end_date"])

# Option symbol filter
option_symbols = trade_details['option_symbol'].unique().tolist()
option_symbols.insert(0, "All")
selected_symbols = st.sidebar.multiselect("Option symbols", option_symbols, default=st.session_state["filters"]["selected_symbols"])

# Trade type filter (only if the column exists)
if 'trade_type' in trade_details.columns:
    trade_types = trade_details['trade_type'].unique().tolist()
    trade_types.insert(0, "All")
    selected_trade_types = st.sidebar.multiselect("Trade types", trade_types, default=st.session_state["filters"]["selected_trade_types"])
else:
    selected_trade_types = ["All"]

# Profit/Loss filter
profit_loss_filter = st.sidebar.slider("Profit/Loss range", 
                                       float(trade_details['pnl'].min()), 
                                       float(trade_details['pnl'].max()), 
                                       st.session_state["filters"]["profit_loss_filter"])

# Time of Day filter
time_of_day_filter = st.sidebar.slider("Time of Day range", 0, 23, st.session_state["filters"]["time_of_day_filter"])

# Trade Duration filter (in hours)
trade_duration_filter = st.sidebar.slider("Trade Duration range (hours)", 
                                          0, 
                                          int(trade_details['trade_duration'].max()), 
                                          (0, int(trade_details['trade_duration'].max())))

# Save filters
if st.sidebar.button("Save Filters"):
    st.session_state["filters"] = {
        "start_date": start_date,
        "end_date": end_date,
        "selected_symbols": selected_symbols,
        "selected_trade_types": selected_trade_types,
        "profit_loss_filter": profit_loss_filter,
        "time_of_day_filter": time_of_day_filter,
        "trade_duration_filter": trade_duration_filter
    }
    st.sidebar.success("Filters saved!")

# Apply filters to trade details
filtered_trades = trade_details[
    (trade_details['trade_date'] >= pd.to_datetime(start_date)) &
    (trade_details['trade_date'] <= pd.to_datetime(end_date))
]

if "All" not in selected_symbols:
    filtered_trades = filtered_trades[filtered_trades['option_symbol'].isin(selected_symbols)]

if "All" not in selected_trade_types and 'trade_type' in trade_details.columns:
    filtered_trades = filtered_trades[filtered_trades['trade_type'].isin(selected_trade_types)]

filtered_trades = filtered_trades[
    (filtered_trades['pnl'] >= profit_loss_filter[0]) &
    (filtered_trades['pnl'] <= profit_loss_filter[1]) &
    (filtered_trades['time_of_day'] >= time_of_day_filter[0]) &
    (filtered_trades['time_of_day'] <= time_of_day_filter[1]) &
    (filtered_trades['trade_duration'] >= trade_duration_filter[0]) &
    (filtered_trades['trade_duration'] <= trade_duration_filter[1])
]

# Detailed trade log
st.header("Detailed Trade Log")
st.dataframe(filtered_trades)

# Download button for trade log
csv = filtered_trades.to_csv(index=False)
st.download_button(label="Download Trade Log as CSV", data=csv, file_name="trade_log.csv", mime="text/csv")

# Calculate daily PNL
daily_pnl = filtered_trades.groupby(filtered_trades['trade_date'].dt.date)['pnl'].sum().reset_index()
daily_pnl.columns = ['Date', 'PNL after Costs']

# Cumulative PNL
st.header("Cumulative PNL")
cumulative_pnl = daily_pnl['PNL after Costs'].cumsum()
fig_cumulative_pnl = px.line(x=daily_pnl['Date'], y=cumulative_pnl, title="Cumulative PNL Over Time", labels={'x': 'Date', 'y': 'Cumulative PNL'})
st.plotly_chart(fig_cumulative_pnl)

# Daily PNL
st.header("Daily PNL")
fig_daily_pnl = px.bar(daily_pnl, x='Date', y='PNL after Costs', title="Daily PNL")
st.plotly_chart(fig_daily_pnl)

# Trade Statistics
st.header("Trade Statistics")
total_trades = len(filtered_trades)
winning_trades = len(filtered_trades[filtered_trades['pnl'] > 0])
losing_trades = len(filtered_trades[filtered_trades['pnl'] <= 0])
win_rate = (winning_trades / total_trades) * 100
average_pnl = filtered_trades['pnl'].mean()
max_drawdown = cumulative_pnl.cummax() - cumulative_pnl
max_drawdown = max_drawdown.max()

st.write(f"Total Trades: {total_trades}")
st.write(f"Winning Trades: {winning_trades}")
st.write(f"Losing Trades: {losing_trades}")
st.write(f"Win Rate: {win_rate:.2f}%")
st.write(f"Average PNL: {average_pnl:.2f}")
st.write(f"Maximum Drawdown: {max_drawdown:.2f}")

# Performance Metrics
st.header("Performance Metrics")

# Sortino Ratio
target_return = 0
negative_returns = daily_pnl['PNL after Costs'][daily_pnl['PNL after Costs'] < target_return]
sortino_ratio = (daily_pnl['PNL after Costs'].mean() - target_return) / negative_returns.std()

# Maximum Profit and Loss
max_profit = filtered_trades['pnl'].max()
max_loss = filtered_trades['pnl'].min()

# Daily Volatility
daily_volatility = daily_pnl['PNL after Costs'].std()

# Calmar Ratio
calmar_ratio = daily_pnl['PNL after Costs'].mean() / max_drawdown if max_drawdown != 0 else np.nan

# Omega Ratio
threshold_return = 0
excess_returns = daily_pnl['PNL after Costs'] - threshold_return
omega_ratio = excess_returns[excess_returns > 0].sum() / -excess_returns[excess_returns < 0].sum()

# Return Over Maximum Drawdown (RoMaD)
return_over_max_drawdown = cumulative_pnl.iloc[-1] / max_drawdown if max_drawdown != 0 else np.nan

st.write(f"Sortino Ratio: {sortino_ratio:.2f}")
st.write(f"Maximum Profit: {max_profit:.2f}")
st.write(f"Maximum Loss: {max_loss:.2f}")
st.write(f"Daily Volatility: {daily_volatility:.2f}")
st.write(f"Calmar Ratio: {calmar_ratio:.2f}")
st.write(f"Omega Ratio: {omega_ratio:.2f}")
st.write(f"Return Over Maximum Drawdown (RoMaD): {return_over_max_drawdown:.2f}")

# Additional visualizations and metrics
# Drawdown Chart
st.header("Drawdown Chart")
drawdowns = cumulative_pnl.cummax() - cumulative_pnl
fig_drawdowns = px.area(x=daily_pnl['Date'], y=drawdowns, title="Drawdowns Over Time", labels={'x': 'Date', 'y': 'Drawdown'})
st.plotly_chart(fig_drawdowns)

# Equity Curve
st.header("Equity Curve")
fig_equity_curve = px.line(x=daily_pnl['Date'], y=cumulative_pnl, title="Equity Curve", labels={'x': 'Date', 'y': 'Equity'})
st.plotly_chart(fig_equity_curve)

# Monthly Heatmap
st.header("Monthly Heatmap")
filtered_trades['month'] = filtered_trades['trade_date'].dt.month
filtered_trades['year'] = filtered_trades['trade_date'].dt.year
monthly_heatmap = filtered_trades.pivot_table(index='year', columns='month', values='pnl', aggfunc='sum', fill_value=0)
fig_monthly_heatmap = go.Figure(data=go.Heatmap(z=monthly_heatmap.values, x=monthly_heatmap.columns, y=monthly_heatmap.index, colorscale='Viridis'))
fig_monthly_heatmap.update_layout(title='Monthly PNL Heatmap', xaxis_nticks=12)
st.plotly_chart(fig_monthly_heatmap)

# Advanced Visualizations
# Rolling Sharpe/Sortino Ratio
st.header("Rolling Sharpe/Sortino Ratio")
window_size = st.sidebar.slider("Rolling Window Size (days)", 10, 100, 30)
rolling_sortino = daily_pnl['PNL after Costs'].rolling(window=window_size).apply(
    lambda x: (x.mean() - target_return) / x[x < target_return].std() if x[x < target_return].std() != 0 else np.nan,
    raw=False
)
fig_rolling_sortino = px.line(x=daily_pnl['Date'], y=rolling_sortino, title="Rolling Sortino Ratio", labels={'x': 'Date', 'y': 'Rolling Sortino Ratio'})
st.plotly_chart(fig_rolling_sortino)

# Cumulative Return Comparison
st.header("Cumulative Return Comparison")
benchmark_data = pd.DataFrame({
    "Date": pd.date_range(start=daily_pnl['Date'].min(), end=daily_pnl['Date'].max(), freq='D')
})
years = (benchmark_data['Date'] - benchmark_data['Date'].min()).dt.days / 365.25
benchmark_data['benchmark'] = (1 + 0.15) ** years
benchmark_data.set_index('Date', inplace=True)
daily_pnl.set_index('Date', inplace=True)
merged_data = pd.merge(daily_pnl, benchmark_data, left_index=True, right_index=True, how='inner')
merged_data.reset_index(inplace=True)

fig_cumulative_return_comparison = go.Figure()
fig_cumulative_return_comparison.add_trace(go.Scatter(x=merged_data['Date'], y=merged_data['PNL after Costs'].cumsum(), mode='lines', name='Strategy'))
fig_cumulative_return_comparison.add_trace(go.Scatter(x=merged_data['Date'], y=merged_data['benchmark'], mode='lines', name='Benchmark'))
fig_cumulative_return_comparison.update_layout(title="Cumulative Return Comparison", xaxis_title='Date', yaxis_title='Cumulative Return')
st.plotly_chart(fig_cumulative_return_comparison)

# Scatter Plot of Returns
st.header("Scatter Plot of Returns")
fig_scatter_returns = px.scatter(merged_data, x='benchmark', y='PNL after Costs', title="Scatter Plot of Returns", labels={'benchmark': 'Benchmark Returns', 'PNL after Costs': 'Strategy Returns'})
st.plotly_chart(fig_scatter_returns)

# User Interaction
st.header("Comments/Notes")
comment_date = st.date_input("Select Date", datetime.today())
comment_text = st.text_area("Leave your comment or note here:")
if st.button("Save Comment"):
    if "comments" not in st.session_state:
        st.session_state["comments"] = []
    st.session_state["comments"].append({"date": comment_date, "comment": comment_text})
    st.success("Comment saved!")

# Display saved comments
if "comments" in st.session_state:
    st.subheader("Saved Comments")
    for comment in st.session_state["comments"]:
        st.write(f"{comment['date']}: {comment['comment']}")

# Export to Excel
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()  # Use close instead of save
    processed_data = output.getvalue()
    return processed_data

df_xlsx = to_excel(filtered_trades)
st.download_button(label='📥 Download Excel file', data=df_xlsx, file_name='trade_log.xlsx')

# PDF Report Generation
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Trading Strategy Performance Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()
pdf.chapter_title("Cumulative PNL")
pdf.chapter_body(f"Cumulative PNL: {cumulative_pnl.iloc[-1]:.2f}")
pdf.chapter_title("Trade Statistics")
pdf.chapter_body(f"Total Trades: {total_trades}\nWinning Trades: {winning_trades}\nLosing Trades: {losing_trades}\nWin Rate: {win_rate:.2f}%\nAverage PNL: {average_pnl:.2f}\nMaximum Drawdown: {max_drawdown:.2f}")
pdf.chapter_title("Performance Metrics")
pdf.chapter_body(f"Sortino Ratio: {sortino_ratio:.2f}\nMaximum Profit: {max_profit:.2f}\nMaximum Loss: {max_loss:.2f}\nDaily Volatility: {daily_volatility:.2f}\nCalmar Ratio: {calmar_ratio:.2f}\nOmega Ratio: {omega_ratio:.2f}\nReturn Over Maximum Drawdown (RoMaD): {return_over_max_drawdown:.2f}")

pdf_output = pdf.output(dest='S').encode('latin1')
b64 = base64.b64encode(pdf_output).decode('latin1')
href = f'<a href="data:application/octet-stream;base64,{b64}" download="report.pdf">Download PDF Report</a>'
st.markdown(href, unsafe_allow_html=True)

# Help Section
st.sidebar.header("Help")
st.sidebar.markdown("""
    ### How to Use This App
    1. **Filters**: Use the sidebar to filter trades by date range, option symbols, trade types, profit/loss range, time of day, and trade duration.
    2. **Trade Log**: View the detailed trade log and download it as a CSV or Excel file.
    3. **Performance Metrics**: Analyze the performance metrics such as cumulative PNL, daily PNL, trade statistics, and risk metrics.
    4. **Visualizations**: Explore various visualizations including drawdowns, equity curve, PNL distribution, and monthly heatmap.
    5. **Advanced Metrics**: Use advanced filters to analyze metrics such as Sharpe ratio, Sortino ratio, and daily volatility over time.
    6. **Comments/Notes**: Leave comments or notes on specific dates or trades.
    ### Contact
    For any questions or support, please contact [tusharwasake@gmail.com](mailto:tusharwasake@gmail.com)
""")

# Footer
st.markdown("""
    ---
    **Developed by Tushar Wasake**
""")

# Placeholder for features requiring further setup
st.sidebar.header("Upcoming Features")
st.sidebar.markdown("""
    - **User Authentication**: Secure login and user roles.
    - **Real-time Data**: Integration with live market data.
    - **Backtesting**: Run backtests with historical data.
    - **Alerts**: Setup alerts for significant events.
    - **Mobile Responsiveness**: Ensure the app works well on mobile devices.
""")
