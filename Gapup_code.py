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

# MongoDB Connection
@st.cache_resource
def get_db_connection():
    return MongoClient("mongodb+srv://my_algo_user:aUdNEfbaM2MDd3k8@cluster0.otr8r.mongodb.net/?retryWrites=true&w=majority")['Test1']

# Fetch data from MongoDB
@st.cache_data
def fetch_pnl_details():
    db = get_db_connection()
    pnl_details = pd.DataFrame(list(db['STBT_PNL'].find()))

    if pnl_details.empty:
        st.error("No data found in the 'STBT_PNL' collection.")
        return pnl_details  # Return an empty DataFrame

    # Handle potential missing '_id' column
    if '_id' in pnl_details.columns:
        pnl_details['_id'] = pnl_details['_id'].astype(str)

    # Ensure the 'Date' column is in datetime format
    if 'Date' in pnl_details.columns:
        pnl_details['Date'] = pd.to_datetime(pnl_details['Date'], errors='coerce')
        pnl_details.rename(columns={'Date': 'trade_date'}, inplace=True)
    else:
        st.error("The 'Date' column is missing in the data.")
        return pd.DataFrame()  # Return an empty DataFrame if 'Date' is missing

    # Drop duplicates to avoid double counting
    pnl_details.drop_duplicates(subset=['trade_date'], inplace=True)

    return pnl_details

# Load data
pnl_details = fetch_pnl_details()

# Streamlit app layout
st.title("Trading Strategy Performance Dashboard")

if pnl_details.empty:
    st.error("No valid data available for display. Please check the data source.")
else:
    # Sidebar filters
    st.sidebar.header("Filters")

    # Set default filters if not already in session state
    if "filters" not in st.session_state:
        st.session_state["filters"] = {
            "start_date": pnl_details['trade_date'].min().date(),
            "end_date": pnl_details['trade_date'].max().date(),
            "profit_loss_filter": (float(pnl_details['PNL'].min()), float(pnl_details['PNL'].max())),
        }

    # Date range filter
    start_date = st.sidebar.date_input("Start date", st.session_state["filters"]["start_date"])
    end_date = st.sidebar.date_input("End date", st.session_state["filters"]["end_date"])

    # Profit/Loss filter
    profit_loss_filter = st.sidebar.slider(
        "Profit/Loss range", 
        float(pnl_details['PNL'].min()), 
        float(pnl_details['PNL'].max()), 
        st.session_state["filters"]["profit_loss_filter"]
    )

    # Save filters
    if st.sidebar.button("Save Filters"):
        st.session_state["filters"] = {
            "start_date": start_date,
            "end_date": end_date,
            "profit_loss_filter": profit_loss_filter,
        }
        st.sidebar.success("Filters saved!")

    # Apply filters to pnl details
    filtered_pnl = pnl_details[
        (pnl_details['trade_date'] >= pd.to_datetime(start_date)) &
        (pnl_details['trade_date'] <= pd.to_datetime(end_date)) &
        (pnl_details['PNL'] >= profit_loss_filter[0]) &
        (pnl_details['PNL'] <= profit_loss_filter[1])
    ]

    # Check for potential data issues (e.g., large discrepancies)
    if filtered_pnl['PNL'].max() - filtered_pnl['PNL'].min() > 10000:  # Arbitrary threshold for large discrepancy
        st.warning("Large discrepancies found in the PNL data. Please review the data source.")

    # Detailed PNL log
    st.header("Detailed PNL Log")
    st.dataframe(filtered_pnl)

    # Download button for PNL log
    csv = filtered_pnl.to_csv(index=False)
    st.download_button(label="Download PNL Log as CSV", data=csv, file_name="pnl_log.csv", mime="text/csv")

    # Calculate daily PNL
    daily_pnl = filtered_pnl.groupby(filtered_pnl['trade_date'].dt.date)['PNL'].sum().reset_index()
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
    total_days = len(filtered_pnl)
    positive_days = len(filtered_pnl[filtered_pnl['PNL'] > 0])
    negative_days = len(filtered_pnl[filtered_pnl['PNL'] <= 0])
    win_rate = (positive_days / total_days) * 100
    average_pnl = filtered_pnl['PNL'].mean()
    max_drawdown = cumulative_pnl.cummax() - cumulative_pnl
    max_drawdown = max_drawdown.max()

    st.write(f"Total Days: {total_days}")
    st.write(f"Positive Days: {positive_days}")
    st.write(f"Negative Days: {negative_days}")
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
    max_profit = filtered_pnl['PNL'].max()
    max_loss = filtered_pnl['PNL'].min()

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
    filtered_pnl['month'] = filtered_pnl['trade_date'].dt.month
    filtered_pnl['year'] = filtered_pnl['trade_date'].dt.year
    monthly_heatmap = filtered_pnl.pivot_table(index='year', columns='month', values='PNL', aggfunc='sum', fill_value=0)
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
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()  # Finalize writing process
        processed_data = output.getvalue()
        return processed_data

    df_xlsx = to_excel(filtered_pnl)
    st.download_button(label='📥 Download Excel file', data=df_xlsx, file_name='pnl_log.xlsx')

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
    pdf.chapter_body(f"Total Days: {total_days}\nPositive Days: {positive_days}\nNegative Days: {negative_days}\nWin Rate: {win_rate:.2f}%\nAverage PNL: {average_pnl:.2f}\nMaximum Drawdown: {max_drawdown:.2f}")
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
        1. **Filters**: Use the sidebar to filter PNL by date range and profit/loss range.
        2. **PNL Log**: View the detailed PNL log and download it as a CSV or Excel file.
        3. **Performance Metrics**: Analyze the performance metrics such as cumulative PNL, daily PNL, and risk metrics.
        4. **Visualizations**: Explore various visualizations including drawdowns, equity curve, and monthly heatmap.
        5. **Advanced Metrics**: Use advanced filters to analyze metrics such as Sharpe ratio, Sortino ratio, and daily volatility over time.
        6. **Comments/Notes**: Leave comments or notes on specific dates or PNL data.
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
