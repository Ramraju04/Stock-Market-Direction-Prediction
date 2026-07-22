import os

file_path = "app/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace PAGE CONFIG
page_config_target = """# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Market Direction System", layout="wide")"""

page_config_replacement = """# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Market Direction System", layout="wide", initial_sidebar_state="expanded")

st.markdown(\"""
<style>
body {
    background-color: #0f172a;
    color: #e2e8f0;
}
.stApp {
    background-color: #0f172a;
}
.stMetric {
    background-color: #1e293b !important;
    padding: 15px;
    border-radius: 10px;
}
[data-testid="stSidebar"] {
    background-color: #1e293b !important;
}
.chat-bubble-user {
    background-color: #3b82f6; color: white; padding: 10px 15px; border-radius: 15px 15px 0 15px;
    margin: 10px 0; max-width: 80%; align-self: flex-end; position: relative; float: right; clear: both;
}
.chat-bubble-ai {
    background-color: #334155; color: #e2e8f0; padding: 10px 15px; border-radius: 15px 15px 15px 0;
    margin: 10px 0; max-width: 80%; align-self: flex-start; position: relative; float: left; clear: both;
}
</style>
\""", unsafe_allow_html=True)"""

content = content.replace(page_config_target, page_config_replacement)

# Cut off before MAIN UI and append new content
main_ui_marker = """# -----------------------
# MAIN UI
# -----------------------"""

split_idx = content.find(main_ui_marker)
if split_idx != -1:
    content = content[:split_idx]

new_ui_content = """# -----------------------
# MAIN UI & SIDEBAR
# -----------------------
import datetime
import plotly.express as px
import plotly.graph_objects as go

st.sidebar.markdown("### 🧭 Navigation")
nav_choice = st.sidebar.radio("Go to", ["📊 Dashboard", "🤖 Models", "💰 Portfolio", "⚙️ Settings"])

features = [
  'Open','High','Low','Close','Volume','VIX',
  'hl_range','candle_body','log_return',
  'momentum_5','momentum_10',
  'SMA_5','SMA_20',
  'return_lag1','return_lag2','return_lag3',
  'RSI','rolling_volatility','volume_change'
]

# Sidebar Inputs (Simulation)
st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard Inputs")
investment = st.sidebar.number_input("Enter Investment Amount (₹)", min_value=1000, value=10000, step=1000)

if "original_volume" not in st.session_state:
  st.session_state.original_volume = 50000.0

input_data = []
for f in features:
  if f == 'Volume':
    val = st.sidebar.slider("Volume", 1000.0, 1000000.0, float(st.session_state.original_volume), help="Trading Volume Simulation")
  elif f == 'VIX':
    val = st.sidebar.slider("VIX", 10.0, 50.0, 20.0, help="Market Volatility")
  elif f == 'momentum_5':
    val = st.sidebar.slider("Momentum", -5.0, 5.0, 0.5)
  else:
    val = 0.0
  input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
volume_idx = features.index('Volume')
current_volume = input_data[volume_idx]

# API KEY
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key and gemini_api_key != "YOUR_API_KEY_HERE":
  genai.configure(api_key=gemini_api_key)
  gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
  gemini_model = None

# PREDICTION
lgb_prob = lgb_model.predict_proba(input_array)[:, 1]
cb_prob = cb_model.predict_proba(input_array)[:, 1]
final_prob = (0.5 * lgb_prob + 0.5 * cb_prob)[0]

confidence = abs(final_prob - 0.5) * 2
conf_pct = confidence * 100

if conf_pct < 40:
    conf_label = "Weak"
    conf_message = "Low Confidence – Avoid big trades"
elif conf_pct < 70:
    conf_label = "Moderate"
    conf_message = "Moderate Confidence – Consider hedging"
else:
    conf_label = "Strong"
    conf_message = "Strong Confidence – Favorable setup"

signal_dir = "DOWN" if final_prob < 0.5 else "UP"

if nav_choice == " Dashboard":
    st.markdown("##  Market Overview")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.container(border=True):
            st.markdown("#### " + ("🔴 MARKET SIGNAL" if signal_dir == "DOWN" else "🟢 MARKET SIGNAL"))
            if signal_dir == "DOWN":
                st.markdown(f"<h1 style='color: #ef4444; font-size: 3rem;'>⬇️ DOWN</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='color: #22c55e; font-size: 3rem;'>⬆️ UP</h1>", unsafe_allow_html=True)
            
            st.write(f"**Confidence:** {conf_pct:.1f}%")
            if conf_label == "Weak":
                st.warning(f"{conf_message} ⚠️")
            elif conf_label == "Moderate":
                st.info(f"{conf_message} ⚖️")
            else:
                st.success(f"{conf_message} ✨")

            if st.button("Save to History"):
                cursor.execute(
                    "INSERT INTO history (username, prediction, confidence) VALUES (?, ?, ?)",
                    (st.session_state.username, signal_dir, float(confidence))
                )
                conn.commit()
                st.toast("Saved prediction to history!", icon='✅')

        with st.container(border=True):
            if final_prob >= 0.5:
              if conf_pct > 70: action, risk = "BUY", "Low Risk"
              elif conf_pct > 55: action, risk = "BUY", "Moderate Risk"
              else: action, risk = "HOLD", "High Risk"
            else:
              if conf_pct > 70: action, risk = "SELL", "Low Risk"
              elif conf_pct > 55: action, risk = "SELL", "Moderate Risk"
              else: action, risk = "HOLD", "High Risk"
            
            st.markdown("#### 🎯 AI Recommendation")
            
            if action == "HOLD":
                st.markdown(f"<h2 style='color: #facc15;'>🟡 {action}</h2>", unsafe_allow_html=True)
                st.write(f"**Risk Level:** {risk}")
                st.markdown("**Reason:** Market unclear + weak trend\\n**Suggestion:** Avoid entry. Wait for confirmation.")
            elif action == "BUY":
                st.markdown(f"<h2 style='color: #22c55e;'>🟢 {action}</h2>", unsafe_allow_html=True)
                st.write(f"**Risk Level:** {risk}")
                st.markdown("**Reason:** Strong bullish momentum detected\\n**Suggestion:** Look for optimal entry points.")
            else:
                st.markdown(f"<h2 style='color: #ef4444;'>🔴 {action}</h2>", unsafe_allow_html=True)
                st.write(f"**Risk Level:** {risk}")
                st.markdown("**Reason:** Downside risk identified\\n**Suggestion:** Consider taking profits or shorting.")
    
    with col2:
        with st.container(border=True):
            st.markdown("#### 📈 Price Chart Context")
            np.random.seed(42)
            dates = pd.date_range(end=datetime.datetime.now(), periods=100)
            close_prices = 150 + np.random.randn(100).cumsum()
            high_prices = close_prices + np.random.rand(100) * 3
            low_prices = close_prices - np.random.rand(100) * 3
            open_prices = close_prices - np.random.randn(100)

            fig_chart = go.Figure(data=[go.Candlestick(x=dates,
                            open=open_prices, high=high_prices, low=low_prices, close=close_prices,
                            increasing_line_color='#22c55e', decreasing_line_color='#ef4444')])
            
            sma_20 = pd.Series(close_prices).rolling(window=20).mean()
            fig_chart.add_trace(go.Scatter(x=dates, y=sma_20, mode='lines', name='SMA 20', line=dict(color='#facc15', width=2)))

            fig_chart.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, height=450
            )
            st.plotly_chart(fig_chart, use_container_width=True)

    st.markdown("---")
    colA, colB = st.columns([1, 1])
    
    with colA:
        with st.container(border=True):
            st.markdown("#### 🧠 What's driving this prediction?")
            input_data_df = pd.DataFrame(input_array, columns=features)
            shap_values = explainer.shap_values(input_data_df)
            shap_df = pd.DataFrame({"feature": input_data_df.columns, "value": shap_values[0]})
            shap_df = shap_df.sort_values(by="value", key=abs, ascending=False)
            top_features = shap_df.head(6)

            feature_map = {
              "VIX": "Market Volatility (VIX)", "momentum_5": "Short-term Momentum", "Volume": "Trading Volume",
              "candle_body": "Candle Strength", "log_return": "Price Return", "volume_change": "Volume Momentum"
            }

            bullish = top_features[top_features["value"] > 0]
            bearish = top_features[top_features["value"] < 0]

            if not bullish.empty:
                st.markdown("<h5 style='color:#22c55e;'>🟢 Bullish factors</h5>", unsafe_allow_html=True)
                for _, row in bullish.iterrows():
                    st.write(f"- {feature_map.get(row['feature'], row['feature'])} (Strength: +{row['value']:.2f})")
            
            if not bearish.empty:
                st.markdown("<h5 style='color:#ef4444;'>🔴 Bearish factors</h5>", unsafe_allow_html=True)
                for _, row in bearish.iterrows():
                    st.write(f"- {feature_map.get(row['feature'], row['feature'])} (Strength: {row['value']:.2f})")

    with colB:
        with st.container(border=True):
            st.markdown("#### 🤖 AI Chat Assistant")
            if gemini_model:
              if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
              
              st.markdown(\"""
              <div style='display:flex; gap:10px; margin-bottom: 15px;'>
                 <span style='background:#1e293b; color:#e2e8f0; padding:5px 10px; border-radius:15px; font-size:12px; border:1px solid #334155;'>💡 Prompt: Why is the market down?</span>
                 <span style='background:#1e293b; color:#e2e8f0; padding:5px 10px; border-radius:15px; font-size:12px; border:1px solid #334155;'>💡 Prompt: Risk level?</span>
              </div>
              \""", unsafe_allow_html=True)
              
              chat_container = st.container(height=280)
              with chat_container:
                  for role, msg in st.session_state.chat_history:
                      if role == "User":
                          st.markdown(f"<div class='chat-bubble-user'>{msg}</div>", unsafe_allow_html=True)
                      else:
                          st.markdown(f"<div class='chat-bubble-ai'>{msg}</div>", unsafe_allow_html=True)
              
              user_query = st.chat_input("Ask about the market...")
              if user_query:
                  context = f"Prediction is {signal_dir}. Top impacting factors: {top_features.to_string(index=False)}"
                  full_prompt = f"{context}\\n\\nUser Question: {user_query}\\nAnswer concisely:"
                  with st.spinner("Analyzing market..."):
                      try:
                          response = gemini_model.generate_content(full_prompt)
                          st.session_state.chat_history.append(("User", user_query))
                          st.session_state.chat_history.append(("AI", response.text))
                          st.rerun()
                      except Exception as e:
                          st.error(f"AI Error: {e}")
            else:
                st.info("Add GEMINI_API_KEY to .env to unlock AI Chat.")

elif nav_choice == "🤖 Models":
    st.markdown("## 🤖 Model Intelligence")
    tab_overview, tab_comparison, tab_best = st.tabs(["📊 Overview", "⚔️ Comparison", "🏆 Best Model"])
    
    data_models = {
      "Model": ["GRU", "LightGBM", "CatBoost", "XGBoost"],
      "Train Accuracy": [79.46, 100.00, 94.28, 97.77],
      "Validation Accuracy": [52.97, 92.91, 94.26, 99.37],
      "Test Accuracy": [52.97, 93.64, 93.97, 93.22],
      "Filtered Accuracy": [52.97, 94.28, 97.77, 94.19],
      "Coverage": [1.00, 0.99, 0.86, 0.98]
    }
    df_models = pd.DataFrame(data_models)
    
    with tab_overview:
        st.dataframe(df_models, use_container_width=True)

    with tab_comparison:
        fig_model = px.bar(df_models, x="Model", y="Test Accuracy", title="Test Accuracy Comparison", color="Model")
        fig_model.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_model, use_container_width=True)
        
    with tab_best:
        with st.container(border=True):
            st.markdown("### 🏆 Best Model: CatBoost")
            st.metric("Filtered Accuracy", "97.77%", "+3.5% vs LightGBM")
            st.write("CatBoost provides the most stable generalization without severe overfitting. Selected for live predictions.")

elif nav_choice == "💰 Portfolio":
    st.markdown("## 💰 Portfolio Simulation")
    
    if "UP" in signal_dir:
      profit = investment * (conf_pct / 100) * 0.02
    else:
      profit = -investment * (conf_pct / 100) * 0.01

    if "portfolio_history" not in st.session_state:
      st.session_state.portfolio_history = [10000]

    final_value = investment + profit # Simpler logic for demonstration, not strictly recurring on itself without action
    st.session_state.portfolio_history.append(final_value)
    
    pnl_pct = (profit / investment) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric("Total Portfolio", f"₹{final_value:.2f}", f"{pnl_pct:.2f}%")
    with col2:
        with st.container(border=True):
            st.metric("Initial Investment", f"₹{investment}", "")
    with col3:
        with st.container(border=True):
            st.metric("Expected PnL", f"₹{profit:.2f}", "Active Check")
            
    with st.container(border=True):
        fig_portfolio = px.line(y=st.session_state.portfolio_history, title="Portfolio Equity Curve Generation")
        fig_portfolio.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_portfolio, use_container_width=True)

elif nav_choice == "⚙️ Settings":
    st.markdown("## ⚙️ Settings / Logs")
    st.write("Prediction history stored in DB over local testing environments:")
    
    df = pd.read_sql("SELECT * FROM history WHERE username=? ORDER BY id DESC", conn, params=(st.session_state.username,))
    st.dataframe(df, use_container_width=True)
"""

content = content + new_ui_content

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updates completed.")
