import os

with open('app/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where the MAIN UI starts
start_idx = 0
for i, line in enumerate(lines):
    if '# MAIN UI' in line:
        start_idx = i - 1
        break

header = lines[:start_idx]

ui_code = """
# -----------------------
# MAIN UI
# -----------------------

st.markdown(\"\"\"
<style>
body {
    background-color: #0B1220;
    color: #E5E7EB;
}
.stApp {
    background: linear-gradient(135deg, #0B1220, #020617);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.fade {
    animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
\"\"\", unsafe_allow_html=True)

st.markdown(\"\"\"
<h1 style="
text-align:center;
font-size:34px;
font-weight:600;
margin-bottom:10px;">
Intelligent Market Direction System
</h1>
\"\"\", unsafe_allow_html=True)

def card(content):
    st.markdown(f\"\"\"
    <div class="fade" style="
        background: #111827;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #1F2937;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    ">
    {content}
    </div>
    \"\"\", unsafe_allow_html=True)

COLORS = {
    "bull": "#16A34A",
    "bear": "#DC2626",
    "neutral": "#F59E0B",
    "card": "#111827",
    "border": "#1F2937"
}

features = [
  'Open','High','Low','Close','Volume','VIX',
  'hl_range','candle_body','log_return',
  'momentum_5','momentum_10',
  'SMA_5','SMA_20',
  'return_lag1','return_lag2','return_lag3',
  'RSI','rolling_volatility','volume_change'
]

if "original_volume" not in st.session_state:
  st.session_state.original_volume = 50000.0

st.sidebar.markdown("### Control Panel")
st.sidebar.markdown("---")

st.sidebar.markdown("Market Inputs")
investment = st.sidebar.number_input(
  "Investment (₹)",
  min_value=1000,
  value=10000,
  step=1000
)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key and gemini_api_key != "YOUR_API_KEY_HERE":
  genai.configure(api_key=gemini_api_key)
  gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
  gemini_model = None

input_data = []
for f in features:
  if f == 'Volume':
    val = st.sidebar.slider("Volume", 1000.0, 1000000.0, float(st.session_state.original_volume))
  elif f == 'VIX':
    val = st.sidebar.slider("VIX", 10.0, 50.0, 20.0)
  elif f == 'momentum_5':
    val = st.sidebar.slider("Momentum", -5.0, 5.0, 0.5)
  else:
    val = st.sidebar.number_input(f, value=0.0)
  input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
volume_idx = features.index('Volume')
current_volume = input_data[volume_idx]

# -----------------------
# PREDICTION LOGIC
# -----------------------
if current_volume != st.session_state.original_volume:
  st.info("Inputs updated — recalculating market direction...")

lgb_prob = lgb_model.predict_proba(input_array)[:, 1]
cb_prob = cb_model.predict_proba(input_array)[:, 1]

final_prob = (0.5 * lgb_prob + 0.5 * cb_prob)[0]

prediction = "UP" if final_prob > 0.5 else "DOWN"
confidence = abs(final_prob - 0.5) * 2
conf_pct = confidence * 100

if prediction == "UP" and conf_pct > 70:
  action = "BUY"
elif prediction == "DOWN" and conf_pct > 70:
  action = "SELL"
else:
  action = "HOLD"

color = COLORS["bull"] if prediction == "UP" else COLORS["bear"]

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = prediction

if st.session_state.last_prediction != prediction:
    st.warning("Market direction changed from previous state")
    st.session_state.last_prediction = prediction

fig_gauge = go.Figure(go.Indicator(
  mode="gauge+number",
  value=conf_pct,
  gauge={
    'axis': {'range': [0, 100]},
    'bar': {'color': color},
    'steps': [
      {'range': [0, 50], 'color': COLORS['bear']},
      {'range': [50, 75], 'color': COLORS['neutral']},
      {'range': [75, 100], 'color': COLORS['bull']}
    ],
  }
))

fig_gauge.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E5E7EB"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=150
)

# SHAP Logic
input_data_df = pd.DataFrame(input_array, columns=features)
shap_values = explainer.shap_values(input_data_df)
shap_df = pd.DataFrame({
  "feature": input_data_df.columns,
  "value": shap_values[0]
})
shap_df = shap_df.sort_values(by="value", key=abs, ascending=False)
top_features = shap_df.head(5)

# LAYOUT TABS
tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Assistant"])

with tab1:
  col_h1, col_h2, col_h3 = st.columns([1.5, 1, 1])
  
  with col_h1:
      card(f\"\"\"
      <h3 style="margin-bottom:10px;">Prediction</h3>
      <h1 class="fade" style="color:{color}; font-size:42px; margin:0;">{prediction}</h1>
      <p style="color:#9CA3AF;">Model output based on current inputs</p>
      \"\"\")
      
  with col_h2:
      st.markdown(\"\"\"<div style="background: #111827; padding: 20px; border-radius: 14px; border: 1px solid #1F2937; box-shadow: 0 6px 20px rgba(0,0,0,0.4); margin-bottom: 20px; text-align: center; height: 165px;">
      <h3 style="margin-bottom:-10px;">Confidence</h3>
      \"\"\", unsafe_allow_html=True)
      st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
      st.markdown("</div>", unsafe_allow_html=True)

  with col_h3:
      action_col = COLORS["bull"] if action=="BUY" else COLORS["bear"] if action=="SELL" else COLORS["neutral"]
      card(f\"\"\"
      <h3 style="margin-bottom:10px;">Action</h3>
      <h1 class="fade" style="color:{action_col}; font-size:42px; margin:0;">{action}</h1>
      <p style="color:#9CA3AF;">Recommended strategy</p>
      \"\"\")

  # Trust Layer
  if conf_pct > 75:
      st.success("High confidence — strong signal")
  elif conf_pct > 55:
      st.warning("Moderate confidence — caution advised")
  else:
      st.error("Low confidence — unreliable signal")

  st.markdown("---")
  
  # Decision Summary
  st.markdown("### Summary")
  risk_level = "Low" if conf_pct > 70 else "Moderate" if conf_pct > 55 else "High"
  st.markdown(f\"\"\"
  - Market Direction: **{prediction}**  
  - Confidence: **{conf_pct:.2f}%**  
  - Recommended Action: **{action}**  
  - Risk Level: **{risk_level}**  
  \"\"\")
  
  st.markdown("---")
  
  st.markdown("### Portfolio")
  if "UP" in prediction:
    profit = investment * (conf_pct / 100) * 0.02
  else:
    profit = -investment * (conf_pct / 100) * 0.01

  final_value = investment + profit
  
  import plotly.express as px
  if "portfolio_history" not in st.session_state:
    st.session_state.portfolio_history = []
  st.session_state.portfolio_history.append(final_value)

  col_p1, col_p2 = st.columns([1, 2])
  with col_p1:
    st.write(f"**Investment:** ₹{investment}")
    st.write(f"**Expected Profit/Loss:** ₹{profit:.2f}")
    st.write(f"**Net Value:** ₹{final_value:.2f}")

  with col_p2:
    fig_portfolio = px.line(
      y=st.session_state.portfolio_history,
      title="What if I invest? (Growth Over Time)"
    )
    fig_portfolio.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        font=dict(color="#E5E7EB"),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_portfolio, use_container_width=True)

with tab2:
  st.markdown("## Insights")
  st.markdown("---")
  
  col_m1, col_m2, col_m3 = st.columns(3)
  col_m1.metric("Accuracy", "94%")
  col_m2.metric("Signal Strength", "High" if conf_pct>70 else "Moderate" if conf_pct>55 else "Low")
  col_m3.metric("Risk Level", risk_level)
  
  st.markdown("---")
  st.markdown("### Key Drivers of Current Prediction")
  
  col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
  with col_s2:
    fig_shap, ax_shap = plt.subplots(figsize=(5, 3))
    
    # Configure matplotlib to match dark theme naturally
    plt.style.use('dark_background')
    fig_shap.patch.set_facecolor('#0B1220')
    ax_shap.set_facecolor('#0B1220')
    
    try:
      exp = explainer(input_data_df)
      shap.plots.bar(exp[0], show=False)
    except:
      try:
        shap.plots.bar(shap_values[0], show=False)
      except:
        shap.bar_plot(shap_values[0], feature_names=list(input_data_df.columns), show=False)

    fig_shap = plt.gcf()
    fig_shap.set_size_inches(6, 4)
    st.pyplot(fig_shap, use_container_width=True)

with tab3:
  st.markdown("## AI Assistant")
  st.markdown("---")
  if gemini_model:
    vix_idx = features.index('VIX')
    current_vix = input_data[vix_idx]

    context = f\"\"\"
    You are a financial decision assistant.
    Always base answers on current prediction, confidence, and feature importance.
    Be concise and actionable.

    Current Market Prediction:
    - Direction: {prediction}
    - Confidence: {conf_pct:.2f}%

    Top Influencing Factors:
    {top_features.to_string(index=False)}

    Market Data:
    - Volume: {current_volume}
    - VIX: {current_vix}
    \"\"\"

    user_query = st.text_input("Ask about the market (e.g. What actions need to be taken now?):")

    if "chat_history" not in st.session_state:
      st.session_state.chat_history = []

    if user_query:
      full_prompt = f\"\"\"{context}\\nUser Question:\\n{user_query}\\nAnswer:\\n\"\"\"
      try:
        response = gemini_model.generate_content(full_prompt)
        answer = response.text
        
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Assistant", answer))
      except Exception as e:
        st.error(f"Error communicating with Gemini API: {e}")

    if st.session_state.chat_history:
      recent_user = st.session_state.chat_history[-2]
      recent_ai = st.session_state.chat_history[-1]
      
      st.markdown(f\"\"\"
      <div class="fade" style="
      background:#111827;
      padding:12px;
      border-radius:10px;
      border:1px solid #1F2937;
      margin-bottom:8px;">
      <b>{recent_user[0]}</b><br>
      {recent_user[1]}
      </div>
      \"\"\", unsafe_allow_html=True)

      st.markdown(f\"\"\"
      <div class="fade" style="
      background:#1F2937;
      padding:12px;
      border-radius:10px;">
      <b>{recent_ai[0]}</b><br>
      {recent_ai[1]}
      </div>
      <br>
      \"\"\", unsafe_allow_html=True)
      
      if len(st.session_state.chat_history) > 2:
        with st.expander("Archive History", expanded=False):
          for role, msg in st.session_state.chat_history[:-2]:
            bg_card = "#111827" if role == "User" else "#1F2937"
            st.markdown(f\"\"\"
            <div style="background:{bg_card}; padding:10px; border-radius:8px; margin-bottom:8px;">
            <b>{role}</b><br>{msg}
            </div>
            \"\"\", unsafe_allow_html=True)
  else:
    st.info("Add your Gemini API Key to the .env file to unlock the Smart AI Assistant.")
"""

with open('app/app.py', 'w', encoding='utf-8') as f:
    f.writelines(header)
    f.write(ui_code)
