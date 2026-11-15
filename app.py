import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

# ---------- স্টাইল ----------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.hero {text-align:center; padding: 8px 0 18px;}
.hero h1 {font-size: 2.3rem; margin-bottom: 0;
  background: linear-gradient(90deg,#FF9F1C,#FFD166); -webkit-background-clip:text;
  -webkit-text-fill-color: transparent;}
.hero p {color:#9FB3C8; margin-top:4px;}
.card {background:#1C2541; border-radius:16px; padding:18px 20px; margin:10px 0;
  border:1px solid rgba(255,255,255,.07);}
.stat {display:flex; gap:10px; margin: 6px 0 4px;}
.stat div {flex:1; background:#0B132B; border-radius:12px; padding:10px; text-align:center;}
.stat b {display:block; font-size:1.3rem; color:#FFD166;}
.stat span {font-size:.75rem; color:#9FB3C8; text-transform:uppercase; letter-spacing:.05em;}
.bar-wrap {background:#0B132B; border-radius:999px; height:34px; overflow:hidden; display:flex;
  margin-top:10px;}
.bar-a {background:linear-gradient(90deg,#FF9F1C,#FFB84D); display:flex; align-items:center;
  padding-left:14px; color:#0B132B; font-weight:700; white-space:nowrap;}
.bar-b {background:linear-gradient(90deg,#3A86FF,#5AA0FF); flex:1; display:flex;
  align-items:center; justify-content:flex-end; padding-right:14px; font-weight:700; white-space:nowrap;}
.names {display:flex; justify-content:space-between; font-weight:600; margin-top:8px;}
.big {font-size:1.1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏏 IPL Win Predictor</h1>
  <p>২য় ইনিংসের যেকোনো মুহূর্তে জেতার সম্ভাবনা জানুন</p>
</div>
""", unsafe_allow_html=True)


# ---------- মডেল লোড ----------
@st.cache_resource
def load_model():
    base = Path(__file__).parent
    files = sorted(base.glob("*.pkl"))
    if not files:
        return None, "NO_FILE"
    try:
        with open(files[0], "rb") as f:
            return pickle.load(f), None
    except Exception as e:  # ভার্সন মিসম্যাচ ইত্যাদি
        return None, f"{type(e).__name__}: {e}"


pipe, load_error = load_model()
if pipe is None:
    if load_error == "NO_FILE":
        st.error("কোনো `.pkl` ফাইল পাওয়া যায়নি। `pipe.pkl` ফাইলটা `app.py`-র পাশের ফোল্ডারে রাখুন।")
        st.write("এই ফোল্ডারে আছে:", sorted(p.name for p in Path(__file__).parent.iterdir()))
    else:
        import sklearn
        st.error("মডেল লোড করা যায়নি। সম্ভবত scikit-learn ভার্সন মিলছে না।")
        st.write("বর্তমান scikit-learn ভার্সন:", sklearn.__version__, "(দরকার: 1.6.1)")
        st.code(load_error)
    st.stop()

# টিম ও শহরের তালিকা সরাসরি মডেল থেকে নেওয়া হচ্ছে, তাই নাম কখনো মিসম্যাচ হবে না
cats = pipe.named_steps["step1"].named_transformers_["trf"].categories_
all_teams = sorted(cats[0])
all_cities = sorted(cats[2])

# ---------- ইনপুট ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2 = st.columns(2)
batting_team = c1.selectbox("🏏 Batting Team", all_teams, index=0)
bowling_options = [t for t in all_teams if t != batting_team]
bowling_team = c2.selectbox("🎯 Bowling Team", bowling_options, index=0)
city = st.selectbox("📍 Host City", all_cities)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
target = st.number_input("Target (জিততে লাগবে)", min_value=1, max_value=300, value=180)
a, b, c, d = st.columns(4)
score = a.number_input("Score", min_value=0, max_value=350, value=90)
overs = b.number_input("Overs", min_value=0, max_value=19, value=10)
balls = c.number_input("Balls", min_value=0, max_value=5, value=0)
wickets_out = d.number_input("Wickets", min_value=0, max_value=10, value=3)
st.markdown("</div>", unsafe_allow_html=True)

predict = st.button("🔮 Predict Probability", use_container_width=True, type="primary")

# ---------- প্রেডিকশন ----------
if predict:
    balls_done = overs * 6 + balls
    balls_left = 120 - balls_done
    runs_left = target - score
    wickets_left = 10 - wickets_out

    if balls_done == 0:
        st.warning("কমপক্ষে ১টা বল খেলা হয়ে থাকতে হবে।")
    elif runs_left <= 0:
        st.success(f"🎉 {batting_team} ইতিমধ্যে টার্গেট ছাড়িয়ে গেছে, ম্যাচ জিতে গেছে!")
    elif wickets_left == 0:
        st.error(f"❌ সব উইকেট পড়ে গেছে, {bowling_team} জিতেছে।")
    else:
        crr = score * 6 / balls_done
        rrr = runs_left * 6 / balls_left

        x = pd.DataFrame({
            "batting_team": [batting_team], "bowling_team": [bowling_team],
            "city": [city], "runs_left": [runs_left], "balls_left": [balls_left],
            "wickets": [wickets_left], "total_runs_x": [target],
            "crr": [crr], "rrr": [rrr],
        })
        p = pipe.predict_proba(x)[0]
        win_bat, win_bowl = round(p[1] * 100), round(p[0] * 100)

        st.markdown(f"""
        <div class="card">
          <div class="stat">
            <div><b>{runs_left}</b><span>Runs left</span></div>
            <div><b>{balls_left}</b><span>Balls left</span></div>
            <div><b>{wickets_left}</b><span>Wickets left</span></div>
          </div>
          <div class="stat">
            <div><b>{crr:.2f}</b><span>Current RR</span></div>
            <div><b>{rrr:.2f}</b><span>Required RR</span></div>
          </div>
        </div>
        <div class="card">
          <div class="names"><span class="big">{batting_team}</span>
            <span class="big">{bowling_team}</span></div>
          <div class="bar-wrap">
            <div class="bar-a" style="width:{max(win_bat, 12)}%">{win_bat}%</div>
            <div class="bar-b">{win_bowl}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        fav = batting_team if win_bat >= win_bowl else bowling_team
        st.caption(f"মডেলের হিসাবে এই মুহূর্তে **{fav}** এগিয়ে আছে।")

st.markdown("---")
st.caption("Logistic Regression মডেল • ঐতিহাসিক IPL ডেটার উপর ভিত্তি করে • শুধু শেখার উদ্দেশ্যে")
