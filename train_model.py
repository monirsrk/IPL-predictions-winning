"""
IPL Win Predictor - মডেল ট্রেনিং স্ক্রিপ্ট
ব্যবহার:  python train_model.py
দরকার:    একই ফোল্ডারে matches.csv ও deliveries.csv
ফলাফল:    pipe.pkl (অ্যাপ এই ফাইলটাই ব্যবহার করে)
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss

# পুরনো/নতুন টিমের নাম একীভূত করা
RENAME = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}
REMOVE = ["Rising Pune Supergiant", "Rising Pune Supergiants",
          "Pune Warriors", "Kochi Tuskers Kerala", "Gujarat Lions"]

mt = pd.read_csv("matches.csv")
dv = pd.read_csv("deliveries.csv")

# ১ম ইনিংসের মোট রান = টার্গেট
first = dv[dv["inning"] == 1].groupby("match_id")["total_runs"].sum().reset_index()
first.rename(columns={"total_runs": "target"}, inplace=True)

mt = mt.merge(first, left_on="id", right_on="match_id")
mt = mt[(mt["dl_applied"] == 0) & mt["winner"].notna() & mt["city"].notna()]
mt = mt[~mt["team1"].isin(REMOVE) & ~mt["team2"].isin(REMOVE)]
mt["winner"] = mt["winner"].replace(RENAME)
mt = mt[["match_id", "city", "winner", "target"]]

# ২য় ইনিংসের বল-বাই-বল ডেটা
d = dv[dv["inning"] == 2].copy()
d["batting_team"] = d["batting_team"].replace(RENAME)
d["bowling_team"] = d["bowling_team"].replace(RENAME)
d = d[~d["batting_team"].isin(REMOVE) & ~d["bowling_team"].isin(REMOVE)]
df = mt.merge(d, on="match_id")

# ফিচার ইঞ্জিনিয়ারিং
df["total_runs"] = pd.to_numeric(df["total_runs"]).fillna(0)
df["current_score"] = df.groupby("match_id")["total_runs"].cumsum()
df["runs_left"] = df["target"] - df["current_score"]

# ওয়াইড/নো-বল বাদ দিয়ে শুধু বৈধ বল গোনা (ডেটাসেটে থাকলে)
if "extras_type" in df.columns:
    legal = ~df["extras_type"].isin(["wides", "noballs"])
else:
    legal = pd.Series(True, index=df.index)
df["balls_done"] = legal.astype(int).groupby(df["match_id"]).cumsum()
df["balls_left"] = 120 - df["balls_done"]

df["wicket"] = df["player_dismissed"].notna().astype(int)
df["wickets"] = 10 - df.groupby("match_id")["wicket"].cumsum()

df = df[(df["balls_done"] > 0) & (df["balls_left"] > 0)]
df["crr"] = df["current_score"] * 6 / df["balls_done"]
df["rrr"] = df["runs_left"] * 6 / df["balls_left"]
df["result"] = (df["batting_team"] == df["winner"]).astype(int)

final = df[["batting_team", "bowling_team", "city", "runs_left", "balls_left",
            "wickets", "target", "crr", "rrr", "result"]].dropna()
final = final.rename(columns={"target": "total_runs_x"})  # আগের নামের সাথে মিল রাখতে

X = final.drop(columns="result")
y = final["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

trf = ColumnTransformer(
    [("trf", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
      ["batting_team", "bowling_team", "city"])],
    remainder="passthrough",
)
pipe = Pipeline([("step1", trf), ("step2", LogisticRegression(solver="liblinear"))])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)
print("Accuracy :", round(accuracy_score(y_test, pred), 4))
print("Log loss :", round(log_loss(y_test, proba), 4))

pickle.dump(pipe, open("pipe.pkl", "wb"))
print("pipe.pkl save হয়েছে ✅")
