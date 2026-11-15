# 🏏 IPL Win Predictor

২য় ইনিংসের যেকোনো সময়ে (স্কোর, ওভার, উইকেট, টার্গেট দেখে) কোন টিম জেতার সম্ভাবনা বেশি, তা বলে দেয়।
মডেল: **Logistic Regression** (One-Hot Encoding সহ Pipeline), অ্যাপ: **Streamlit**।

## ফোল্ডারের ফাইল

| ফাইল | কাজ |
|---|---|
| `app.py` | ওয়েব অ্যাপ (UI + প্রেডিকশন) |
| `train_model.py` | `matches.csv` ও `deliveries.csv` থেকে মডেল ট্রেন করে `pipe.pkl` বানায় |
| `pipe.pkl` | ট্রেন করা মডেল (train_model.py চালালে তৈরি হবে) |
| `requirements.txt` | দরকারি লাইব্রেরি |
| `.streamlit/config.toml` | ডার্ক থিম |

## লোকালি চালানোর নিয়ম

```bash
pip install -r requirements.txt

# ১) মডেল বানান (matches.csv ও deliveries.csv এই ফোল্ডারে রাখুন)
python train_model.py

# ২) অ্যাপ চালান
streamlit run app.py
```

ব্রাউজারে `http://localhost:8501` খুলবে।

> আপনার notebook থেকেই যদি `pipe.pkl` বানান: notebook-এর শেষে
> `pickle.dump(pipe, open('pipe.pkl','wb'))` চালিয়ে ফাইলটা এই ফোল্ডারে রাখুন।
> তখন `train_model.py` লাগবে না।

## ফ্রিতে অনলাইনে Deploy (Streamlit Community Cloud)

1. এই ফোল্ডারের সব ফাইল (`pipe.pkl` সহ) একটা GitHub repository-তে আপলোড করুন
2. https://share.streamlit.io এ GitHub দিয়ে লগইন করুন
3. **New app** → repository ও `app.py` সিলেক্ট করে **Deploy** চাপুন
4. কিছুক্ষণ পর পাবলিক লিংক পাবেন

**গুরুত্বপূর্ণ:** `pipe.pkl` যে `scikit-learn` ভার্সনে বানিয়েছেন, `requirements.txt`-এ সেই ভার্সন লিখুন
(যেমন `scikit-learn==1.5.2`)। ভার্সন না মিললে মডেল লোড হতে সমস্যা হতে পারে।
`pip show scikit-learn` দিয়ে ভার্সন দেখুন।

## নোট

- অ্যাপের টিম ও শহরের তালিকা সরাসরি মডেল থেকে নেওয়া হয়, তাই নামের ভুল হবে না।
- এটি শুধু শেখার/প্রদর্শনের প্রজেক্ট, বাজি বা জুয়ার জন্য নয়।
