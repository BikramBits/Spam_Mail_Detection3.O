from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import os
import imaplib
import email
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import threading
import time

app = Flask(__name__)

# --- GLOBAL VARIABLES FOR NEW FEATURES ---
bg_fetch_active = False
bg_credentials = {"email": "", "password": ""}
global_inbox = []  # Stores fetched emails for real-time inbox and dashboard
processed_email_ids = set()

# --- CONFIGURATION & ML LOGIC (UNCHANGED) ---

DATASETS = [
    {"path": "SMS_Text (1).csv", "text_col": "data", "label_col": "label"},
    {"path": "spam.csv", "text_col": "v2", "label_col": "v1"},
    {"path": "spamassassin.csv", "text_col": "text", "label_col": "label"},
    {"path": "Dataset_10191.csv", "text_col": "TEXT", "label_col": "LABEL"}
]

WHITELIST = ["quincy", "bikram", "freecodecamp"]

SUSPICIOUS_TERMS = [
    "geek squad", "best buy", "mcafee", "norton", "subscription",
    "annual payment", "auto-deduct", "moscow", "russia", "unauthorized login",
    "tax refund", "irs notice", "unclaimed", "refund portal", "24 hours",
    "hiring manager", "telegram", "remote position", "whatsapp", "bitcoin"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r'"', '', text)
    text = re.sub(r"subject:", "", text)
    text = re.sub(r'[^a-z0-9\s$!]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

model = None
vectorizer = None

def load_and_combine_data(datasets_config):
    all_dfs = []
    SPAM_INDICATORS = ['1', '1.0', 'spam', 'smishing', 'phishing', 'yes']

    for config in datasets_config:  
        if os.path.exists(config["path"]):  
            try:  
                temp_df = pd.read_csv(config["path"], encoding='latin-1')  
                new_df = pd.DataFrame()  
                new_df['text'] = temp_df[config["text_col"]].apply(clean_text)  
                new_df['label'] = temp_df[config["label_col"]].apply(  
                    lambda x: 1 if str(x).lower().strip() in SPAM_INDICATORS else 0  
                )  
                new_df = new_df.dropna(subset=['text'])  
                all_dfs.append(new_df)  
                print(f"✅ Loaded {config['path']} ({len(new_df)} rows)")  
            except Exception as e:  
                print(f"⚠️ Skipping {config['path']}: {e}")  

    if not all_dfs: return None  
    return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['text'])

try:
    master_df = load_and_combine_data(DATASETS)
    if master_df is not None:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
        X = vectorizer.fit_transform(master_df['text'])
        y = master_df['label']
        model = MultinomialNB()
        model.fit(X, y)
        print(f"🚀 Trained on {len(master_df)} unique samples.")
except Exception as e:
    print(f"❌ Model Init Error: {e}")

def check_spam(text):
    if model is None: return "OFFLINE", 0, ["Model not loaded."]

    reasons = []  
    text_lower = text.lower()  
      
    for word in WHITELIST:  
        if re.search(fr'\b{re.escape(word)}\b', text_lower):  
            return "SAFE", 0.0, [f"Trusted sender recognized: {word}"]  

    cleaned = clean_text(text)  
    vec = vectorizer.transform([cleaned])  
    ai_prob = model.predict_proba(vec)[0][1] * 100  
      
    if ai_prob > 30:  
        reasons.append(f"AI identified scam patterns ({round(ai_prob)}% base score).")  

    manual_boost = 0  
    for term in SUSPICIOUS_TERMS:  
        if term in text_lower:  
            manual_boost += 25  
            reasons.append(f"Suspicious term detected: '{term}'.")  

    final_score = min(ai_prob + manual_boost, 100.0)  
      
    if final_score >= 70: result = "SPAM"  
    elif final_score >= 35: result = "SUSPICIOUS"  
    else: result = "SAFE"  
      
    return result, round(final_score, 2), reasons



# --- BACKGROUND FETCHING THREAD ---
def background_email_worker():
    global bg_fetch_active, global_inbox, processed_email_ids
    while True:
        if bg_fetch_active and bg_credentials.get("email") and bg_credentials.get("password"):
            try:
                mail = imaplib.IMAP4_SSL("imap.gmail.com")
                mail.login(bg_credentials["email"], bg_credentials["password"])
                mail.select("inbox")
                _, messages = mail.search(None, 'ALL')
                ids = messages[0].split()
                
                if ids:
                    latest_ids = ids[-10:]
                    for num in latest_ids:
                        if num not in processed_email_ids:
                            processed_email_ids.add(num)
                            _, data = mail.fetch(num, "(RFC822)")
                            msg = email.message_from_bytes(data[0][1])
                            subject = msg["subject"] or "(No Subject)"
                            sender = msg.get("From", "Unknown Sender")

                            body = ""
                            if msg.is_multipart():
                                for part in msg.walk():
                                    if part.get_content_type() == "text/plain":
                                        body = part.get_payload(decode=True).decode(errors='ignore')
                                        break
                            else:
                                body = msg.get_payload(decode=True).decode(errors='ignore')

                            res, conf, reasons = check_spam(body)
                            
                            global_inbox.insert(0, {
                                "id": num.decode(),
                                "sender": sender,
                                "subject": subject,
                                "result": res,
                                "confidence": conf,
                                "body_snippet": body[:150] + "...",
                                "full_body": body,
                                "reasons": reasons,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            if len(global_inbox) > 100:
                                global_inbox.pop()
                mail.logout()
            except Exception as e:
                print(f"Background Fetch Error: {e}")
        time.sleep(15)

threading.Thread(target=background_email_worker, daemon=True).start()

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/manual")
def manual():
    return render_template("index.html")

@app.route("/auto-fetch")
def auto_fetch():
    return render_template("fetch.html")

@app.route("/bg-setup")
def bg_setup():
    return render_template("bg_setup.html")

@app.route("/bg-menu")
def bg_menu():
    return render_template("bg_menu.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/inbox")
def inbox():
    return render_template("inbox.html")
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
    

# --- APIs ---

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("text", "")
    res, conf, reasons = check_spam(user_input)
    return jsonify({"result": res, "confidence": conf, "reasons": reasons})

@app.route("/fetch_mail", methods=["POST"])
def fetch_mail():
    email_addr = request.form.get("imap_email")
    app_pass = request.form.get("imap_pass")

    try:  
        mail = imaplib.IMAP4_SSL("imap.gmail.com")  
        mail.login(email_addr, app_pass)  
        mail.select("inbox")  

        _, messages = mail.search(None, 'ALL')  
        ids = messages[0].split()  
          
        if not ids:  
            return jsonify({"status": "empty", "message": "No emails found in inbox."})  

        latest_ids = ids[-5:]  
        results_list = []  

        for num in reversed(latest_ids):  
            _, data = mail.fetch(num, "(RFC822)")  
            msg = email.message_from_bytes(data[0][1])  
            subject = msg["subject"] or "(No Subject)"  
            sender = msg.get("From", "Unknown Sender")

            body = ""  
            if msg.is_multipart():  
                for part in msg.walk():  
                    if part.get_content_type() == "text/plain":  
                        body = part.get_payload(decode=True).decode(errors='ignore')  
                        break  
            else:  
                body = msg.get_payload(decode=True).decode(errors='ignore')  

            res, conf, reasons = check_spam(body)  
              
            # Included sender and full_body for the new viewer UI
            results_list.append({  
                "subject": subject,  
                "sender": sender,
                "result": res,  
                "confidence": conf,  
                "body_snippet": body[:120] + "...",  
                "full_body": body,
                "reasons": reasons,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })  

        mail.logout()  
        return jsonify({"status": "success", "emails": results_list})  

    except imaplib.IMAP4.error:
        return jsonify({"status": "error", "message": "Invalid Email or App Password. Please check your credentials."})
    except Exception as e:  
        return jsonify({"status": "error", "message": f"Connection Error: {str(e)}"})

@app.route("/api/start_background", methods=["POST"])
def start_background():
    global bg_fetch_active, bg_credentials, global_inbox, processed_email_ids
    data = request.json
    email_addr = data.get("email")
    app_pass = data.get("password")
    
    # Strictly validate credentials before starting
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_addr, app_pass)
        mail.logout()
    except imaplib.IMAP4.error:
        return jsonify({"status": "error", "message": "Invalid Email or App Password. Access denied."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Network error: {str(e)}"})

    # If valid, start background process
    bg_credentials["email"] = email_addr
    bg_credentials["password"] = app_pass
    bg_fetch_active = True
    global_inbox = [] # Clear old session data
    processed_email_ids = set()
    
    return jsonify({"status": "success", "message": "Background fetching started."})

@app.route("/api/stop_background", methods=["POST"])
def stop_background():
    global bg_fetch_active
    bg_fetch_active = False
    return jsonify({"status": "success", "message": "Background fetching stopped."})

@app.route("/api/get_inbox", methods=["GET"])
def get_inbox():
    return jsonify({"status": "success", "emails": global_inbox, "is_active": bg_fetch_active})

@app.route("/api/dashboard_stats", methods=["GET"])
def dashboard_stats():
    total = len(global_inbox)
    spam = sum(1 for e in global_inbox if e["result"] == "SPAM")
    suspicious = sum(1 for e in global_inbox if e["result"] == "SUSPICIOUS")
    safe = sum(1 for e in global_inbox if e["result"] == "SAFE")
    
    return jsonify({
        "total": total,
        "spam": spam,
        "suspicious": suspicious,
        "safe": safe
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


