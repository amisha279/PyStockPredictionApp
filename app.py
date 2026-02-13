#from flask import Flask
#app = Flask(__name__)
#@app.route('/')
#def hello_world():
    #return 'Hello World!'

#from flask import Flask, render_template, request, jsonify
#app = Flask(__name__)
#@app.route("/")
#def chat_page():
#    return render_template("chat.html")

#@app.route("/ask", methods=["POST"])
#def ask():
#    user_msg = request.json.get("message", "")

    # Your logic here — for now, return a simple response
#    bot_reply = f"You said: {user_msg}"

#    return jsonify({"reply": bot_reply})

#if __name__ == "__main__":
#    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import joblib
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# Load vectorizer + model
vectorizer, clf = joblib.load("model.pkl")
model = joblib.load("stock_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")
ticker_categories = joblib.load("ticker_categories.pkl")

def build_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
    return df.dropna()

def get_features(ticker):
    df = yf.download(ticker, period="60d", interval="1d")
    df = build_features(df)

    if df.empty or len(df) < 5:
        return jsonify({"reply": f"Not enough data to build features for {ticker}"}), 400

    # Encode ticker using same categories as training
    if ticker in ticker_categories:
        ticker_code = ticker_categories.index(ticker)
    else:
        ticker_code = -1  # unseen ticker

    df["TickerCode"] = ticker_code
    return df[feature_cols].iloc[-1:].values

def predict_price(ticker):
    features = get_features(ticker)
    return model.predict(features)[0]

@app.route("/")
def chat_page():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_msg = request.json.get("message", "")

    # Transform input and predict
    X = vectorizer.transform([user_msg])
    prediction = clf.predict(X)[0]

    return jsonify({"reply": f"Prediction: {prediction}"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker", "").upper()

    try:
        df = yf.download(ticker, period="60d", interval="1d")

        if df.empty:
            return jsonify({"reply": f"No data found for {ticker}"}), 400

        # FIX: Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = build_features(df)

        if df.empty or len(df) < 5:
            return jsonify({"reply": f"Not enough data to build features for {ticker}"}), 400

        df.index = pd.to_datetime(df.index)

        closes = df["Close"].astype(float).tolist()
        dates = df.index.strftime("%Y-%m-%d").tolist()

        ticker_code = ticker_categories.index(ticker) if ticker in ticker_categories else -1
        df["TickerCode"] = ticker_code

        X_latest = df[feature_cols].iloc[-1:].values
        pred = float(model.predict(X_latest)[0])

        return jsonify({
            "reply": f"Predicted next-day price for {ticker}: {pred:.2f}",
            "ticker": ticker,
            "dates": dates,
            "closes": closes,
            "predicted": pred
        })

    except Exception as e:
        import traceback
        print("ERROR IN /predict ROUTE:")
        traceback.print_exc()
        return jsonify({"reply": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)

