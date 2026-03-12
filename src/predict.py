import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model(r"C:\Users\anant\Desktop\sentiment_analysis\src\models\sentiment_model.h5")

# Load tokenizer
with open(r"C:\Users\anant\Desktop\sentiment_analysis\src\models\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def predict_sentiment(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(seq, maxlen=200)

    pred = model.predict(padded)

    if pred[0][0] > 0.5:
        return "Positive"
    else:
        return "Negative"