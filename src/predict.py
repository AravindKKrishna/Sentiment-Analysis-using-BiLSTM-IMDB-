import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("models/sentiment_model.h5")

tokenizer = pickle.load(open("models/tokenizer.pkl","rb"))

def predict_sentiment(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(seq,maxlen=200)

    pred = model.predict(padded)

    if pred[0][0] > 0.5:
        return "Positive"
    else:
        return "Negative"