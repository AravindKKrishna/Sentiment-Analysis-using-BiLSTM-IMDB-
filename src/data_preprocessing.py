import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

# Load stopwords once
stop_words = set(stopwords.words('english'))


def clean_text(text):
    
    # convert to lowercase
    text = text.lower()

    # remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # split into words
    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    # join words again
    return " ".join(words)


# dataset path
path = r"C:\Users\anant\Desktop\sentiment_analysis\data\IMDB Dataset.csv"


def load_data(path):

    # read dataset
    df = pd.read_csv(path)

    # convert sentiment to numeric
    df['sentiment'] = df['sentiment'].map({
        'positive': 1,
        'negative': 0
    })

    # clean reviews
    df['review'] = df['review'].apply(clean_text)

    return df


# run program
if __name__ == "__main__":

    df = load_data(path)

    print(df.head())