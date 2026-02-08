import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import math

st.set_page_config(page_title="Machine Learning Practicals", layout="wide")

st.sidebar.title("Select Practical")
choice = st.sidebar.selectbox(
    "Choose Practical",
    [
        "Practical 1: KNN Weather Classification",
        "Practical 2: ID3 Decision Tree",
        "Practical 3: Spam Email Detection",
        "Practical 4: Sentiment Analysis (IMDB)",
        "Practical 5: Diabetes Progression Prediction",
        "Practical 6: Linear Regression Demo",
        "Practical 7: Titanic Survival Prediction"
    ]
)

# ---------------- PRACTICAL 1 ----------------
def practical_1():
    st.title("KNN Weather Classification")

    X = np.array([[50,70],[25,80],[27,60],[31,65],[23,85],[20,75]])
    y = np.array([0,1,0,0,1,1])
    label_map = {0: "Sunny", 1: "Rainy"}

    temp = st.sidebar.slider("Temperature", 10, 60, 26)
    hum = st.sidebar.slider("Humidity", 50, 95, 78)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    pred = knn.predict([[temp, hum]])[0]
    st.write(f"Predicted Weather: **{label_map[pred]}**")

    fig, ax = plt.subplots()
    ax.scatter(X[y==0,0], X[y==0,1], color="orange", label="Sunny", s=100)
    ax.scatter(X[y==1,0], X[y==1,1], color="blue", label="Rainy", s=100)
    ax.scatter(temp, hum, color="red", marker="*", s=300, label="New Data")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Humidity")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ---------------- PRACTICAL 2 ----------------
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    return -sum((c/len(col)) * math.log2(c/len(col)) for c in counts)

def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals = df[attr].unique()
    weighted = sum(
        (len(df[df[attr]==v]) / len(df)) * entropy(df[df[attr]==v][target])
        for v in vals
    )
    return total_entropy - weighted

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]

    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}

    for val in df[best].unique():
        sub = df[df[best] == val]
        tree[best][val] = id3(sub, target, [a for a in attrs if a != best])
    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    value = sample.get(root)
    return predict(tree[root].get(value, "Unknown"), sample)

def practical_2():
    st.title("ID3 Decision Tree")

    data = {
        "outlook": ["sunny","sunny","overcast","rain","rain","overcast"],
        "humidity": ["high","normal","high","normal","high","high"],
        "playtennis": ["no","yes","yes","yes","no","yes"]
    }
    df = pd.DataFrame(data)

    target = "playtennis"
    features = ["outlook", "humidity"]

    if st.button("Train Model"):
        tree = id3(df, target, features)
        st.session_state.tree = tree
        st.json(tree)

    if "tree" in st.session_state:
        inputs = {f: st.selectbox(f, df[f].unique()) for f in features}
        if st.button("Predict"):
            st.write("Result:", predict(st.session_state.tree, inputs))

# ---------------- PRACTICAL 3 ----------------
def practical_3():
    st.title("Spam Email Detection")

    emails = [
        "Win a free phone now",
        "Meeting at 11 am",
        "Claim your prize",
        "Project discussion",
        "Limited offer buy now"
    ]
    labels = [1,0,1,0,1]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(emails)

    model = LinearSVC()
    model.fit(X, labels)

    msg = st.text_area("Enter Email Text")
    if st.button("Check"):
        pred = model.predict(vectorizer.transform([msg]))[0]
        st.write("Spam" if pred==1 else "Not Spam")

# ---------------- PRACTICAL 4 ----------------
def practical_4():
    st.title("IMDB Sentiment Analysis")

    uploaded = st.file_uploader("Upload IMDB CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df["sentiment"] = df["sentiment"].map({"positive":1,"negative":0})

        X_train, X_test, y_train, y_test = train_test_split(
            df["review"], df["sentiment"], test_size=0.2, random_state=42
        )

        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

# ---------------- PRACTICAL 5 ----------------
def practical_5():
    st.title("Diabetes Progression Prediction")

    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("MSE:", np.mean((y_test - y_pred)**2))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    st.pyplot(fig)

# ---------------- PRACTICAL 6 ----------------
def practical_6():
    st.title("Simple Linear Regression Demo")

    x = np.arange(1, 20)
    y = 2*x + np.random.randn(19)

    model = LinearRegression()
    model.fit(x.reshape(-1,1), y)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, model.predict(x.reshape(-1,1)), color="red")
    st.pyplot(fig)

# ---------------- PRACTICAL 7 ----------------
def practical_7():
    st.title("Titanic Survival Prediction")

    file = st.file_uploader("Upload Titanic CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        df["Age"].fillna(df["Age"].median(), inplace=True)
        df["Fare"].fillna(df["Fare"].median(), inplace=True)
        df = pd.get_dummies(df, columns=["Sex","Embarked"], drop_first=True)

        features = ["Pclass","Age","Fare","Sex_male"]
        X = df[features]
        y = df["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- RUN ----------------
if choice.startswith("Practical 1"):
    practical_1()
elif choice.startswith("Practical 2"):
    practical_2()
elif choice.startswith("Practical 3"):
    practical_3()
elif choice.startswith("Practical 4"):
    practical_4()
elif choice.startswith("Practical 5"):
    practical_5()
elif choice.startswith("Practical 6"):
    practical_6()
elif choice.startswith("Practical 7"):
    practical_7()
