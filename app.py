import streamlit as st
import joblib


# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Page configuration
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)


# Title
st.title("📧 Email Spam Detection")
st.write(
    "Enter an email or message below to determine whether it is "
    "Spam or Not Spam."
)

st.divider()


# Message input
message = st.text_area(
    "✉️ Enter your message",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize..."
)


# Prediction
if st.button("🔍 Check Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        # Convert message to TF-IDF
        message_tfidf = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_tfidf)[0]

        # Probability
        probabilities = model.predict_proba(message_tfidf)[0]
        spam_probability = probabilities[
            list(model.classes_).index("spam")
        ]

        st.divider()

        # Result
        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM MESSAGE")

        # Probability
        st.subheader("Prediction Probability")
        st.progress(float(spam_probability))

        st.write(
            f"Spam Probability: **{spam_probability * 100:.2f}%**"
        )

        # Message summary
        st.subheader("📝 Message Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Characters", len(message))

        with col2:
            st.metric("Words", len(message.split()))

        st.write("**Entered Message:**")
        st.info(message)
        