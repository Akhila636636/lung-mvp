import streamlit as st
from inference.predict import predict

st.title("🫁 Lung MVP Demo")

uploaded = st.file_uploader("Upload any file")

if uploaded is not None:
    st.write("Running prediction...")

    result = predict(uploaded.name)

    st.success("Prediction complete!")
    st.write("### Probability:", result["probability"])
    st.write("### Report:")
    st.write(result["report"])
