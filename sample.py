import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Function to load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "distilbert-base-uncased"  # You can change this to any Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Function to perform text transformation
def transform_text(text, tokenizer, model):
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    result = classifier(text)
    return result[0]['label'], result[0]['score']

# Streamlit app
def main():
    st.title("Hugging Face Transformers Text Transformation App")
    
    # Load model and tokenizer
    tokenizer, model = load_model()

    # Text input
    text = st.text_area("Enter text for transformation:")

    if st.button("Transform"):
        if text:
            st.write("Transforming text...")

            # Perform text transformation
            label, score = transform_text(text, tokenizer, model)

            # Display the result
            st.write(f"Transformed Sentiment: {label}")
            st.write(f"Confidence Score: {score}")

if __name__ == "__main__":
    main()
