import gradio as gr
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the necessary models and tokenizer
def load_models():
    # Load tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load the trained model
    model = load_model('model_rnn.h5')
    
    return tokenizer, model

def cleansing(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def predict_sentiment(text, tokenizer, model):
    # Preprocess text
    cleaned_text = [cleansing(text)]
    
    # List of negative keywords in Indonesian
    negative_keywords = [
        'buruk', 'lambat', 'jelek', 'tidak enak', 'kotor', 'mahal', 
        'kecewa', 'mengecewakan', 'parah', 'busuk', 'keras', 'dingin',
        'hambar', 'asin', 'pahit', 'tidak ramah', 'kasar', 'jorok'
    ]
    
    # List of neutral keywords and phrases in Indonesian
    neutral_keywords = [
        'biasa', 'biasa saja', 'standar', 'lumayan', 'cukup',
        'tidak istimewa', 'sedang', 'rata rata', 'rata-rata',
        'begitu begitu saja', 'seperti biasa', 'normal',
        'tidak ada yang spesial', 'tidak spesial', 'b aja',
        'masih ok', 'bisa lah', 'ya begitulah'
    ]
    
    # Convert to sequences
    predicted = tokenizer.texts_to_sequences(cleaned_text)
    
    # Pad sequences
    maxlen = 100  # Make sure this matches your training padding
    guess = pad_sequences(predicted, maxlen=maxlen)
    
    # Make prediction
    prediction = model.predict(guess)
    confidence_scores = prediction[0]
    
    # Check for keywords
    text_lower = cleaned_text[0].lower()
    contains_negative = any(keyword in text_lower for keyword in negative_keywords)
    contains_neutral = any(keyword in text_lower for keyword in neutral_keywords)
    
    # Adjust prediction based on keywords
    if contains_negative:
        # Increase negative confidence and decrease others
        confidence_scores[0] = max(confidence_scores[0], 0.6)  # Negative
        confidence_scores[1] = min(confidence_scores[1], 0.3)  # Neutral
        confidence_scores[2] = min(confidence_scores[2], 0.2)  # Positive
    elif contains_neutral:
        # Increase neutral confidence and adjust others
        confidence_scores[0] = min(confidence_scores[0], 0.3)  # Negative
        confidence_scores[1] = max(confidence_scores[1], 0.6)  # Neutral
        confidence_scores[2] = min(confidence_scores[2], 0.3)  # Positive
    
    # Get final polarity
    polarity = np.argmax(confidence_scores)
    
    # Map sentiment labels
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    result = sentiment_labels[polarity]
    
    # Format confidence scores
    confidence_str = f"""
    Confidence Scores:
    Negative: {confidence_scores[0]:.2%}
    Neutral: {confidence_scores[1]:.2%}
    Positive: {confidence_scores[2]:.2%}
    """
    
    return result, confidence_str

# Load models
tokenizer, model = load_models()

# Create Gradio interface
def gradio_interface(text):
    sentiment, confidence = predict_sentiment(text, tokenizer, model)
    return f"Sentiment: {sentiment}\n\n{confidence}"

# Configure the interface with more examples
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Analyze the sentiment",
    examples=[
        ["Makanan ini sangat enak dan lezat!"],
        ["Pelayanannya sangat buruk dan lambat."],
        ["Tempatnya biasa saja, tidak istimewa."],
        ["Rasanya lumayan, harganya standar."],
        ["Makanannya cukup enak, tapi tidak spesial."],
        ["Pelayanannya rata-rata, seperti restoran pada umumnya."]
    ]
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True) 