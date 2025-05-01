import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample FAQs and responses
faq_data = {
    "What is your return policy?": "You can return any item within 30 days of purchase with a receipt.",
    "How can I track my order?": "You can track your order using the tracking number provided in your confirmation email.",
    "Do you offer international shipping?": "Yes, we ship internationally. Shipping fees and delivery times vary by location.",
    "How do I reset my password?": "Click on 'Forgot Password' on the login page and follow the instructions sent to your email.",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and other popular payment methods."
}

# Preprocess the FAQs and responses
faq_questions = list(faq_data.keys())
faq_answers = list(faq_data.values())

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)

def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0, best_match_idx]
    
    if best_match_score > 0.2:  # Confidence threshold
        return faq_answers[best_match_idx]
    else:
        return "I'm sorry, I don't have an answer for that. Can you rephrase?"

# Chat loop
if __name__ == "__main__":
    print("Chatbot: Hello! Ask me a question about our services.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
