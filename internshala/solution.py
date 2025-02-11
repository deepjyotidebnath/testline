import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# API Endpoints
QUIZ_ENDPOINT = "https://jsonkeeper.com/b/LLQT"
HISTORICAL_ENDPOINT = "https://api.jsonserve.com/XgAgFJ"

# Fetch Data
def fetch_data(url, params=None):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data from {url}")
        return None

# Analyze Data
def analyze_performance(current_quiz, historical_data):
    topic_accuracy = {}
    difficulty_accuracy = {}
    
    for quiz in historical_data:
        for q_id, selected_option in quiz['response_map'].items():
            topic = quiz['questions'][q_id]['topic']
            difficulty = quiz['questions'][q_id]['difficulty']
            correct = quiz['questions'][q_id]['correct_option'] == selected_option
            
            if topic not in topic_accuracy:
                topic_accuracy[topic] = []
            topic_accuracy[topic].append(correct)
            
            if difficulty not in difficulty_accuracy:
                difficulty_accuracy[difficulty] = []
            difficulty_accuracy[difficulty].append(correct)
    
    # Compute average accuracy
    topic_accuracy = {k: np.mean(v) * 100 for k, v in topic_accuracy.items()}
    difficulty_accuracy = {k: np.mean(v) * 100 for k, v in difficulty_accuracy.items()}
    
    return topic_accuracy, difficulty_accuracy

# Generate Recommendations
def generate_recommendations(topic_accuracy, difficulty_accuracy):
    recommendations = []
    
    for topic, accuracy in topic_accuracy.items():
        if accuracy < 50:
            recommendations.append(f"Focus on {topic}, as accuracy is {accuracy:.2f}%")
    
    for difficulty, accuracy in difficulty_accuracy.items():
        if accuracy < 50:
            recommendations.append(f"Practice more {difficulty} level questions ({accuracy:.2f}% accuracy)")
    
    return recommendations

# Predict NEET Rank
def predict_neet_rank(historical_data):
    scores = [quiz['score'] for quiz in historical_data]
    ranks = [quiz['neet_rank'] for quiz in historical_data]
    
    model = LinearRegression()
    model.fit(np.array(scores).reshape(-1, 1), ranks)
    predicted_rank = model.predict([[scores[-1]]])[0]
    
    return predicted_rank

# Main Execution
if __name__ == "__main__":
    current_quiz = fetch_data(QUIZ_ENDPOINT)
    historical_data = fetch_data(HISTORICAL_ENDPOINT)
    
    if current_quiz and historical_data:
        topic_acc, diff_acc = analyze_performance(current_quiz, historical_data)
        recommendations = generate_recommendations(topic_acc, diff_acc)
        predicted_rank = predict_neet_rank(historical_data)
        
        print("\nPerformance Insights:")
        print("Topic-wise Accuracy:", topic_acc)
        print("Difficulty-wise Accuracy:", diff_acc)
        
        print("\nPersonalized Recommendations:")
        for rec in recommendations:
            print("-", rec)
        
        print(f"\nPredicted NEET Rank: {predicted_rank:.0f}")
