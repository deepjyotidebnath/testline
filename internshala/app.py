import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# API Endpoints
QUIZ_ENDPOINT = "https://jsonkeeper.com/b/LLQT"
HISTORICAL_ENDPOINT = "hhttps://api.jsonserve.com/XgAgFJ"

# Fetch Data
def fetch_data(url, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Analyze Performance
def analyze_performance(historical_data):
    topic_accuracy = {}
    difficulty_accuracy = {}
    
    for quiz in historical_data:
        for q_id, selected_option in quiz['response_map'].items():
            question = quiz['questions'].get(q_id, {})
            topic = question.get('topic', 'Unknown')
            difficulty = question.get('difficulty', 'Unknown')
            correct = question.get('correct_option') == selected_option
            
            topic_accuracy.setdefault(topic, []).append(correct)
            difficulty_accuracy.setdefault(difficulty, []).append(correct)
    
    topic_accuracy = {k: np.mean(v) * 100 for k, v in topic_accuracy.items()}
    difficulty_accuracy = {k: np.mean(v) * 100 for k, v in difficulty_accuracy.items()}
    
    return topic_accuracy, difficulty_accuracy

# Generate Recommendations
def generate_recommendations(topic_accuracy, difficulty_accuracy):
    recommendations = []
    
    for topic, accuracy in topic_accuracy.items():
        if accuracy < 50:
            recommendations.append(f"Improve {topic} (accuracy: {accuracy:.2f}%)")
    
    for difficulty, accuracy in difficulty_accuracy.items():
        if accuracy < 50:
            recommendations.append(f"Practice more {difficulty} level questions (accuracy: {accuracy:.2f}%)")
    
    return recommendations

# Predict NEET Rank
def predict_neet_rank(historical_data):
    scores = [quiz['score'] for quiz in historical_data]
    ranks = [quiz['neet_rank'] for quiz in historical_data]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(np.array(scores).reshape(-1, 1), ranks)
    predicted_rank = model.predict([[scores[-1]]])[0]
    
    return predicted_rank

# Main Execution
if __name__ == "__main__":
    historical_data = fetch_data(HISTORICAL_ENDPOINT)
    
    if historical_data:
        topic_acc, diff_acc = analyze_performance(historical_data)
        recommendations = generate_recommendations(topic_acc, diff_acc)
        predicted_rank = predict_neet_rank(historical_data)
        
        print("\nPerformance Insights:")
        print("Topic-wise Accuracy:", topic_acc)
        print("Difficulty-wise Accuracy:", diff_acc)
        
        print("\nPersonalized Recommendations:")
        for rec in recommendations:
            print("-", rec)
        
        print(f"\nPredicted NEET Rank: {predicted_rank:.0f}")
