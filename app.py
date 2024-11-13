from flask import Flask, render_template, request
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", top_k=None)

def analyze_sentiment(text):
    result = sentiment_pipeline(text)  # Get result as a list of dictionaries
    print("Sentiment analysis result:", result)  # Output to inspect the result structure
    
    # Initialize sentiment confidence values
    positive_confidence = 0
    negative_confidence = 0
    neutral_confidence = 0
    
    # The expected labels are 'LABEL_0' (negative), 'LABEL_1' (neutral), and 'LABEL_2' (positive)
    for sentiment in result[0]:
        print(f"Sentiment: {sentiment}")  # Debugging: print individual sentiment dictionary
        label = sentiment.get('label')  # LABEL_0, LABEL_1, or LABEL_2
        score = sentiment.get('score')  # Confidence score for the label
        
        if label == 'LABEL_2':  # Positive sentiment
            positive_confidence = score * 100
        elif label == 'LABEL_1':  # Neutral sentiment
            neutral_confidence = score * 100
        elif label == 'LABEL_0':  # Negative sentiment
            negative_confidence = score * 100
    
    # Store all sentiment confidence values in a dictionary
    sentiment_confidences = {
        'Positive': positive_confidence,
        'Negative': negative_confidence,
        'Neutral': neutral_confidence
    }
    
    # Find the sentiment label with the highest confidence
    highest_label = max(sentiment_confidences, key=sentiment_confidences.get)  # Get label with highest confidence
    
    # Return the dictionary of all sentiment confidences and the highest sentiment label
    return sentiment_confidences, highest_label

# Load summarization model
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    # Summarization returns a list of dictionaries, so we need to extract the 'summary_text' from the first element
    summary = summarization_pipeline(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']  # Extract summary text from the dictionary

# Load question answering model for keyword extraction
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def extract_keywords(text):
    questions = ["What is the main topic?", ]
    keywords = set()

    for question in questions:
        result = qa_model(question=question, context=text)
        keywords.add(result['answer'])

    return list(keywords)

# Define the route to process customer query
@app.route("/", methods=["GET", "POST"])
def process_query():
    result = None
    if request.method == "POST":
        query = request.form["query"]
        
        # Perform sentiment analysis, summarization, and keyword extraction
        sentiment_confidences, highest_label = analyze_sentiment(query)
        summary = summarize_text(query)
        keywords = extract_keywords(query)
        
        result = {
            "original_query": query,
            "sentiment_confidences": sentiment_confidences,  # Send the dictionary of all sentiment confidences
            "label": highest_label,
            "summary": summary,
            "keywords": keywords
        }
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
