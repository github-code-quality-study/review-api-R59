import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')
sorted_reviews = sorted(reviews, key=lambda x: datetime.strptime(x['Timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True)

cities = [
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona"
]

def get_sentiment(review_str):
    words = word_tokenize(review_str)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    processed_text = ' '.join(filtered_words)
    scores = sia.polarity_scores(processed_text)
    return {
        "neg": scores['neg'],
        "neu": scores['neu'],
        "pos": scores['pos'],
        "compound": scores['compound']
    }

def format_reviews(filtered_reviews):
    formated_reviews = []
    for review in filtered_reviews:
        formated_reviews.append({
        "ReviewId": review['ReviewId'],
        "ReviewBody": review['ReviewBody'],
        "Location": review['Location'],
        "Timestamp": review['Timestamp'],
        "sentiment": get_sentiment(review['ReviewBody'])
    })
    return formated_reviews

def filter_by_date(start_date, end_date, location=None):
    filtered_reviews_loc = sorted_reviews
    filtered_reviews = []
    if location:
        filtered_reviews_loc = get_only_location_reviews(location)

    if start_date and end_date:
        filtered_reviews = [
            review for review in filtered_reviews_loc
            if start_date <= datetime.strptime(review.get('Timestamp'), '%Y-%m-%d %H:%M:%S') <= end_date
        ]
    elif start_date:
        filtered_reviews = [
            review for review in filtered_reviews_loc
            if start_date <= datetime.strptime(review.get('Timestamp'), '%Y-%m-%d %H:%M:%S')
        ]        
    elif end_date:
        filtered_reviews = [
            review for review in filtered_reviews_loc
            if datetime.strptime(review.get('Timestamp'), '%Y-%m-%d %H:%M:%S') <= end_date
        ]
    else:
        return filtered_reviews_loc

    return filtered_reviews

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # Attempt to parse with a fallback format (e.g., without time)
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            # Default to start of the day
            return dt
        except ValueError:
            # Handle cases with incorrect formats
            raise ValueError(f"Date format for '{date_str}' is invalid")
        
def add_default_times(start_date_str, end_date_str):
    start_date = None
    end_date = None
    try:
        # Try parsing both start and end dates
        if start_date_str:
            start_date = parse_date(str(start_date_str))
            if start_date.hour == 0 and start_date.minute == 0 and start_date.second == 0:
                start_date = start_date.replace(hour=0, minute=0, second=0)
        
        if end_date_str:
            end_date = parse_date(str(end_date_str))
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0:
                end_date = end_date.replace(hour=23, minute=59, second=59)
        
        return start_date, end_date
    
    except ValueError as e:
        print(e)
        return None, None
    
def get_only_location_reviews(location):
    filtered_reviews = [
        review for review in sorted_reviews
        if review.get('Location') and all(keyword.lower() in review.get('Location').lower() for keyword in location.split())
    ]
    return filtered_reviews

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")
            
            # Write your code here
            params = parse_qs(environ.get('QUERY_STRING'))
            start_date = params.get('start_date')[0] if params.get('start_date') else None
            if start_date:
                start_date = parse_date(start_date)
            end_date = params.get('end_date')[0] if params.get('end_date') else None
            if end_date:
                end_date = parse_date(end_date)

            location = params.get('location')[0] if params.get('location') else None

            start_date, end_date = add_default_times(start_date, end_date)
            filtered_reviews = filter_by_date(start_date, end_date, location)
            formated_reviews = format_reviews(filtered_reviews)
            response_body = json.dumps(formated_reviews).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            content_length = int(environ.get('CONTENT_LENGTH', 0))
            post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
            parsed_data = parse_qs(post_data)
            
            location = parsed_data.get('Location', [''])[0]
            if not location:
                response_data = {
                    "message": "'missing location"
                }
                response_body = json.dumps(response_data)
                status = '400 Bad Request'
                headers = [('Content-type', 'application/json; charset=utf-8')]
                start_response(status, headers)
                return [response_body.encode('utf-8')]
            
            elif location not in cities:
                response_data = {
                    "message": "'invalid location"
                }
                response_body = json.dumps(response_data)
                status = '400 Bad Request'
                headers = [('Content-type', 'application/json; charset=utf-8')]
                start_response(status, headers)
                return [response_body.encode('utf-8')]
            
            review_body = parsed_data.get('ReviewBody', [''])[0]

            if not review_body:
                response_data = {
                    "message": "'No review body"
                }
                response_body = json.dumps(response_data)
                status = '400 Bad Request'
                headers = [('Content-type', 'application/json; charset=utf-8')]
                start_response(status, headers)
                return [response_body.encode('utf-8')]
            
            ReviewId = str(uuid.uuid4())
            TimeStamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            response_data = {
                "ReviewId": ReviewId,
                "Location": location,
                "Timestamp": str(TimeStamp),
                "ReviewBody": review_body
            }
            response_body = json.dumps(response_data)
            status = '201 OK'
            headers = [('Content-type', 'application/json; charset=utf-8')]
            start_response(status, headers)

            return [response_body.encode('utf-8')]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()