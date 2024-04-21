from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.views.decorators.csrf import csrf_exempt

nltk.download('punkt')
nltk.download('stopwords')


# NLP pkgs
import numpy as np
import pandas as pd
# Utils
import joblib

model = joblib.load(open('ml/models/spam_detection_new.pkl', 'rb'))
count_vector = joblib.load(open('ml/models/count_vector.pkl', 'rb'))
pipe_lr = joblib.load(open('ml/models/emotion_pipe_lr.pkl', 'rb'))
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}
# Create your views here.

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def prediction_probability(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def predict_spam(docs):
    # count_vector = CountVectorizer()
    docs_vector = count_vector.transform([docs])
    print(docs_vector)
    results = model.predict(docs_vector)
    return results

def extract_keywords(text, num_keywords=20):
    num_keywords = 0
    print(len(text))
    if len(text) > 1000 :
        num_keywords = 35
    elif len(text) <= 1000 and len(text) > 600:
        num_keywords = len(text) //25
    elif len(text) <= 600 and len(text) > 100:
        num_keywords = len(text) //20
    else:
        num_keywords = 5
    print(num_keywords)
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Remove stopwords (common words that usually don't carry much meaning)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Calculate word frequencies
    freq_dist = FreqDist(filtered_words)

    # Get the most common words as keywords
    keywords = [word for word, freq in freq_dist.most_common(num_keywords)]

    return keywords

def index(request):
    return render(request, "evika/index.html")

@api_view(["POST"])
def get_keywords(request):
    if(request.data["text"] == None or request.data["text"] == ""):
        return Response({ "message" : "text is required"},status=status.HTTP_400_BAD_REQUEST) 
    try:
        text = request.data["text"]
        # r = Rake()
        # r.extract_keywords_from_text(text)
        # keywords = r.get_ranked_phrases()
        keywords = extract_keywords(text)
    except:
        return Response({"message": "something went wrong"},status=status.HTTP_400_BAD_REQUEST)
    
    return Response({"data" : keywords},status=status.HTTP_200_OK )


@api_view(["POST"])
def detect_sapm(request):
    print(request.data["text"])
    if(request.data["text"] == None or request.data["text"] == ""):
        return Response({ "message" : "text is required"},status=status.HTTP_400_BAD_REQUEST) 
    try:
        text = request.data["text"]
        result = predict_spam(text)
        print(result)
    except:
        return Response({"message": "something went wrong"},status=status.HTTP_400_BAD_REQUEST)
    
    return Response({"data" : result},status=status.HTTP_200_OK )

@api_view(["POST"])
def get_posts_by_user_preferences(request):
    if(request.data["posts"] == None or len(request.data["posts"]) == 0):
        return Response({ "message" : "posts is required"},status=status.HTTP_400_BAD_REQUEST)
    if(request.data["preferences"] == None or len(request.data["preferences"]) == 0):
        return Response({ "message" : "preferences is required"},status=status.HTTP_400_BAD_REQUEST)
    try:
        posts = request.data["posts"]
        post_descriptions = posts
        preferences = request.data["preferences"]

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        post_vectors = vectorizer.fit_transform(post_descriptions)
        user_profile_vector = vectorizer.transform([" ".join(preferences)])

        # Calculate Cosine Similarity
        cosine_similarities = cosine_similarity(user_profile_vector, post_vectors).flatten()
        print("cosine_similarities", cosine_similarities)
        # Rank and Recommend
        # post_ranking = [(post, score) for post, score in zip(posts, cosine_similarities)]
        # post_ranking.sort(key=lambda x: x[1], reverse=True)

        # postList = [ post[0] for post in post_ranking]
       
    except Exception as e:
        print("error ", e)
        return Response({"message": "something went wrong"},status=status.HTTP_400_BAD_REQUEST)
    
    return Response(cosine_similarities,status=status.HTTP_200_OK )


import json
@csrf_exempt
@api_view(["POST"])
def get_emotion(request):
    data =  json.loads(request.body)
    if(data["text"] == ""):
        data["text"] = "This is demo text netural"
    print(data["text"])
    if(data["text"] == None or data["text"] == ""):
        return Response({ "message" : "text is required"},status=status.HTTP_400_BAD_REQUEST) 
    try:
        text = data["text"]
        emotion = predict_emotion(text)
        probability = prediction_probability(text)
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
    except:
        return Response({"message": "something went wrong"},status=status.HTTP_400_BAD_REQUEST)
    
    return Response({"data" : { "emotion": emotion, "probability": probability, "probability_df" : proba_df}},status=status.HTTP_200_OK )