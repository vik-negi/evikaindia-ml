from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')


# Create your views here.

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
    if(request.data["text"] == "" and request.data["text"] == None):
        return Response({ "message" : "text is required"},status=status.HTTP_400_BAD_REQUEST) 
    if(request.data["user_id"] =="" and request.data["user_id"] != None):
        return Response({"message" : "user_id is required"},status=status.HTTP_400_BAD_REQUEST) 
    try:
        text = request.data["text"]
        # r = Rake()
        # r.extract_keywords_from_text(text)
        # keywords = r.get_ranked_phrases()
        keywords = extract_keywords(text)
    except:
        return Response({"message": "something went wrong"},status=status.HTTP_400_BAD_REQUEST)
    
    return Response({"data" : keywords},status=status.HTTP_200_OK )
