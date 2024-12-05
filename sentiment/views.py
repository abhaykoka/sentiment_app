from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .utils import classify_sentiment

def sentiment_analysis(request):
    result = None
    if request.method == 'POST':
        text = request.POST.get('text')
        result = classify_sentiment(text)
    return render(request, 'home.html', {'result': result})
