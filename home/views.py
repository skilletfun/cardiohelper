from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

def index(request):
    if request.user_agent.is_mobile:
        page = loader.get_template('m_index.html')
    else:
        page = loader.get_template('index.html')

    return HttpResponse(page.render())
