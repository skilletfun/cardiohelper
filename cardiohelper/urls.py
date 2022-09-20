from django.contrib import admin
from django.urls import path, include
from predict.views import PredictAPIView


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/predict', PredictAPIView.as_view()),
    path('home/', include('home.urls'))
]
