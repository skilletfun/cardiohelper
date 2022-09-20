from rest_framework.response import Response
from rest_framework.views import APIView
from predict.converter import main


class PredictAPIView(APIView):
    def post(self, request):
        try:
            data = request.data["image"].replace(' ', '+')
            res = main(data)

            return Response({
                'prediction': res
            })
        except Exception as e:
            return Response(f'error: {e}')
