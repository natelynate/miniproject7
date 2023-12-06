from django.shortcuts import render
from django.utils import timezone
import logging
from django.conf import settings
from django.core.files.storage import default_storage
import numpy as np
import cv2
import string
import mlflow
import mlflow.keras
from chatgpt.views import chatGPT
logger = logging.getLogger('mylogger')
#signlanguage/models.py의 Result 모델을 import한다.
from .models import ChatResult, Result
# model_path = './mlruns/3/9c40519238064b8287170056a49c1568/artifacts/model'
# model = mlflow.keras.load_model(model_path)

mlflow_uri = "http://mlflow.carpediem.so"
mlflow.set_tracking_uri(mlflow_uri)
model_uri = "models:/model_06/1" 
model_path = "models:/model_06/latest"
model = mlflow.keras.load_model(model_uri)
# Create your views here.

'''
1. 원칙은 ORM을 사용하여 별도 sql 문이 없는 것이다.
2. 하지만, ORM을 사용하면서도 sql문을 사용해야 하는 경우가 있다.
3. 이때는 아래와 같이 사용한다.
 - 물론 이 부분도 view가 sql을 알면 안되서 분리해야 하지만, 짧은 교육상 이곳에 둔다. 
'''
def getChatResult(self, id):
        query = "SELECT * FROM signlanguagetochatgpt_chatresult WHERE id = {0}".format(id)
        logger.info(">>>>>>>> getChatResult SQL : "+query)
        chatResult = self.t_exec(query)

def index(request):
    return render(request, 'languagechat/index.html')

def load_and_preprocess_image(file):
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (28, 28))

    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0

    return image

class_names = [chr(i) for i in range(ord('a'), ord('z') + 1)]

def predict_letter(file):
    processed_image = load_and_preprocess_image(file)
    pred_probs = model.predict(processed_image)
    pred_index = np.argmax(pred_probs[0])
    predicted_letter = class_names[pred_index]

    return predicted_letter

def chat(request):
    if request.method == 'POST':
        files = request.FILES.getlist('files[]')
        if not files:
            return render(request, 'languagechat/error.html', {'error': '파일이 업로드되지 않았습니다.'})

        results = []
        chatGptPrompt = ""
        
        for idx, file in enumerate(files, start=0):
            try:
                # 파일 처리 및 모델 예측
                letter = predict_letter(file)
                results.append(letter)
                chatGptPrompt += letter

            except Exception as e:
                logger.error(f"파일 처리 중 오류 발생: {e}")
                return render(request, 'languagechat/error.html', {'error': '파일 처리 중 오류가 발생했습니다.'})

        # ChatGPT 대화 처리
        try:
            chatResult = ChatResult(prompt=chatGptPrompt, pub_date=timezone.now())
            chatResult.save()

            content = chatGPT(chatResult.prompt)
            chatResult.content = content
            chatResult.save()

            context = {
                'question': chatResult.prompt,
                'result': chatResult.content,
                'predictions': results  # 예측 결과 추가
            }
        except Exception as e:
            logger.error(f"ChatGPT 처리 중 오류 발생: {e}")
            return render(request, 'languagechat/error.html', {'error': 'ChatGPT 처리 중 오류가 발생했습니다.'})

        return render(request, 'languagechat/result.html', context)
    else:
        return render(request, 'languagechat/index.html')