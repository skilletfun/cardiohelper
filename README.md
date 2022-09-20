# Как запустить сайт

1) Создать виртуальное окружение `python3 -m venv venv`
2) Активировать виртуальное окружение `source venv/bin/activate` (Linux) `venv/scripts/activate.bat` (Windows)
3) Скачать репозиторий
4) Установить зависимости `pip3 install -r requirements.txt`
5) Запустить сервер `python3 manage.py runserver`

# Инструкция по пользованию
Загрузить на сайт картинку с ЭКГ. Дождаться распознавания и предсказания.

Возможные результаты:
1) N - нормальная форма ЭКГ;
2) A - фибрилляция;
3) O - картинка с шумом, невозможно предсказать;
4) ~ - что-то другое.

# Куда отправляется картинка для предсказания
Картинку отправляется на точку доступа `api/v1/predict` в формате `base64` в атрибуте `image`.
Результат вернется в формате `json`. Предсказание будет доступно по атрибуту `prediction`.
При ошибке вернется `json` с ошибкой, текст которой будет доступен по атрибуту `error`.