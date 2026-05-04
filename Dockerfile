# Берём официальный образ Python 3.11 (лёгкая версия slim — меньше размер)
FROM python:3.11-slim

# Устанавливаем рабочую папку внутри контейнера
WORKDIR /app


# Устанавливаем системную библиотеку которую требует LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*


# Копируем файл зависимостей первым — это ускоряет пересборку
# (Docker кэширует этот слой, если requirements.txt не менялся)
COPY requirements.txt .

# Устанавливаем все Python-библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код проекта в контейнер
COPY . .

# Команда запуска при старте контейнера
CMD ["python", "main.py"]