# Базовый образ с Python
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все нужные файлы проекта
COPY checker.py Orchestrator.py config.yaml data_quality.json drift.py \
     encoder.py gather.json main.py missing.py monitor.py nn_model.py \
     pipeline.py quality.py shell.py stream.py train_raw.csv database.db ./

# Запускаем pipeline, а после его успешного завершения — main
CMD python pipeline.py && python main.py