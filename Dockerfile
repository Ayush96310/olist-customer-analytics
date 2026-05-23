FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY churn_model.pkl .
COPY encoders.pkl .
COPY cat_values.json .
COPY window_lookup.json .

EXPOSE 5000

CMD ["python", "api.py"]