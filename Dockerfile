#python image
FROM python:3.13-slim

#working directory
WORKDIR /app

#install requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#src folder copy
COPY src ./src

COPY models ./models

#port application runs on
EXPOSE 8080 

#command to run
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]

