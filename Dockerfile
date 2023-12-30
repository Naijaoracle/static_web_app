FROM python:3.9

WORKDIR /app

COPY src/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

ARG AZURE_BLOB_CONNECTION_STRING

ENV AZURE_BLOB_CONNECTION_STRING=$AZURE_BLOB_CONNECTION_STRING

CMD ["python", "tbcxr_model01.py"]