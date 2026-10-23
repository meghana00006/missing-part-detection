FROM python:3.12-slim

WORKDIR /app

COPY UI/requirements.txt /app/UI/requirements.txt

RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "pyarrow<15"

RUN pip install --no-cache-dir -r /app/UI/requirements.txt

COPY UI /app/UI
COPY model /app/model
COPY references /app/references
COPY scripts /app/scripts

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "UI/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
