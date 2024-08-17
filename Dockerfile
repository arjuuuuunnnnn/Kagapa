FROM python:3.9-slim

WORKDIR /app

RUN pip install virtualenv && \
	apt-get update && \
    apt-get install -y sox libsox-dev libsox-fmt-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN virtualenv /env_main
COPY requirements.txt requirements.txt
RUN /env_main/bin/pip install --no-cache-dir -r requirements.txt

RUN virtualenv /env_model4
COPY requirements1.txt requirements4.txt
RUN /model2_env/bin/pip install --no-cache-dir -r requirements4.txt

COPY . .

EXPOSE 80

CMD ["python", "main.py"]

# build command
# docker build -t kagapa .


# To match and get the artifacts out of the container
# docker run -v $(pwd)/artifacts:/app/artifacts my_model_pipeline

