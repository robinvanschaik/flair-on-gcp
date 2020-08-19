# Dockerfile
# Uses python 3.8 with FastAPI pre-installed as a base image.
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
WORKDIR /root

# Expose the set port.
ENV PORT=8080
EXPOSE ${PORT}

# In production you want to pin dependencies.
# Installs flair, sentence_transformers and FastAPI.
RUN pip install -U flair sentence_transformers

# Copies the trainer code.
# Creates data folder.
# Creates a checkpoint folder which will contain the training results.
# /root/
#    ├── app.py
#    └── model
#         └─- best-model.pt

# Create the model directory.
RUN mkdir /root/model

# Copies the FLAIR pytorch model to the model directory.
COPY model/best-model.pt /root/model/best-model.pt
COPY app.py /root/app.py

CMD exec uvicorn --host 0.0.0.0 --port ${PORT} app:app
