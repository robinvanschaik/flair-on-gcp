# Dockerfile
# Uses python 3.8 as a base image.
FROM python:3.8
WORKDIR /root

# # In production you want to pin dependencies.
# Installs flair and sentence-transformers.
RUN pip install -U pip install -U flair sentence_transformers google-cloud-storage

# Copies the trainer code
# Creates data folder
# Creates a checkpoint folder which will contain the training results.
# /root/trainer
#         ├── text-classification-training.py
#         ├── data
#         |     ├── test.csv (optional)
#         |     ├── dev.csv (optional)
#         │     └── train.csv
#         └─- checkpoint
#               ├── final-model.pt
#               ├── best-model.pt
#               └── training.log

# Create the directories
RUN mkdir /root/trainer && \
    mkdir /root/trainer/data && \
    mkdir /root/trainer/checkpoint

# Copies the text-classification-training script to the trainer directory.
COPY trainer/text-classification-training.py /root/trainer/text-classification-training.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "trainer/text-classification-training.py"]
