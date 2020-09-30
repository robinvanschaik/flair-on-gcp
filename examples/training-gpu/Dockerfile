# Dockerfile
# Uses pytorch + CUDA + Apex as a base image.
FROM pytorchignite/apex:latest
WORKDIR /root

# Install git.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

# In production you want to pin dependencies.
# Installs flair and sentence-transformers.
RUN pip install --no-cache-dir -U pip install git+https://github.com/flairNLP/flair.git sentence_transformers google-cloud-storage

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
