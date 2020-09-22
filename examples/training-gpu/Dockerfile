# Dockerfile
# Uses pytorch + CUDA as a base image.
# Is a convenient way use GPU acceleration in Flair.
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /root

# # In production you want to pin dependencies.
# Installs flair and huggingface transformers sentence-transformers.
RUN pip install --no-cache-dir -U pip install flair sentence_transformers google-cloud-storage

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
