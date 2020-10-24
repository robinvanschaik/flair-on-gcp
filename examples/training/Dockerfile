# Dockerfile
# Uses pytorch + CUDA + Apex as a base image.
FROM pytorchignite/apex:1.6.0-0.4.2
WORKDIR /root

# Install git.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

# In production you should pin dependencies.
# Installs flair, transformers & huggingface transformers,
# and Google Cloud Storage.
RUN  pip install --no-cache-dir flair==0.6.1 && \
     pip install --no-cache-dir transformers==3.3.1 && \
     pip install --no-cache-dir sentence-transformers==0.3.8 && \
     pip install --no-cache-dir google-cloud-storage==1.31.0 && \
     pip install --no-cache-dir pandas==1.1.3 && \
     pip install --no-cache-dir numpy==1.19.2 && \
     pip install --no-cache-dir scikit-learn==0.23.2

# Copies the trainer code
# Creates data folder
# Creates a checkpoint folder which will contain the training results.
# /root/text-classification
#         ├── training_orchestrator.py
#         ├── data
#         │     ├── test.csv (optional)
#         │     ├── dev.csv (optional)
#         │     └── train.csv
#         ├── checkpoint
#         │     ├── final-model.pt
#         │     ├── best-model.pt
#         │     └── training.log
#         ├── trainer
#         │     ├── text_classification_trainer.py
#         │     └── trainer_utils.py
#         └── training_io
#                └──  file_utils.py

# Create the directories.
RUN mkdir /root/text-classification && \
    mkdir /root/text-classification/data && \
    mkdir /root/text-classification/checkpoint

# Copies all the prerequisite files to the container.
COPY text-classification /root/text-classification/

# Sets up the entry point to invoke the training orchestrator.
ENTRYPOINT ["python", "text-classification/training_orchestrator.py"]
