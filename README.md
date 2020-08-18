# flair-on-gcp
**Please note: this repository is not affiliated with Google.**

This repository adds examples on how to train [Flair](https://github.com/flairNLP/flair) on [Google Cloud Platform](https://cloud.google.com/) (GCP) [AI Platform](https://cloud.google.com/ai-platform).

It also covers how to deploy a pre-trained Flair-model on GCP [Cloud Run](https://cloud.google.com/run) using [Cloud Build](https://cloud.google.com/cloud-build) and [Cloud Registry](https://cloud.google.com/container-registry) to serve predictions.

**Please note: this repository is only meant as an example, and is WIP.**

---

## Table of contents
* [Getting started](#getting-started)
* [Training guide](#training)
* [Deployment guide](#deployment)

---

## Getting started
**Google Cloud SDK**

It is recommended to install the [Google Cloud SDK](https://cloud.google.com/sdk) so that you can interact with Google Cloud services from your local command line interface (CLI).

You can find instructions to install the Google Cloud SDK [here](https://cloud.google.com/sdk/install).

**Access management**

Please review whether you have proper access rights to the appropriate services by checking the IAM documentation for each service.

To get started:

**Training**
* [AI Platform](https://cloud.google.com/ai-platform/training/docs/access-control)
* [Cloud Storage](https://cloud.google.com/storage/docs/access-control/iam-roles)
* [Logging](https://cloud.google.com/logging/docs/access-control)

**Deployment**
* [Cloud Run](https://cloud.google.com/run/docs/reference/iam/roles)
* [Cloud Build](https://cloud.google.com/cloud-build/docs/iam-roles-permissions)
* [Cloud Container Registry](https://cloud.google.com/container-registry/docs/access-control)

---

## Training

It is possible to submit training jobs on GCP AI platform by making use of [Docker Containers](https://cloud.google.com/ai-platform/training/docs/using-containers).

This allows us:
* To make use of open source frameworks relatively easily.
* To scale up with [computational requirements](https://cloud.google.com/ai-platform/training/docs/machine-types).
* To take a "hands off" approach in comparison to spinning up [AI platform notebooks](https://cloud.google.com/ai-platform-notebooks).  


The [training](examples/training) folder contains a Docker file with requirements, and [a Python script](examples/training/text-classification-training.py) to train a text classification model facilitated by Flair.

#### Overview

The Docker container installs the required libraries and creates the following folder structure:
```
/root/trainer
         ├── text-classification-training.py
         ├── data
         |     ├── test.csv (optional)
         |     ├── dev.csv (optional)
         │     └── train.csv
         └─- checkpoint
               ├── final-model.pt
               ├── best-model.pt
               └── training.log
```

The [text classification script](examples/training/text-classification-training.py) also discusses the parameters more in-depth.


The gist is, that at training time, the Docker container executes the text classification script which:
* Parses the supplied arguments
* Handles ingress of test.csv / train.csv / dev.csv files from the GCS bucket to the container
* Initiates a Flair text-classification loop with the parsed arguments
* Handles egress of the trained models and logging files to the GCS bucket

#### Building the Docker image

While we can build the Docker image locally, this might take a long time given that we are reliant on hefty dependencies.


Using the cloud SDK, navigate to the folder of the Dockerfile.
With the following snippet we can submit the Docker image to be built and stored in Google Cloud Container Registry.

```shell
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE_URI=gcr.io/$PROJECT_ID/flair/text-classification:latest

# Build the container and submit to Cloud Container Registry.
gcloud builds submit --tag $IMAGE_URI
```

#### Store data in Cloud Storage
We can copy our local datasets to a Google Cloud Storage bucket by making use of the [gsutil commands](
https://cloud.google.com/storage/docs/gsutil/commands/cp) in the Cloud SDK.

For the sake of illustration we will copy .csv files to a bucket named *flair-bucket* and in the *custom-container/dataset/* nested folder.

```shell
gsutil cp *.csv gs://flair-bucket/custom-container/dataset/
```

#### Testing the Docker
Rather than submitting the job immediately to AI platform, we should first test the image in Cloud Shell. Through this way we can quickly validate and debug, and saves us valuable time rather than having to wait on AI platform spinning up machines.

By navigating to [https://console.cloud.google.com/](https://console.cloud.google.com/) and run the following command in Cloud Shell.
```shell
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE_URI=gcr.io/$PROJECT_ID/flair/text-classification:latest

# Run in Cloud Shell.
docker run $IMAGE_URI \
  --label_column_index 0 \
  --text_column_index 1   \
  --epochs 1   \
  --patience 1   \
  --gcs_data_path "custom-container/dataset/"  \
  --gcs_output_path "custom-container/output/" \
  --gcs_bucket_name "flair-bucket"   \
  --delimiter ","
```


#### Submit the training job
After we have verified that the Docker works. We can submit the container to AI platform.

For the full list of parameters, please refer to [this documentation](https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training).
```shell
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE_URI=gcr.io/$PROJECT_ID/flair/text-classification:latest
export REGION=europe-west1
export JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)

# Submit training to AI platform.
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --text_column_index 1   \
  --epochs 10   \
  --patience 3   \
  --gcs_data_path "custom-container/dataset/"  \
  --gcs_output_path "custom-container/output/" \
  --gcs_bucket_name "flair-bucket"   \
  --delimiter ","
```

---

## Deployment

To be added later.
---

### Contributions & Suggestions

[Pull requests](https://github.com/robinvanschaik/flair-on-gcp/pulls) and [issues](https://github.com/robinvanschaik/flair-on-gcp/issues) are very welcome!

I am hoping to learn more and to improve the code.


---
### Authors

* [Robin van Schaik](https://github.com/robinvanschaik)

---

### Acknowledgements

* [Flair](https://github.com/flairNLP/flair) for the text classification training framework.
* [Sentence transformers](https://github.com/UKPLab/sentence-transformers) for great sentence-level language models that can be used in Flair.
* [Huggingface](https://github.com/huggingface/transformers) for a large collection of language models that can be used in Flair.
