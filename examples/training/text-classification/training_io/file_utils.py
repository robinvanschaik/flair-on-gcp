from os import listdir
from os.path import isfile, join
from google.cloud import storage


def gcs_data_to_docker(gcs_bucket_name, gcs_data_path):
    """Copies the data from Google Cloud Storage in the project to the Docker executing the job.
        Params:
            gcs_bucket_name: a str containing the name of the gcs bucket within the same project.
            gcs_data_path: path to the folder containing the test.csv, train.csv, dev files.
    """
    # Initialize storage client.
    storage_client = storage.Client()

    print(f'Retrieving data from {gcs_bucket_name}')

    bucket = storage_client.get_bucket(gcs_bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_data_path)

    for blob in blobs:
        # GCS will list the folder as well, thus check whether it is a .csv file.
        if blob.name.endswith('.csv'):
            print(f'Downloading {blob.name} to the /data/ folder')
            # GCS does not use folders in the background, it contains the full name as the blob path.
            # Split by slashes to get the filename in the folder.
            filename = blob.name.split("/")[-1]
            blob.download_to_filename('/root/text-classification/data/' + filename)
            print(f'Copied {blob.name} to the /data/ folder')


def training_output_to_gcs(gcs_bucket_name, gcs_output_path):
    # Thanks https://hackersandslackers.com/manage-files-in-google-cloud-storage-with-python/
    """Copies the data from the Docker executing the job to the Google Cloud Storage in the project.
        Params:
            gcs_bucket_name: a str containing the name of the gcs bucket within the same project.
            gcs_data_path: path to the folder to store the outputs.
    """

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs_bucket_name)
    files = [f for f in listdir("/root/text-classification/checkpoint/")
             if isfile(join("/root/text-classification/checkpoint/", f))]
    for file in files:
        localFile = "/root/text-classification/checkpoint/" + file
        blob = bucket.blob(gcs_output_path + file)
        blob.upload_from_filename(localFile)
        print(f'Uploaded {localFile} to "{gcs_output_path}" bucket.')
