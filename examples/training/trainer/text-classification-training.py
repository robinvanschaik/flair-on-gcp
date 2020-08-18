# For parsing arguments from the CLI to AI platform.
import argparse

# Import modules from flair.
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from torch.optim.adam import Adam

# For interfacing with google cloud storage.
from google.cloud import storage

# Only log essentials.
logging.basicConfig(level=logging.ERROR)


def get_args():
    """Parses the arguments as input for the training function.
      Params:
        --label_column_index: Indicates the column-index of the label in the train.csv.
                              Defaults to index 0.
        --text_column_index:  Indicates the column-index of the feature in the train.csv.
                              Defaults to index 1.
        --delimiter:          The delimiter used in the train.csv in order to parse columns.
                              Defaults to ;.
        --model_type:         Whether to use SentenceTransformerDocumentEmbeddings or TransformerDocumentEmbeddings.
                              Defaults to SentenceTransformerDocumentEmbeddings.
        --model:              Specifies which language model embeddings to use.
                              Defaults to distiluse-base-multilingual-cased.
        --epochs:             Indicates the number of epochs to train for.
                              Defaults to 10.
        --patience:           Indicates the number of epochs without improvement before aborting.
                              Defaults to 3.
        --gcs_data_path:      The Google Cloud Storage (gcs) folder path containing the data.
        --gcs_output_path:    The Google Cloud Storage (gcs) folder path for storing the outputs.
      Output:
        Dictionary of arguments.
    """

    parser = argparse.ArgumentParser(description='Text Classification with Flair On GCP via Docker Container')

    parser.add_argument(
        '--label_column_index',
        type=int,
        default=0,
        metavar='N',
        help='Indicates the column-index of the label in the train.csv.')

    parser.add_argument(
        '--text_column_index',
        type=int,
        default=1,
        metavar='N',
        help='Indicates the column-index of the feature in the train.csv.')

    parser.add_argument(
        '--delimiter',
        type=str,
        default=';',
        help='The delimiter used in the train.csv in order to parse columns.')

    parser.add_argument(
        '--model_type',
        type=str,
        default='SentenceTransformerDocumentEmbeddings',
        help='Whether to use SentenceTransformerDocumentEmbeddings or TransformerDocumentEmbeddings.')

    parser.add_argument(
        '--model',
        type=str,
        default='distiluse-base-multilingual-cased',
        help='Specifies which language model embeddings to use from SentenceTransformerDocumentEmbeddings or '
             'TransformerDocumentEmbeddings')

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='Indicates the number of epochs to train for.')

    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        metavar='N',
        help='Indicates the number of epochs without improvement before aborting.')

    parser.add_argument(
        '--gcs_bucket_name',
        type=str,
        default=None,
        help="The Google Cloud Storage bucket name."
    )

    parser.add_argument(
        '--gcs_data_path',
        type=str,
        default=None,
        help="The Google Cloud Storage bucket folder path containing the input data."
    )
    parser.add_argument(
        '--gcs_output_path',
        type=str,
        default=None,
        help="The Google Cloud Storage bucket folder path for storing the outputs."
    )

    args = parser.parse_args()

    return args


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
            blob.download_to_filename('./trainer/data/' + filename)
            print(f'Copied {blob.name} to the /data/ folder')


def initialize_training(text_column_index, label_column_index, delimiter=';',
                        model_type="SentenceTransformerDocumentEmbeddings", model='distiluse-base-multilingual-cased',
                        max_epochs=10, patience=3):
    """
    Create a text classification model using FLAIR and SentenceTransformers/Huggingface Transformers.
    ------------------------
    Params:
    data_folder_path: Folder path with each file titled appropriately i.e. train.csv test.csv dev.csv.
                      Will create a 80/10/10 split if only train is supplied.
    output_folder_path: Folder path for storing the best model & checkpoints.
    text_column_index: In which index (starting from 0) the input column is located.
    label_column_index: In which index (starting from 0) the label column is located.
    delimiter: type of delimiter used in the .csv file.
    model_type: SentenceTransformerDocumentEmbeddings or TransformerDocumentEmbeddings
    model: Which model to use. Defaults to a multilingual model.
    max_epochs: Number of epochs to train the model for.
    ------------------------
    Output:
    best-model.pt
    final-model.pt
    training.log
    """

    # 1. Column format indicating which columns hold the text and label(s)
    column_name_map = {text_column_index: "text", label_column_index: "label_topic"}

    # 2. Load corpus containing training, test and dev data and if CSV has a header, you can skip it
    corpus: Corpus = CSVClassificationCorpus("./trainer/data/",
                                             column_name_map,
                                             skip_header=True,
                                             delimiter=delimiter)

    # Print statistics about the corpus.
    training_data_statistics = corpus.obtain_statistics()
    print(training_data_statistics)

    # 3. Create a label dictionary.
    label_dict = corpus.make_label_dictionary()

    # 4. Initialize the sentence_transformers model.
    if model_type == "SentenceTransformerDocumentEmbeddings":
        document_embeddings = SentenceTransformerDocumentEmbeddings(model)
    elif model_type == "TransformerDocumentEmbeddings":
        document_embeddings = TransformerDocumentEmbeddings(model, fine_tune=True)
    else:
        raise Exception("Pick SentenceTransformerDocumentEmbeddings or TransformerDocumentEmbeddings.")

    # 5. create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

    # 6. initialize the text classifier trainer with Adam optimizer
    trainer = ModelTrainer(classifier,
                           corpus,
                           optimizer=Adam,
                           use_tensorboard=False)

    # 7. start the training
    trainer.train("./trainer/checkpoint/",
                  learning_rate=3e-5,  # use very small learning rate
                  mini_batch_size=16,
                  mini_batch_chunk_size=4,
                  max_epochs=max_epochs,
                  patience=patience,
                  checkpoint=True)


def training_output_to_gcs(gcs_bucket_name, gcs_output_path):
    # Thanks https://hackersandslackers.com/manage-files-in-google-cloud-storage-with-python/
    """Copies the data from the Docker executing the job to the Google Cloud Storage in the project.
        Params:
            gcs_bucket_name: a str containing the name of the gcs bucket within the same project.
            gcs_data_path: path to the folder to store the outputs.
    """

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs_bucket_name)
    files = [f for f in listdir("./trainer/checkpoint/") if isfile(join("./trainer/checkpoint/", f))]
    for file in files:
        localFile = "./trainer/checkpoint/" + file
        blob = bucket.blob(gcs_output_path + file)
        blob.upload_from_filename(localFile)
        print(f'Uploaded {localFile} to "{gcs_output_path}" bucket.')


def main():
    # Retrieve the supplied training settings
    args = get_args()

    # Copy data from GCS to the Docker.
    gcs_data_to_docker(gcs_data_path=args.gcs_data_path, gcs_bucket_name=args.gcs_bucket_name)

    # Once the data is copied to the Docker, initialize training.
    initialize_training(text_column_index=args.text_column_index,
                        label_column_index=args.label_column_index,
                        delimiter=args.delimiter,
                        model_type=args.model_type,
                        model=args.model,
                        max_epochs=args.epochs,
                        patience=args.patience)

    # Copy the training output from the Docker to GCS.
    training_output_to_gcs(gcs_output_path=args.gcs_output_path, gcs_bucket_name=args.gcs_bucket_name)


if __name__ == '__main__':
    main()
