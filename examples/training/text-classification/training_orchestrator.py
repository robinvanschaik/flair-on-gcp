import argparse
import logging
from trainer.text_classification_trainer import initialize_training
from training_io.file_utils import gcs_data_to_docker, training_output_to_gcs

# Only log essentials to AI platform.
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
        --model:              Specifies which language model embeddings to use.
                              Defaults to distiluse-base-multilingual-cased.
        --epochs:             Indicates the number of epochs to train for.
                              Defaults to 10.
        --patience:           Indicates the number of epochs without improvement
                              before adjusting learning rate.
                              Defaults to 3.
        --use_amp:            Indicates which whether Automatic Mixed Precision
                              should be used.
        --calc_class_weights: Indicates whether class weights should be calculated
                              in order to deal with imbalanced datasets.
        --gcs_data_path:      The Google Cloud Storage (gcs) folder path containing the data.
        --gcs_output_path:    The Google Cloud Storage (gcs) folder path for storing the outputs.
      Output:
        Dictionary of arguments.
    """

    parser = argparse.ArgumentParser(
        description='Text Classification with Flair On GCP via Docker Container.')

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
        default=None,
        help='Whether to use SentenceTransformerDocumentEmbeddings, SentenceTransformerDocumentEmbeddings, WordEmbeddings or TransformerDocumentEmbeddings.')

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Specifies which language model embeddings to use from SentenceTransformerDocumentEmbeddings, WordEmbeddings, an ensemble or '
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
        '--use_amp',
        type=int,
        default=0,
        metavar='N',
        help='Indicates which whether Automatic Mixed Precision should be used.'
             '1 for True, 0 for False')

    parser.add_argument(
        '--calc_class_weights',
        type=int,
        default=0,
        metavar='N',
        help='Indicates whether class weights should be calculated in order to deal with imbalanced datasets.'
             '1 for True, 0 for False')

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


def main():
    # Retrieve the supplied training settings
    args = get_args()

    # Copy data from GCS to the Docker.
    gcs_data_to_docker(gcs_data_path=args.gcs_data_path,
                       gcs_bucket_name=args.gcs_bucket_name)

    # Once the data is copied to the Docker, initialize training.
    initialize_training(text_column_index=args.text_column_index,
                        label_column_index=args.label_column_index,
                        delimiter=args.delimiter,
                        model_type=args.model_type,
                        model=args.model,
                        max_epochs=args.epochs,
                        patience=args.patience,
                        use_amp=args.use_amp,
                        calc_class_weights=args.calc_class_weights)

    # Copy the training output from the Docker to GCS.
    training_output_to_gcs(gcs_output_path=args.gcs_output_path,
                           gcs_bucket_name=args.gcs_bucket_name)


if __name__ == '__main__':
    main()
