from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from torch.optim.adam import Adam
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings
from flair.embeddings import WordEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import DocumentRNNEmbeddings
from trainer_utils import create_weight_dict


def initialize_training(text_column_index,
                        label_column_index,
                        delimiter=';',
                        model_type=None,
                        model=None,
                        max_epochs=10,
                        patience=3,
                        use_amp=0,
                        calc_class_weights=0):
    """
    Create a text classification model using FLAIR, SentenceTransformers and
    Huggingface Transformers.
    Params:
    data_folder_path: Folder path with each file titled appropriately i.e.
                      train.csv test.csv dev.csv.
                      Will create a 80/10/10 split if only train is supplied.
    output_folder_path: Folder path for storing the best model & checkpoints.
    text_column_index: In which index (starting from 0) the input column is located.
    label_column_index: In which index (starting from 0) the label column is located.
    delimiter: type of delimiter used in the .csv file.
    model_type: SentenceTransformerDocumentEmbeddings or TransformerDocumentEmbeddings
    model: Which model to use.
    max_epochs: Number of epochs to train the model for.
    patience: Number of epochs without improvement before adjusting learning rate.
    use_amp: Whether to enable automatic mixed precisions (AMP).
    calc_class_weights: Whether to create a dictionary with class weights to deal
                        with imbalanced datasets.
    Output:
        best-model.pt
        final-model.pt
        training.log
    """

    # 1. Column format indicating which columns hold the text and label(s)
    column_name_map = {text_column_index: "text",
                       label_column_index: "label_topic"}

    # 2. Load corpus containing training, test and dev data.
    corpus: Corpus = CSVClassificationCorpus("/root/text-classification/data/",
                                             column_name_map,
                                             skip_header=True,
                                             delimiter=delimiter)

    # Print statistics about the corpus.
    training_data_statistics = corpus.obtain_statistics()
    print(training_data_statistics)

    # 3A. Create a label dictionary.
    label_dict = corpus.make_label_dictionary()

    # 3B. Calculate class weights.
    if bool(calc_class_weights):
        weight_dict = create_weight_dict(delimiter=delimiter,
                                         label_index=label_column_index)
    else:
        weight_dict = None

    # 4. Initialize the sentence_transformers model.
    if model_type == "SentenceTransformerDocumentEmbeddings":
        document_embeddings = SentenceTransformerDocumentEmbeddings(model)
    elif model_type == "TransformerDocumentEmbeddings":
        document_embeddings = TransformerDocumentEmbeddings(
            model, fine_tune=True)
    elif model_type == "WordEmbeddings":
        word_embeddings = [WordEmbeddings(model)]
        document_embeddings = DocumentRNNEmbeddings(
            word_embeddings, hidden_size=256)
    elif model_type == "StackedEmbeddings":
        document_embeddings = DocumentRNNEmbeddings([
            WordEmbeddings('glove'),
            FlairEmbeddings(model + '-backward'),
            FlairEmbeddings(model + '-forward')
        ])
    else:
        raise Exception(
            "Pick SentenceTransformerDocumentEmbeddings, StackedEmbeddings, WordEmbeddings or TransformerDocumentEmbeddings.")

    # 5. create the text classifier
    classifier = TextClassifier(document_embeddings,
                                label_dictionary=label_dict,
                                loss_weights=weight_dict)

    # 6. initialize the text classifier trainer with Adam optimizer
    trainer = ModelTrainer(classifier,
                           corpus,
                           optimizer=Adam,
                           use_tensorboard=False)

    # 7. start the training
    trainer.train("/root/text-classification/checkpoint/",
                  learning_rate=3e-5,
                  max_epochs=max_epochs,
                  patience=patience,
                  use_amp=bool(use_amp),
                  checkpoint=True,
                  mini_batch_size=16,
                  mini_batch_chunk_size=4)
