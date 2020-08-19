# Load the libraries.
from fastapi import FastAPI
from pydantic import BaseModel

# Load the required modules from Flair.
from flair.models import TextClassifier
from flair.data import Sentence

# Set up the schema & input validation for the inputs.
class Case(BaseModel):
    text: str

# Load the model in as a global variable.
classifier = TextClassifier.load('./model/best-model.pt')

# Define the prediction function.
def classify_text(classifier, sentence):

    '''
    A small function to classify the incoming string.
    ------------------------
    Params:
    classifier: The loaded model object.
    sentence: A string to classify.
    ------------------------
    Output:
    A list of tuples containing labels & probabilities.
    '''

    sentence = Sentence(sentence)
    classifier.predict(sentence, multi_class_prob=True)
    return sentence.labels

# Initialize the FastAPI endpoint.
app = FastAPI()

# Set the address and await calls.
@app.post('/classify-text')
async def classify_text_endpoint(case : Case):
    """Takes the text request and returns a record with the labels & associated probabilities."""

    # Use the pretrained model to classify the incoming text in the request.
    classified_text = classify_text(classifier, case.text)

    return classified_text
