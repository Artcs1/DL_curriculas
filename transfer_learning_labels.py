import pandas as pd
from os import scandir, getcwd
from os.path import abspath,isfile
import numpy as np

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.language_modeling import LanguageModelingModel
import logging
import argparse
import torch

from Loader.data_loader import Curriculas


def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("path")
    args = parser.parse_args()



    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Preparing train data
    traind = Curriculas(path=args.path, data = 'train')
    train_data = [list(x) for x in zip(traind.x, traind.y)]

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text", "labels"]

    # Preparing eval data
    evald  = Curriculas(path=args.path, data = 'valid')
    eval_data =  [list(x) for x in zip(evald.x, evald.y)]
    eval_df = pd.DataFrame(eval_data)
    eval_df.columns = ["text", "labels"]

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=1)
    # Create a ClassificationModel
    model = ClassificationModel(
        'bert',
        'bert-base-uncased',
        num_labels=5,
        args=model_args,
        use_cuda=torch.cuda.is_available())

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    # Make predictions with the model

    testd = Curriculas(path=args.path, data = 'test')
    test_data = [list(x) for x in zip(testd.x, testd.y)]
    test_df = pd.DataFrame(test_data)
    test_df.columns = ["text", "labels"]
    predictions, raw_outputs = model.predict(["Sam was a Wizard"])

if __name__ == '__main__':
    main()
