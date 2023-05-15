# Author: "Mars Gokturk Buchholz"
# Version: "CS224u, Stanford, Winter 2023"

import subprocess
import pandas as pd
import wandb

wandb.login()
import time


class ModelRun:
    """ Client to make requests to the OpenAI given hyper-parameters"""

    def __init__(self, base_model: str, n_epochs: int, batch_size: int, learning_rate_multiplier: float,
                 openai_apikey: str, train_file, valid_file, test_file, compute_classification_metrics=True):
        self.base_model = base_model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate_multiplier = learning_rate_multiplier
        self.compute_classification_metrics = compute_classification_metrics
        self.api_key = openai_apikey
        self.n_classes = 6
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file

        # preprocess files for fine-tuning consumption
        train_ds = LiarDataset(file_path=self.train_file, name="liar_train")
        train_ds.process()
        train_ds.write()

        valid_ds = LiarDataset(file_path=self.valid_file, name="liar_valid")
        valid_ds.process()
        valid_ds.write()

        test_ds = LiarDataset(file_path=self.test_file, name="liar_test")
        test_ds.process()
        test_ds.write()

        train_df = train_ds.get()
        valid_df = valid_ds.get()
        dev_df = pd.concat([train_df, valid_df])
        dev_df.to_json(f"dev.jsonl", orient='records', lines=True)

    def request_finetune(self):
        cmd = f"""/usr/local/Caskroom/miniconda/base/envs/nlp-deep/bin/openai api fine_tunes.create -t liar_train.jsonl -v liar_valid.jsonl -m {self.base_model} --compute_classification_metrics --classification_n_classes 6 --suffix liar --batch_size {self.batch_size} --n_epochs {self.n_epochs} --learning_rate_multiplier {self.learning_rate_multiplier}
        """
        print(f"Executing cmd = {cmd}")
        return self.subprocess_run(cmd)

    def request_finetune_dev(self):
        cmd = f"""/usr/local/Caskroom/miniconda/base/envs/nlp-deep/bin/openai api fine_tunes.create -t dev.jsonl -v liar_test.jsonl -m {self.base_model} --compute_classification_metrics --classification_n_classes 6 --suffix liar --batch_size {self.batch_size} --n_epochs {self.n_epochs} --learning_rate_multiplier {self.learning_rate_multiplier}"""
        print(f"Executing cmd = {cmd}")
        return self.subprocess_run(cmd)

    def subprocess_run(self, cmd: bytes) -> str:
        try:
            return subprocess.run(cmd, input=" \n \n".encode(), shell=True, env={"OPENAI_API_KEY": self.api_key})
        except subprocess.CalledProcessError:
            raise ValueError(f"Failed to execute the command: {cmd}")


class LiarDataset:
    prompt_end = "\n\n###\n\n"
    completion_start = " "
    completion_end = "###"

    def __init__(self, file_path, name):
        self.data: pd.DataFrame = pd.read_csv(file_path,
                                              index_col=False,
                                              delimiter="\t",
                                              header=None,
                                              names=["id", "label", "statement", "subject",
                                                     "speaker", "speaker_job_title", "state_info",
                                                     "party_affiliation",
                                                     "barely_true_c", "half_true_c", "false_c", "mostly_true_c",
                                                     "pantsonfire_c", "context"])
        self.name = name

    def process(self, include_meta=False):

        def process_label(x):
            return f"{LiarDataset.completion_start}{x}{LiarDataset.completion_end}"

        def process_prompt(x):
            return f"{x}{LiarDataset.prompt_end}"

        def process_prompt_with_meta(x):
            """Use space to concat statement subject context fields.
            """
            result = f"{x['statement']} {x['subject']} {x['context']}{LiarDataset.prompt_end}"
            return result

        self.data["completion"] = self.data["label"].map(lambda x: process_label(x))

        if include_meta:
            self.data["prompt"] = self.data.apply(lambda x: process_prompt_with_meta(x), axis=1)
        else:
            self.data["prompt"] = self.data["statement"].map(lambda x: process_prompt(x))

        self.data = self.data.drop_duplicates()
        self.data = self.data[["prompt", "completion"]]

    def get(self):
        return self.data

    def write(self):
        self.data.to_json(f"{self.name}.jsonl", orient='records', lines=True)


if __name__ == "__main__":
    # 1. hyper-parameter tune using train and valid datasets    #
    learning_rate_multipliers = [0.02, 0.04, 0.08, 0.16, 0.2, 0.22]
    n_epochs = [2, 3, 4, 5]
    trials = 5

    for lrm in learning_rate_multipliers:
        for ne in n_epochs:
            for i in range(trials):
                mr = ModelRun(base_model="ada",
                              n_epochs=ne,
                              batch_size=256,
                              learning_rate_multiplier=lrm,
                              openai_apikey="####",
                              train_file="./liar_dataset/train.tsv",
                              valid_file="./liar_dataset/valid.tsv",
                              test_file="./liar_dataset/test.tsv")
                result = mr.request_finetune()
                time.sleep(5)

                print(result)
                print("---------------------------------------------------------------------------")

    # get all runs from the CMD and publish to W&B
    # openai api fine_tunes.list
    # openai wandb sync

    # Train it using train + valid dataset and compute metrics on the test set (5) times
    # ada and curie method.
    models = ["ada", "curie"]
    for model in models:
        for i in range(trials):
            # use best hyper-parameters to get test accuracy
            mr = ModelRun(base_model=model,
                          n_epochs=5,
                          batch_size=256,
                          learning_rate_multiplier=0.2,
                          openai_apikey="####",
                          train_file="../liar_dataset/train.tsv",
                          valid_file="../liar_dataset/valid.tsv",
                          test_file="../liar_dataset/test.tsv")
            result = mr.request_finetune_dev()
            time.sleep(5)

            print(result)
            print("---------------------------------------------------------------------------")

    # get all runs from the CMD and publish to W&B
    # openai api fine_tunes.list
    # openai wandb sync
