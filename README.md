# Assessing the Effectiveness of GPT-3 in Detecting False Political Statements

This repository contains the code and data used in our academic paper: [Assessing the Effectiveness of GPT-3 in Detecting False Political Statements: A Case Study on the LIAR Dataset](https://arxiv.org/abs/2306.08190).

## Project Description

In this project, we probe the capabilities of GPT-3, an advanced language model by OpenAI, in identifying false political statements. Our study is anchored on two primary experiments: the fine-tuning of GPT-3 and its application in a zero-shot learning context. Both experiments leverage the LIAR dataset, an established benchmark resource curated by William Yang Wang from the Politifact website specifically for fake news detection tasks.

## Repository Contents

1. `api_requests/`: This directory contains code for making requests to the OpenAI API.
2. `input_data/`: This directory contains the input datasets used for training and testing.
3. `output_data/`: This directory contains the output data generated by our experiments.
4. `results/`: This directory contains the experimental results, including model performance metrics.

## Setup & Usage

To reproduce our work, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Install necessary dependencies listed in `requirements.txt`.
3. **Run the Code**: Navigate to the `api_requests/` directory and run the scripts to interact with the OpenAI API.
4. **Analyze the Results**: Navigate to the `results/` directory to analyze the experimental results.

## Experiment Tracking

We used [Weights & Biases](https://wandb.ai/site) for experiment tracking. OpenAI's integration with Weights & Biases allows for easy synchronization of experiment results. Our experiment results can be accessed in our [Weights & Biases project page](https://api.wandb.ai/links/mars-works/b6w8u2mi).

## Acknowledgments

We would like to express our sincere gratitude to Christopher Potts, the course professor of the Stanford Artificial Intelligence professional program: Natural Language Understanding XCS224U class. 
His exceptional teaching and guidance were instrumental in the successful completion of this project. 
We are also grateful to the course facilitators for their valuable support throughout the course.

## Citation
If you use our code in your research, please cite our work:
```bibtex
@misc{buchholz2023assessing,
      title={Assessing the Effectiveness of GPT-3 in Detecting False Political Statements: A Case Study on the LIAR Dataset}, 
      author={Mars Gokturk Buchholz},
      year={2023},
      eprint={2306.08190},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

