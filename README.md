# Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks
TO BE COMPLETED...(by the end of July 3, 2026)

This repository is the official implementation of AdSent [Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks](https://doi.org/10.1145/3774904.3792606) published in WWW 2026.

## Requirements

To install requirements:
```
conda create -n Adsent python=3.8.20
conda activate AdSent
conda install pytorch==2.4.1 cudatoolkit=11.3.1 -c pytorch
pip install -r requirements.txt
```

## Dataset

Additional details for Politifact and Gossipcop dataset can be found on their [official website](https://github.com/kaidmml/fakenewsnet).

## Hugging Face Access

The repository uses meta-llama/Llama-3.1-8B-Instruct model both for training AdSent and for sentiment reformation. This is a gated Hugging Face model. Before running the code, make sure that:

1- you have access to the model on Hugging Face

2- you are logged in locally:
```
huggingface-cli login
```
## Running the code
### 1. Sentiment manipulation
To generate sentiment-altered versions of news articles while preserving their factual content, use the `reformation.py` script. An example command is shown below:

```bash
python reformation.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset politifact \
    --sentiment neutral
```
Main arguments:

| Argument     |                         Description                         | 
|--------------|:-----------------------------------------------------------:| 
| --model_name |         Hugging Face model used for text generation         | 
| --dataset      |     Dataset name (`politifact`, `gossipcop`, or `lun`)      |
| --sentiment |   Target sentiment (`positive`, `negative`, or `neutral`)   | 
| --base_dir      | Directory containing the dataset files (default: `./data`)  |

The generated dataset is automatically saved under the data/ directory. The sentiment-manipulated datasets used in the paper are also provided with this repository.

### 2. Fake News Detection

For fake news detection, use the `main.py` script. We provide an example bash script, `eval.sh`, which includes three parts:

- inference with the zero shot LLMs,
- training AdSent,
- inference with the trained AdSent checkpoint.

Modify the desired section in `eval.sh` and run:

```bash
bash eval.sh
```
## Citation
If you use this code, please cite:
```
@inproceedings{tahmasebi2026adsent,
  title={Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks},
  author={Tahmasebi, Sahar and M{\"u}ller-Budack, Eric and Ewerth, Ralph},
  booktitle={Proceedings of the ACM Web Conference 2026},
  year={2026}
}
```
## Credit
This repository is built by [Sahar Tahmasebi](https://github.com/sahartahmasebi). 

## License

Our work is licenced under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
