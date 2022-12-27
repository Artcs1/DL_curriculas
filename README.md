# Improving Embeddings Representations for Comparing Higher Education Curricula: A Use Case in Computing

This is the official repository of the EMNLP2022 main conference named "Improving Embeddings Representations for Comparing Higher Education Curricula: A Use Case in Computing".

Visit our paper:

## Installation

1. Create a conda enviorenment

```
conda create --name py37-curriculas python=3.7
conda activate py37-curriculas
```

2. Install the requirements

```
pip install -r requirements.txt
```


## Usage

1. Activate the conda enviorenment

```
conda activate py37-curriculas
```

2. Generate the embeddings representation

```
python3 generate_representations.py --model <model_name> --mode [curricula|curso] DATA_TG100/
```

3. Plot representations

```
python3 plot_representations.py <model_name>
```

4. Metric Evaluation

```
python3 metrics_valid.py --nargs <model_name>
```

## Benchmark



## Cite our work

```
@InProceedings{Murrugarra-Llerena_2022_EMNLP,
    author    = {Murrugarra-Llerena, Jeffri and Alva-Manchego, Fernando and Murrugarra-Llerena, Nils},
    title     = {Improving Embeddings Representations for Comparing Higher Education Curricula: A Use Case in Computing},
    booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
    month     = {December},
    year      = {2022},
    pages     = {}
}
```
