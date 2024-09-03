# KAFY Library

The KAFY library provides an extensible system for various trajectory operations and includes a versatile command-line interface (CLI) for managing these operations using SQL-like commands. It functions as a toolkit for researchers, facilitating the construction, management, and execution of diverse trajectory operations.

## Features

- **Extensible System**: Designed to accommodate various trajectory operations with flexibility and ease.
- **User-Friendly**: Simplifies the construction and management of trajectory operations, making it accessible to researchers.
- **SQL-like Command Interface**: Allows users to execute trajectory operations through intuitive, SQL-like commands.

## Installation

You can install the KAFY library using pip:

```bash
pip install KAFYTraj==0.1.13
```

## Usage

To use the library, import the `TrajectoryPipeline` class from `KAFY`:

```python
from KAFY import TrajectoryPipeline

# Initialize the pipeline
my_pipeline = TrajectoryPipeline(
    mode="pretraining",
    operation_type="generation",
    use_tokenization=True,
    use_detokenization=True,
    use_spatial_constraints=True,
    modify_spatial_constraints=True,
    use_predefined_spatial_constraints=True,
    project_path="/content/"
)
```
# SQL-like Command Interface
To use the SQL-like command interface, you can execute commands directly from the terminal. The CLI allows you to perform operations such as pretraining, fine-tuning, and summarizing data.

## Pretraining a Model:
```python
kafy_parser "PRETRAIN bert \
FROM mydata.csv \
USING bert_config.json \
AS my_pretrained_model.pkl"
```

## Fine-Tune a Model:

```python
kafy_parser "FINETUNE bert \
FOR summarization \
USING my_pretrained_model.pkl \
WITH finetune_config.json \
AS my_finetuned_model.pkl"
```
## Summarize Data:

```python
kafy_parser "SUMMARIZE \
FROM requested_data_to_summarize.csv \
USING my_finetuned_model.pkl"
```


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or support, please contact husse408@umn.edu.