import re
import sys
import json
import os
from KAFY.Pipeline import TrajectoryPipeline
from .transformersPlugin.BERT.pretrainBERT import pretrain_BERT


def pretrain_model(model, data_path, config_path, output_name):
    # @YOUSSEF DO: I need to use the optut_name to save the model using this name in the pyramid
    _project_path = "/speakingTrajectories"

    pipeline = TrajectoryPipeline(
        mode="pretraining",
        use_tokenization=True,
        project_path=_project_path,
    )
    # Pretrain logic
    trajectories_list = pipeline.get_trajectories_from_csv(file_path=data_path)
    pipeline.set_trajectories(trajectories_list)
    pipeline.set_tokenization_resolution(resolution=10)
    model_path, tokenized_dataset_path = pipeline.run()
    with open(config_path, "r", encoding="utf-8") as json_file:
        model_configs = json.load(json_file)
    # @YOUSSEF DO: I need to have a set of available models to check if he chose an existing architecture
    model = model.lower()

    model_configs["checkpoint_filepath"] = model_path
    model_configs["dataset_path"] = tokenized_dataset_path
    model_configs["output_dir"] = os.path.join(
        _project_path, "temp_data_train_val_test"
    )
    print(model_configs)
    if model == "bert":
        pretrain_BERT(model_configs)

    else:
        raise ValueError("Chosen Transformer Arch. Is Not Available..")
    print(f"Pretrained model saved as {output_name}", pipeline)


def finetune_model(task, pretrained_model_path, config_path, output_name):
    # Initialize and run your fine-tuning pipeline here
    pipeline = TrajectoryPipeline(
        mode="finetuning",
        operation_type=task,
        # other relevant configurations
    )
    # Fine-tuning logic
    print(f"Fine-tuned model saved as {output_name}")


def summarize_data(data_path, model_path):
    # Initialize and run your summarization pipeline here
    pipeline = TrajectoryPipeline(
        mode="operation",
        operation_type="summarization",
        # other relevant configurations
    )
    # Summarization logic
    print("Summarization complete")


def parse_command(command=None):
    if command is None:
        command = " ".join(
            sys.argv[1:]
        )  # Join command-line arguments into a single string

    # The rest of your parsing logic remains the same

    pretrain_match = re.match(
        r"PRETRAIN\s+(\w+)\s+FROM\s+(\S+)\s+USING\s+(\S+)\s+AS\s+(\S+)",
        command,
        re.IGNORECASE,
    )
    finetune_match = re.match(
        r"FINETUNE\s+(\w+)\s+FOR\s+(\w+)\s+USING\s+(\S+)\s+WITH\s+(\S+)\s+AS\s+(\S+)",
        command,
        re.IGNORECASE,
    )
    summarize_match = re.match(
        r"SUMMARIZE\s+FROM\s+(\S+)\s+USING\s+(\S+)", command, re.IGNORECASE
    )

    if pretrain_match:
        model, data, config, output_name = pretrain_match.groups()
        pretrain_model(model, data, config, output_name)
    elif finetune_match:
        model, task, pretrained_model, config, output_name = finetune_match.groups()
        finetune_model(task, pretrained_model, config, output_name)
    elif summarize_match:
        data, model = summarize_match.groups()
        summarize_data(data, model)
    else:
        print("Command not recognized.")
