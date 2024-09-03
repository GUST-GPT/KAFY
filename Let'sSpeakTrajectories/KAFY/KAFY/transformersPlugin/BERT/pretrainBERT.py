import os
import time
from KAFY.installPackages import install_package

install_package("tensorflow")
install_package("tensorflow_text")
install_package("keras_nlp")
install_package("datasets")

import tensorflow as tf
import keras_nlp
from datasets import load_dataset
from tensorflow import keras
import pickle


def split_dataset(dataset_path, split_ratio):
    """
    Splits a dataset stored in a .pkl file into training, validation, and test sets.

    Args:
        dataset_path (str): Path to the .pkl file containing the dataset.
        split_ratio (tuple): Ratios for splitting the dataset into training, validation, and test sets.

    Returns:
        tuple: Three lists containing the training, validation, and test datasets.
    """
    # Load the dataset from the .pkl file
    with open(dataset_path, "rb") as file:
        data = pickle.load(file)

    # Ensure data is in the expected format (list of strings or lines)
    if not isinstance(data, list):
        raise ValueError(
            "Loaded data is not in the expected format. It should be a list."
        )

    total_items = len(data)
    train_split = int(split_ratio[0] * total_items)
    val_split = int(split_ratio[1] * total_items)

    # Split the dataset
    train_data = data[:train_split]
    val_data = data[train_split : train_split + val_split]
    test_data = data[train_split + val_split :]

    return train_data, val_data, test_data


def save_split_datasets(train_lines, val_lines, test_lines, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Ensure each line is a string
    if not all(isinstance(line, str) for line in train_lines):
        train_lines = [str(line) for line in train_lines]
    if not all(isinstance(line, str) for line in val_lines):
        val_lines = [str(line) for line in val_lines]
    if not all(isinstance(line, str) for line in test_lines):
        test_lines = [str(line) for line in test_lines]

    with open(os.path.join(output_dir, "train.txt"), "w") as train_file:
        train_file.writelines(train_lines)
    with open(os.path.join(output_dir, "val.txt"), "w") as val_file:
        val_file.writelines(val_lines)
    with open(os.path.join(output_dir, "test.txt"), "w") as test_file:
        test_file.writelines(test_lines)

    return {
        "train": os.path.join(output_dir, "train.txt"),
        "validation": os.path.join(output_dir, "val.txt"),
        "test": os.path.join(output_dir, "test.txt"),
    }


# Function to generate vocabulary from the dataset
def generate_vocabulary(dataset_path, output_dir):
    # Load the dataset and create a vocabulary set
    with open(dataset_path, "r") as file:
        words = set(file.read().split())

    # Add [UNK] token to the vocabulary if it's not already present
    words.add("[UNK]")  # Ensure [UNK] is in the vocabulary
    words.add("[MASK]")  # Ensure [UNK] is in the vocabulary

    vocab_file_path = os.path.join(output_dir, "vocab.txt")

    # Write the vocabulary to a file
    with open(vocab_file_path, "w") as vocab_file:
        vocab_file.write("\n".join(sorted(words)))

    return vocab_file_path


# Load and preprocess data
def load_and_preprocess_data(data_files, batch_size):
    ds = load_dataset("text", data_files=data_files)
    return {split: convert_to_tf_dataset(ds[split], batch_size) for split in ds}


def convert_to_tf_dataset(dataset, batch_size):
    def generator():
        for example in dataset:
            yield example["text"]

    return (
        tf.data.Dataset.from_generator(
            generator, output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
        )
        .filter(lambda x: tf.strings.length(x) > 100)
        .batch(batch_size)
    )


# Tokenizer and masker functions
def get_tokenizer(vocab_file_path, seq_length):
    return keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab_file_path,
        sequence_length=seq_length,
        lowercase=True,
        strip_accents=True,
        oov_token="[UNK]",  # Specify the OOV token
    )


def get_masker(tokenizer, mask_rate, predictions_per_seq):
    return keras_nlp.layers.MaskedLMMaskGenerator(
        vocabulary_size=tokenizer.vocabulary_size(),
        mask_selection_rate=mask_rate,
        mask_token_id=tokenizer.token_to_id("[MASK]"),
        mask_selection_length=predictions_per_seq,
    )


def preprocess(tokenizer, masker, inputs):
    inputs = tokenizer(inputs)
    outputs = masker(inputs)
    features = {
        "token_ids": outputs["token_ids"],
        "mask_positions": outputs["mask_positions"],
    }
    labels = outputs["mask_ids"]
    weights = outputs["mask_weights"]
    return (features, labels, weights)


# Build BERT model
def build_bert_model(config, tokenizer):
    inputs = keras.Input(shape=(config["seq_length"],), dtype=tf.int32)

    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=tokenizer.vocabulary_size(),
        sequence_length=config["seq_length"],
        embedding_dim=config["model_dim"],
    )
    outputs = embedding_layer(inputs)
    outputs = keras.layers.LayerNormalization(epsilon=config["norm_epsilon"])(outputs)
    outputs = keras.layers.Dropout(rate=config["dropout"])(outputs)

    for _ in range(config["num_layers"]):
        outputs = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=config["intermediate_dim"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            layer_norm_epsilon=config["norm_epsilon"],
        )(outputs)

    encoder_model = keras.Model(inputs, outputs)
    return encoder_model


# Custom time-based checkpoint callback
class TimeBasedCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_freq_minutes=10):
        super(TimeBasedCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq_minutes = save_freq_minutes
        self.last_save_time = time.time()

    def on_batch_end(self, batch, logs=None):
        current_time = time.time()
        if (current_time - self.last_save_time) >= self.save_freq_minutes * 60:
            self.model.save(self.filepath)
            print(f"\nCheckpoint saved at {self.filepath}")
            self.last_save_time = current_time


# Train BERT model
def train_bert_model(
    pretrain_ds,
    pretrain_val_ds,
    encoder_model,
    config,
    checkpoint_filepath,
):
    inputs = {
        "token_ids": keras.Input(shape=(config["seq_length"],), dtype=tf.int32),
        "mask_positions": keras.Input(
            shape=(config["predictions_per_seq"],), dtype=tf.int32
        ),
    }

    encoded_tokens = encoder_model(inputs["token_ids"])
    outputs = keras_nlp.layers.MaskedLMHead(
        token_embedding=encoder_model.get_layer(
            "token_and_position_embedding"
        ).token_embedding,
        activation="softmax",
    )(encoded_tokens, mask_positions=inputs["mask_positions"])

    pretraining_model = keras.Model(inputs, outputs)

    pretraining_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.AdamW(
            learning_rate=config["pretraining_learning_rate"]
        ),
        weighted_metrics=["sparse_categorical_accuracy"],
        jit_compile=True,
    )

    time_based_checkpoint = TimeBasedCheckpoint(
        filepath=checkpoint_filepath,
        save_freq_minutes=10,
    )

    pretraining_model.fit(
        pretrain_ds,
        validation_data=pretrain_val_ds,
        epochs=config["pretraining_epochs"],
        callbacks=[time_based_checkpoint],
    )
    pretraining_model.save(checkpoint_filepath)


# Main function to execute the workflow
def pretrain_BERT(config):
    # Split the dataset
    train_lines, val_lines, test_lines = split_dataset(
        config["dataset_path"], config["split_ratio"]
    )

    # Save the split datasets
    data_files = save_split_datasets(
        train_lines, val_lines, test_lines, config["output_dir"]
    )

    # Generate vocabulary
    vocab_file_path = generate_vocabulary(data_files["train"], config["output_dir"])

    # Load and preprocess the data
    datasets = load_and_preprocess_data(data_files, config["pretraining_batch_size"])
    tokenizer = get_tokenizer(vocab_file_path, config["seq_length"])
    masker = get_masker(tokenizer, config["mask_rate"], config["predictions_per_seq"])

    pretrain_ds = (
        datasets["train"]
        .map(
            lambda x: preprocess(tokenizer, masker, x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    pretrain_val_ds = (
        datasets["validation"]
        .map(
            lambda x: preprocess(tokenizer, masker, x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    encoder_model = build_bert_model(config, tokenizer)
    train_bert_model(
        pretrain_ds,
        pretrain_val_ds,
        encoder_model,
        config,
        config["checkpoint_filepath"],
    )
