"""Pipeline Class Definition"""

# On TOP of all of this, the user shall define FLOW.py which should
# give him the desired trajectory operation output
import sys
import os

from .utilFunctions import tokenize_trajectory, detokenize_trajectory
from .constraintsClass import SpatialConstraints
from .partioningClass import PartitioningModule

import logging
import warnings
import pickle
import random
import datetime
import string
import pandas as pd
from typing import Tuple, List

# Configure the logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TrajectoryPipeline:
    """
    A configurable pipeline that orchestrates various processes such as
    tokenization, spatial constraints,
    trajectory plugins, and de-tokenization. This class is designed to be
    flexible and extensible, allowing
    the user to customize and modify different components according to their needs.

    Attributes:
        mode (str): Either 'training' or 'testing' mode.
        use_tokenization (bool): Indicates if tokenization should be used.
        use_spatial_constraints (bool): Indicates if spatial constraints should be used.
        use_trajectory_plugin (bool): Indicates if the trajectory plugin should be used.
        use_detokenization (bool): Indicates if de-tokenization should be used.
        modify_transformers_plugin (bool): Indicates if transformers plugin should be modified.
        modify_trajectory_plugin (bool): Indicates if trajectory plugin should be modified.
        modify_spatial_constraints (bool): Indicates if spatial constraints should be modified.
        use_predefined_spatial_constraints (bool): Indicates if predefined spatial
                                                constraints should be used.
        ...
    """

    def __init__(
        self,
        mode: str = "pretraining",
        operation_type: str = "",
        use_tokenization: bool = True,
        use_spatial_constraints: bool = True,
        use_trajectory_plugin=False,
        use_detokenization: bool = True,
        modify_transformers_plugin: bool = False,
        modify_trajectory_plugin: bool = False,
        modify_spatial_constraints: bool = False,
        use_predefined_spatial_constraints: bool = False,
        project_path: str = "/home/",
    ):
        """
        Initializes the pipeline with needed params.

        Args:
            mode (str): Either 'pretraining', 'finetuning', or 'testing'.
            operation_type (str): User sets the operationt type
            use_tokenization (bool): Whether to use tokenization.
            use_spatial_constraints (bool): Whether to use spatial constraints.
            use_trajectory_plugin (bool): Whether to use trajectory plugin.
            use_detokenization (bool): Whether to use de-tokenization.
            modify_transformers_plugin (bool): Whether to modify transformers plugin.
            modify_trajectory_plugin (bool): Whether to modify trajectory plugin.
            modify_spatial_constraints (bool): Whether to modify spatial constraints.
            use_predefined_spatial_constraints (bool): Whether to user predefined
                                                        spatial constraints or not.
            project_path (str): where to save modelsRepo and trajectoryStore
        """

        self.mode = mode
        self.use_tokenization = use_tokenization
        self.use_spatial_constraints = use_spatial_constraints
        self.use_trajectory_plugin = use_trajectory_plugin
        self.use_detokenization = use_detokenization
        self.modify_transformers_plugin = modify_transformers_plugin
        self.modify_trajectory_plugin = modify_trajectory_plugin
        self.modify_spatial_constraints = modify_spatial_constraints
        self.use_predefined_spatial_constraints = use_predefined_spatial_constraints
        self.operation_type = operation_type
        # Create ModelsRepo and trajectoryStore at specified projectPath if they don't exist already
        if project_path == "/home/":
            raise Warning("No project path provided\n Will default to /home/ directory")
        else:
            self.models_repository_path = os.path.join(project_path, "modelsRepo")
            self.trajecotry_store_path = os.path.join(project_path, "trajectoryStore")
            if not os.path.exists(self.models_repository_path):
                os.makedirs(self.models_repository_path)
            if not os.path.exists(self.trajecotry_store_path):
                os.makedirs(self.trajecotry_store_path)
        # Initialize any other attributes or perform setup based on the parameters
        self.model = None
        self.tokenizer = None
        self.spatial_constraints = None
        self.trajectory_plugin = None
        self.tokenized_trajectories = None
        self.trajectories_list = None
        self.input_attributes = None
        self.resolution = 10
        self.resolution_set_by_user = False
        self.spatial_constraints = None
        self.user_did_define_spatial_constraints = False
        self.trajectories_got_tokenized = False
        self.data_saved_to_trajectory_store = False
        # current_directory = os.path.dirname(os.path.abspath(__file__))
        # self.models_repository_path = os.path.join(current_directory, "modelsRepo")
        # self.trajecotry_store_path = os.path.join(current_directory, "trajectoryStore")
        self.data_path_trajectory_store, self.metadata_path_trajectory_store = "", ""
        logging.info("Initializing the pipeline with mode: %s", self.mode)

        if self.use_detokenization:
            self.__setup_detokenization()

        if self.use_spatial_constraints and not self.modify_spatial_constraints:
            # self.spatialConstraints =
            self.__setup_spatial_constraints()

        if self.use_trajectory_plugin:
            self.__setup_trajectory_plugin()

        if self.modify_transformers_plugin:
            self.__modify_transformers_plugin()

        if self.modify_trajectory_plugin:
            self.modify_trajectory_plugin()
        # self.__partioning_module_interface()

    def __run_pretraining(self):
        """
        Set up components and configurations specific to pretraining mode.
        """
        logging.info("Initializing for pretraining mode.")
        # Implementation for initializing pretraining mode

        if self.use_tokenization and self.trajectories_list is not None:
            logging.info("Tokenizing the provided trajectories.")
            self.tokenized_trajectories = self.__tokenization_module(
                self.trajectories_list
            )
            self.trajectories_got_tokenized = True
            self.data_path_trajectory_store, self.metadata_path_trajectory_store = (
                self.__save_trajectories_to_store(
                    self.tokenized_trajectories, "pretraining"
                )
            )

        # @Youssef DO: I need to call the partioning module, need to think about this
        # in the case he is not using tokenization, i.e. passing traj,summary
        return self.__partioning_module_interface()

    def __run_testing(self):
        """
        Set up components and configurations specific to testing mode.
        """
        logging.info("Initializing for testing mode.")
        # Implementation for initializing testing mode
        if self.use_tokenization and self.trajectories_list is not None:
            logging.info("Tokenizing the provided trajectories.")
            self.tokenized_trajectories = self.__tokenization_module(
                self.trajectories_list
            )
        else:
            pass
            # @Youssef DO: I need to get the attributes here
        return

    def __save_trajectories_to_store(self, dataset, operation):
        if self.use_tokenization and dataset is not None:
            self.data_saved_to_trajectory_store = False
            # Generate a random dataset name
            dataset_name = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=10)
            )
            dataset_filename = os.path.join(
                self.trajecotry_store_path, operation, f"{dataset_name}.pkl"
            )
            # Ensure the directory exists
            os.makedirs(os.path.dirname(dataset_filename), exist_ok=True)

            # Save the tokenized trajectories to a .pkl file
            with open(dataset_filename, "wb") as f:
                pickle.dump(dataset, f)

            # Create metadata
            metadata = {
                "total_number_of_trajectories": len(dataset),
                "total_number_of_tokens": sum(len(traj) for traj in dataset),
                "date_of_data_storage": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "type_of_data": operation,
            }

            # Save metadata to a .txt file
            metadata_filename = os.path.join(
                self.trajecotry_store_path,
                operation,
                f"{dataset_name}_metadata.txt",
            )
            # Ensure the directory exists
            os.makedirs(os.path.dirname(metadata_filename), exist_ok=True)
            with open(metadata_filename, "w", encoding="utf-8") as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")

            logging.info(
                "Tokenized trajectories saved to %s with metadata.", dataset_filename
            )
            self.data_saved_to_trajectory_store = True
            return dataset_filename, metadata_filename

    def set_tokenization_resolution(self, resolution: int = 10):
        """
        Sets the resolution to be used if tokenization is enabled.

        Args:
            resolution (int):resolution for the tokenization.

        Returns:
            None
        """
        if not self.use_tokenization:
            raise ValueError("Tokenization is not used. No need to set resolution.")
        self.resolution = resolution
        self.resolution_set_by_user = True

    def get_trajectories_from_csv(
        self, file_path: str = ""
    ) -> List[List[Tuple[float, float]]]:
        """
        Reads a CSV file containing trajectories and returns the trajectories as a list of lists of tuples.

        Each trajectory is represented as a list of (latitude, longitude) tuples.

        Args:
            file_path (str): A CSV file path containing the trajectories.

        Returns:
            List[List[Tuple[float, float]]]: List of trajectories, where each trajectory is a list of (latitude, longitude) tuples.
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Initialize a list to hold the trajectories
        trajectories = []

        # Process each row in the DataFrame
        for _, row in df.iterrows():
            # Extract the trajectory string
            trajectory_str = row["trajectory"]

            # Split the trajectory string into individual points
            points_str = trajectory_str.split(",")

            # Convert the points to (latitude, longitude) tuples
            trajectory = [
                (float(point.split()[0]), float(point.split()[1]))
                for point in points_str
            ]

            # Append the trajectory to the list
            trajectories.append(trajectory)

        return trajectories

    def set_trajectories(self, trajectories: list[list[tuple[float, float]]]):
        """
        Sets the list of trajectories to be used if tokenization is enabled.

        Args:
            trajectories (list of list of tuples): List of trajectories, where each
                                trajectory is a list of (latitude, longitude) tuples.

        Returns:
            None
        """
        if not self.use_tokenization:
            raise ValueError("Tokenization is not used. No need to set trajectories.")
        self.trajectories_list = trajectories
        logging.info("Trajectories set for tokenization.")

    def __setup_detokenization(self):
        """
        Set up de-tokenization components and configurations.
        """
        # logging.info("Setting up de-tokenization.")
        # Implementation for setting up de-tokenization

    def __setup_spatial_constraints(self):
        """
        Set up spatial constraints components and configurations.
        """
        logging.info("Setting up spatial constraints, reverting to default constraints")
        # Implementation for setting up spatial constraints
        # from constra

        self.spatial_constraints = SpatialConstraints(
            rules=None, usepredefined_rules=True
        )
        self.user_did_define_spatial_constraints = True
        # Example usage
        token = "8f2830831ffffff"  # --> This should come from the model
        previous_tokens = [
            "8f2830830ffffff",
            "8f2830831ffffff",
        ]  # --> This should be previously stored in the class
        result, rule = self.spatial_constraints.check_token(token, previous_tokens)
        if result is False:
            print("Token didn't pass the following constraint: ", rule)
        else:
            print("Token meets all conditions")

    def define_spatial_constraints(self, rules=None):
        """
        Modify spatial constraints based on passed rules.
        Args:
            rules (list of callables, optional): A list of functions that take a token
            and previous tokens as input and return True if the condition is met, otherwise False.
        """
        logging.info("Adding user defined spatial constraints..")

        self.spatial_constraints = SpatialConstraints(
            rules, usepredefined_rules=self.use_predefined_spatial_constraints
        )
        self.user_did_define_spatial_constraints = True
        # Example usage
        token = "8f2830831ffffff"  # --> This should come from the model
        previous_tokens = [
            "8f2830830ffffff",
            "8f2830831ffffff",
        ]  # --> This should be previously stored in the class
        result, rule = self.spatial_constraints.check_token(
            token, previous_tokens
        )  # --> This should be applied after each model output
        if result is False:
            print("Token didn't pass the following constraint: ", rule)
        else:
            print("Token meets all conditions")
            # Then continue your operation

    def __setup_trajectory_plugin(self):
        """
        Set up trajectory plugin components and configurations.
        """
        logging.info("Setting up trajectory plugin.")
        logging.info(self)
        # Implementation for setting up trajectory plugin

    def __modify_transformers_plugin(self):
        """
        Modify transformers plugin based on configurations.
        """
        logging.info("Modifying transformers plugin.")
        logging.info(self)
        # Implementation for modifying transformers plugin

    def define_trajectory_plugin(self):
        """
        Modify trajectory plugin based on configurations.
        """
        logging.info("Modifying trajectory plugin.")
        logging.info(self)
        # Implementation for modifying trajectory plugin

    def __tokenization_module(
        self, trajectories: list[list[tuple[float, float]]]
    ) -> list[list[str]]:
        """
        Tokenizes a list of trajectories.

        Args:
            trajectories (list of list of tuple[float, float]]): A list of trajectories,
                                                    where each trajectory is a list of
            (latitude, longitude) tuples.

        Returns:
            list of list of str: A list of tokenized trajectories, where
                                    each trajectory is a list of tokens.
        """

        if not self.resolution_set_by_user:
            info = "Tokenization Resolution Set By Default to: " + self.resolution
            logging.info(info)
        tokenized_trajectories = [
            tokenize_trajectory(trajectory, self.resolution)
            for trajectory in trajectories
        ]
        return tokenized_trajectories

    def __detokenization_module(
        self, tokenized_trajectories: list[list[str]]
    ) -> list[list[tuple[float, float]]]:
        """
        Detokenizes a list of tokenized trajectories.

        Args:
            tokenized_trajectories (list of list of str): A list of tokenized
                        trajectories, where each trajectory is a list of tokens.

        Returns:
            list of list of tuple[float, float]]: A list of detokenized trajectories,
                            where each trajectory is a list of
            (latitude, longitude) tuples.
        """

        detokenized_trajectories = [
            detokenize_trajectory(tokenized_trajectory)
            for tokenized_trajectory in tokenized_trajectories
        ]
        return detokenized_trajectories

    def __partioning_module_interface(self):
        """
        Provides an interface to the paritioning Module.
        The user doesn't have access to this function
        """
        model_path, dataset_path = None, None
        if (
            self.mode == "pretraining"
            and self.trajectories_got_tokenized
            and self.data_saved_to_trajectory_store
        ):
            # Only in the case of pretraining add to pretraining pyramids
            module = PartitioningModule(
                models_repo_path=self.models_repository_path, operation=self.mode
            )
            logging.info("Updating pyramid modelsRepo with the new dataset")
            model_path = module.update_pretraining_repository(
                self.data_path_trajectory_store, self.metadata_path_trajectory_store
            )
            print(model_path)
            dataset_path = self.data_path_trajectory_store
        elif self.mode == "finetuning":
            # @Youssef DO: Need to think about handling this here
            print("TODO")
        elif self.mode == "testing" and self.trajectories_got_tokenized:
            module = PartitioningModule(
                models_repo_path=self.models_repository_path, operation=self.mode
            )
            logging.info("Fetching proper model from the models repo")
            model_path = module.find_proper_model(
                self.tokenized_trajectories, self.operation_type
            )
            # @Youssef DO: I need a way to load the model after getting its path, TensorFlow
            logging.info("Found proper model in the repo")
            print(model_path)
            dataset_path = None
        else:  # i.e. user entered other attributes
            # @Youssef DO: I need to think about this case
            pass
        return model_path, dataset_path

    def run(self):
        """
        Where the user is ale to run the pipeline
        """
        if (
            self.modify_spatial_constraints
            and not self.user_did_define_spatial_constraints
            and not self.use_predefined_spatial_constraints
        ):
            raise ValueError(
                "User requested to create spatial constraints from scratch without defining any constraints."
            )
        if (
            self.modify_spatial_constraints
            and not self.user_did_define_spatial_constraints
        ):
            warnings.warn(
                "User requested to modify spatial constraints without defining any constraints."
            )
        if self.mode == "pretraining":
            return self.__run_pretraining()
        elif self.mode == "testing":
            return self.__run_testing()

        logging.info("Pipeline Started Running Successfully")
