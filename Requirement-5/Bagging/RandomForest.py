import numpy as np
from typing import Optional
from sklearn.tree import DecisionTreeClassifier
from Bagging import Bagging


class RandomForestClassifier(Bagging):
    def __init__(
        self,
        # Bagging Hyperparameters
        n_estimators: int = 100,
        max_samples: float = 1.0,
        # Tree Hyperparameters
        max_depth: Optional[int] = None,
        max_features: Optional[int] = None,
        min_samples_split: int = 2,
        # Common
        random_state: Optional[int] = None,
        **kwargs
    ):
        # Init new parameters. You will use these in the next two TODOs
        self.n_estimators = n_estimators
        self.max_features = max_features or "sqrt"
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        # TODO 1.0: Create a new DecisionTreeClassifier and pass to it the relevant hyperparameters (from self)
        # TODO 1.1: Which of the features passed is responsible for column subsampling? "max_features"
        self.base_model : DecisionTreeClassifier = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )

        # TODO 2: Initialize the Bagging base class (from self)
        super().__init__(
            model=self.base_model,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )

    # Ignore this function. It allows setting parameters for debugging or visualization.
    def set_params(self, **params):
        # Get the current parameters
        current_params = self.__dict__.copy()
        current_params.update(params)
        self.__init__(**current_params)
