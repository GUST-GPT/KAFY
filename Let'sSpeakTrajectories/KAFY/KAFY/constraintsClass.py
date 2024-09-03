"""Definition of the SpatialCosntraints Module"""

# some examples of dummy predefined rules


def no_repeat_rule(token, previous_tokens):
    """
    Rule: The token should not be repeated in the trajectory.
    """
    return token not in previous_tokens


def far_enough_rule(token, previous_tokens, min_distance=2):
    """
    Rule: The token should be at least `min_distance` away from the previous token.

    This is just an example. You'll need to implement the actual distance calculation
    based on your specific use case.
    """
    if not previous_tokens:
        return True
    # Example distance check; needs a real implementation
    return abs(int(token, 16) - int(previous_tokens[-1], 16)) > min_distance


class SpatialConstraints:
    """
    A class to manage and enforce spatial constraints on tokens within a trajectory.

    This class allows the definition of custom rules to validate tokens
    based on user-defined conditions.
    It also supports predefined rules for common use cases.

    Attributes:
        rules (list of callables): A list of functions that take a token and
        previous tokens as input
        and return True if the condition is met, otherwise False.
    """

    def __init__(self, rules=None, usepredefined_rules: bool = False):
        """
        Initializes the SpatialConstraints module with user-defined rules.

        Args:
            rules (list of callables, optional): A list of functions that take a token
            and previous tokens as input and return True if the condition is met, otherwise False.
        """
        self.rules = None
        predefined_rules = [
            no_repeat_rule,
            lambda token, previous_tokens: far_enough_rule(
                token, previous_tokens, min_distance=5
            ),
        ]

        if usepredefined_rules:
            self.rules = predefined_rules
        if rules is None:
            # @Youssef DO: Define some global rules for all operations to follow
            rules = []
        for rule in rules:
            self.add_rule(rule)

    def add_rule(self, rule):
        """
        Adds a new rule to the list of rules.

        Args:
            rule (callable): A function that takes a token and previous tokens as input
            and returns True if the condition is met, otherwise False.
        """
        self.rules.append(rule)

    def check_token(self, token, previous_tokens):
        """
        Checks if a token meets all user-defined conditions.

        Args:
            token (str): The token to check.
            previous_tokens (list of str): The list of previous tokens in the trajectory.

        Returns:
            bool: True if the token meets all conditions, otherwise False.
        """
        for rule in self.rules:
            if not rule(token, previous_tokens):
                return False, rule
        return True, None
