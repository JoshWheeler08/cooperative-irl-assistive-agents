from abc import ABC, abstractmethod
import pprint

class IRLImplementation(ABC):
    
    """ Abstract Class for standardising the implementation of my IRL algorithms """

    def __init__(self, owner_agent, env_obj, training_args):
        self.owner_agent = owner_agent # Commonly used to generate a set of expert demonstrations
        self.env_obj = env_obj  # Used to create the KAZ environment 
        self.args = training_args # Allows a user to configure the IRL method's hyperparameters

    @abstractmethod
    def train(self):
        # To be implemented
        pass

    def output_config_info(self):
        """ Outputs the IRL configuration information stored in 'args' variable """
        print("\n[IRL Configuration Information]\n")
        pprint.pprint(self.args)
        print("\n")
