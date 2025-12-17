
class EnvObject:
    """ This is a Python Class for storing all the related information needed to create and use an environment """

    def __init__(
        self, 
        env_id="insert here", 
        n_envs=4, 
        seed=42, 
        wrapper_class=None, 
        env_kwargs=None, 
        wrapper_kwargs=None, 
        total_timesteps=1000,
        logger=None,
    ):
        self.env_id = env_id
        self.n_envs = n_envs # number of vectorised environments
        self.seed = seed
        self.wrapper_class = wrapper_class
        self.env_kwargs = env_kwargs
        self.wrapper_kwargs = wrapper_kwargs
        self.total_timesteps = total_timesteps # how long the agent using the EnvObject should train on the environment
        self.logger = logger

    # Get the environment's id
    def env_id(self):
        return self.env_id

    # Gets the value determining the number of vectorised instances of the environment
    def n_envs(self):
        return self.n_envs

    # Gets the random seed to be used in the environment
    def seed(self):
        return self.seed

    # Gets the wrapper class (if there is one) to be added around the environment
    def wrapper_class(self):
        return self.wrapper_class
    
    # Gets the environment's keyword arguments 
    def env_kwargs(self):
        return self.env_kwargs

    # Get's the environment's wrapper's keyword arguments
    def wrapper_kwargs(self):
        return self.wrapper_kwargs
    
    # Gets the total number of timesteps that an agent using the EnvObject should train on the environment
    def total_timesteps(self):
        return self.total_timesteps

    # Gets the environment's logger
    def logger(self):
        return self.logger