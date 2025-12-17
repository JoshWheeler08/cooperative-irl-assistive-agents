class PolicyObject:

    """ This is a Python Class for storing a policy with all its related information """

    def __init__(self, policy, policy_type, total_timesteps):
        self.policy = policy # PPO, DQN or A2C
        self.policy_type = policy_type # MlpPolicy, CnnPolicy, MultiInputPolicy
        self.total_timesteps = total_timesteps # how long the policy was trained for

    @property
    def policy(self):
        return self.policy

    @property
    def policy_type(self):
        return self.policy_type

    @property
    def total_timesteps(self):
        return self.total_timesteps

