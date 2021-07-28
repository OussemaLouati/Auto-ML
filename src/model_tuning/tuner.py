import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining,MedianStoppingRule,ASHAScheduler

class Raytuner():
    def __init__(self,model):
        self.model = model
        self.analysis = None
        ray.init()

    def ASHAScheduler(self,n_samples,search_space,grace_period=1):
        self.analysis = tune.run(
        self.model,
        num_samples=n_samples,
        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=grace_period),
        config=search_space)
        return self.analysis
    
    def PopulationBasedTraining(self,perturbation_inter=10,hyperparam_mutations=None):
        self.analysis = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=perturbation_inter, 
            hyperparam_mutations=hyperparam_mutations
            )
        return self.analysis

    def MedianStoppingRules(self,n_samples,search_space):
        self.analysis = tune.run(
        self.model,
        num_samples=n_samples,
        scheduler=MedianStoppingRule(),
        config=search_space)
        return self.analysis

    def get_best_param(self):
        print("Best config: ", self.analysis.get_best_config(metric="mean_accuracy"))
        # Get a dataframe for analyzing trial results.
        return self.analysis.dataframe()