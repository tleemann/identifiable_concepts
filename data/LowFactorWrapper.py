# A wrapper around a dataset object that pretends there are only some of the given factors.
# This can be used to calculate, e.g., DCI on only some interesting factors instead of all.

from disentanglement_lib.data.ground_truth import ground_truth_data

class LowFactorWrapper(ground_truth_data.GroundTruthData):
    def __init__(self, data, factor_ids) -> None:
        # Input:
        # data - a ground_truth_data.GroundTruthData object
        # factor_ids - a list of integers giving the factors we want to "keep".
        #              At least 1 entry and at most as many as data has.
        super().__init__()
        self.data = data
        self.factor_ids = factor_ids

    @property
    def num_factors(self):
        return len(self.factor_ids)

    @property
    def factors_num_values(self):
        all_factors = self.data.factors_num_values
        return [all_factors[i] for i in self.factor_ids]

    @property
    def observation_shape(self):
        return self.data.observation_shape()

    def sample_factors(self, num, random_state):
        return self.data.sample_factors(num, random_state)[:, self.factor_ids]

    def sample(self, num, random_state):
        factors, x = self.data.sample(num, random_state)
        return factors[:, self.factor_ids], x

    def sample_observations_from_factors(self, factors, random_state):
        # Input:
        # factors - a tensor with [batchsize, n_factors]. n_factors may either give the full factors of the original
        #           dataset, or only the ones specified in this wrapper. In the latter case, the remaining factors are
        #           sampled.
        if factors.shape[1] == len(self.factor_ids):
            # Fill up the remaining factors with random states
            full_factors = self.data.sample_factors(factors.shape[0], random_state)
            full_factors[:, self.factor_ids] = factors
            factors = full_factors
        if factors.shape[1] != self.data.num_factors:
            raise ValueError("Gave too many or too few factors.")

        return self.data.sample_observations_from_factors(factors, random_state)

