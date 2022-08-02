import os
import pickle
import itertools
import logging
import re
import time
import glob
import inspect

# used to parallelize evaluation
from joblib import Parallel, delayed

# numerical methods and arrays
import numpy as np
import pandas as pd

# import packages used for the implementation of sampling methods
from sklearn.model_selection import (RepeatedStratifiedKFold, KFold,
                                     cross_val_score, StratifiedKFold)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, ClassifierMixin


# for handler in _logger.root.handlers[:]:
#    _logger.root.removeHandler(handler)

# setting the _logger format
_logger = logging.getLogger('smote_variants')
_logger.setLevel(logging.DEBUG)
_logger_ch = logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)


def mode(data):
    values, counts = np.unique(data, return_counts=True)
    return values[np.where(counts == max(counts))[0][0]]

class StatisticsMixin:
    """
    Mixin to compute class statistics and determine minority/majority labels
    """

    def class_label_statistics(self, X, y):
        """
        determines class sizes and minority and majority labels
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        unique, counts = np.unique(y, return_counts=True)
        self.class_stats = dict(zip(unique, counts))
        self.min_label = unique[0] if counts[0] < counts[1] else unique[1]
        self.maj_label = unique[1] if counts[0] < counts[1] else unique[0]
        # shorthands
        self.min_label = self.min_label
        self.maj_label = self.maj_label

    def check_enough_min_samples_for_sampling(self, threshold=2):
        if self.class_stats[self.min_label] < threshold:
            m = ("The number of minority samples (%d) is not enough "
                 "for sampling")
            m = m % self.class_stats[self.min_label]
            _logger.warning(self.__class__.__name__ + ": " + m)
            return False
        return True


class RandomStateMixin:
    """
    Mixin to set random state
    """

    def set_random_state(self, random_state):
        """
        sets the random_state member of the object

        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError(
                "random state cannot be initialized by " + str(random_state))


class ParameterCheckingMixin:
    """
    Mixin to check if parameters come from a valid range
    """

    def check_in_range(self, x, name, r):
        """
        Check if parameter is in range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x < r[0] or x > r[1]:
            m = ("Value for parameter %s outside the range [%f,%f] not"
                 " allowed: %f")
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_out_range(self, x, name, r):
        """
        Check if parameter is outside of range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x >= r[0] and x <= r[1]:
            m = "Value for parameter %s in the range [%f,%f] not allowed: %f"
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal(self, x, name, val):
        """
        Check if parameter is less than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x > val:
            m = "Value for parameter %s greater than %f not allowed: %f > %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x > y:
            m = ("Value for parameter %s greater than parameter %s not"
                 " allowed: %f > %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less(self, x, name, val):
        """
        Check if parameter is less than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x >= val:
            m = ("Value for parameter %s greater than or equal to %f"
                 " not allowed: %f >= %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x >= y:
            m = ("Value for parameter %s greater than or equal to parameter"
                 " %s not allowed: %f >= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal(self, x, name, val):
        """
        Check if parameter is greater than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x < val:
            m = "Value for parameter %s less than %f is not allowed: %f < %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x < y:
            m = ("Value for parameter %s less than parameter %s is not"
                 " allowed: %f < %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater(self, x, name, val):
        """
        Check if parameter is greater than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x <= val:
            m = ("Value for parameter %s less than or equal to %f not allowed"
                 " %f < %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_par(self, x, name_x, y, name_y):
        """
        Check if parameter is greater than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x <= y:
            m = ("Value for parameter %s less than or equal to parameter %s"
                 " not allowed: %f <= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal(self, x, name, val):
        """
        Check if parameter is equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x == val:
            m = ("Value for parameter %s equal to parameter %f is not allowed:"
                 " %f == %f")
            m = m % (name, val, x, val)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x == y:
            m = ("Value for parameter %s equal to parameter %s is not "
                 "allowed: %f == %f")
            m = m % (name_x, name_y, x, y)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_isin(self, x, name, li):
        """
        Check if parameter is in list
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            li (list): list to check if parameter is in it
        Throws:
            ValueError
        """
        if x not in li:
            m = "Value for parameter %s not in list %s is not allowed: %s"
            m = m % (name, str(li), str(x))
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_n_jobs(self, x, name):
        """
        Check n_jobs parameter
        Args:
            x (int/None): number of jobs
            name (str): the parameter name
        Throws:
            ValueError
        """
        if not ((x is None)
                or (x is not None and isinstance(x, int) and not x == 0)):
            m = "Value for parameter n_jobs is not allowed: %s" % str(x)
            raise ValueError(self.__class__.__name__ + ": " + m)


class ParameterCombinationsMixin:
    """
    Mixin to generate parameter combinations
    """

    @classmethod
    def generate_parameter_combinations(cls, dictionary, raw):
        """
        Generates reasonable paramter combinations
        Args:
            dictionary (dict): dictionary of paramter ranges
            num (int): maximum number of combinations to generate
        """
        if raw:
            return dictionary
        keys = sorted(list(dictionary.keys()))
        values = [dictionary[k] for k in keys]
        combinations = [dict(zip(keys, p))
                        for p in list(itertools.product(*values))]
        return combinations


class NoiseFilter(StatisticsMixin,
                  ParameterCheckingMixin,
                  ParameterCombinationsMixin):
    """
    Parent class of noise filtering methods
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def remove_noise(self, X, y):
        """
        Removes noise
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        pass

    def get_params(self, deep=False):
        """
        Return parameters

        Returns:
            dict: dictionary of parameters
        """

        return {}

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self



class OverSampling(StatisticsMixin,
                   ParameterCheckingMixin,
                   ParameterCombinationsMixin,
                   RandomStateMixin):
    """
    Base class of oversampling methods
    """

    categories = []

    cat_noise_removal = 'NR'
    cat_dim_reduction = 'DR'
    cat_uses_classifier = 'Clas'
    cat_sample_componentwise = 'SCmp'
    cat_sample_ordinary = 'SO'
    cat_sample_copy = 'SCpy'
    cat_memetic = 'M'
    cat_density_estimation = 'DE'
    cat_density_based = 'DB'
    cat_extensive = 'Ex'
    cat_changes_majority = 'CM'
    cat_uses_clustering = 'Clus'
    cat_borderline = 'BL'
    cat_application = 'A'

    def __init__(self):
        pass

    def det_n_to_sample(self, strategy, n_maj, n_min):
        """
        Determines the number of samples to generate
        Args:
            strategy (str/float): if float, the fraction of the difference
                                    of the minority and majority numbers to
                                    generate, like 0.1 means that 10% of the
                                    difference will be generated if str,
                                    like 'min2maj', the minority class will
                                    be upsampled to match the cardinality
                                    of the majority class
        """
        if isinstance(strategy, float) or isinstance(strategy, int):
            return max([0, int((n_maj - n_min)*strategy)])
        else:
            m = "Value %s for parameter strategy is not supported" % strategy
            raise ValueError(self.__class__.__name__ + ": " + m)

    def sample_between_points(self, x, y):
        """
        Sample randomly along the line between two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
        Returns:
            np.array: the new sample
        """
        return x + (y - x)*self.random_state.random_sample()

    def sample_between_points_componentwise(self, x, y, mask=None):
        """
        Sample each dimension separately between the two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
            mask (np.array): array of 0,1s - specifies which dimensions
                                to sample
        Returns:
            np.array: the new sample being generated
        """
        if mask is None:
            return x + (y - x)*self.random_state.random_sample()
        else:
            return x + (y - x)*self.random_state.random_sample()*mask

    def sample_by_jittering(self, x, std):
        """
        Sample by jittering.
        Args:
            x (np.array): base point
            std (float): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample() - 0.5)*2.0*std

    def sample_by_jittering_componentwise(self, x, std):
        """
        Sample by jittering componentwise.
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample(len(x))-0.5)*2.0 * std

    def sample_by_gaussian_jittering(self, x, std):
        """
        Sample by Gaussian jittering
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return self.random_state.normal(x, std)

    def sample(self, X, y):
        """
        The samplig function reimplemented in child classes
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        return X, y

    def fit_resample(self, X, y):
        """
        Alias of the function "sample" for compatibility with imbalanced-learn
        pipelines
        """
        return self.sample(X, y)

    def sample_with_timing(self, X, y):
        begin = time.time()
        X_samp, y_samp = self.sample(X, y)
        _logger.info(self.__class__.__name__ + ": " +
                     ("runtime: %f" % (time.time() - begin)))
        return X_samp, y_samp

    def preprocessing_transform(self, X):
        """
        Transforms new data according to the possible transformation
        implemented by the function "sample".
        Args:
            X (np.matrix): features
        Returns:
            np.matrix: transformed features
        """
        return X

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

    def __str__(self):
        return self.descriptor()



class SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{smote,
                author={Chawla, N. V. and Bowyer, K. W. and Hall, L. O. and
                            Kegelmeyer, W. P.},
                title={{SMOTE}: synthetic minority over-sampling technique},
                journal={Journal of Artificial Intelligence Research},
                volume={16},
                year={2002},
                pages={321--357}
              }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0
            means that after sampling the number of minority samples will
                                 be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            # _logger.warning(self.__class__.__name__ +
            #                ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the model
        n_neigh = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        if n_to_sample == 0:
            return X.copy(), y.copy()

        # generating samples
        base_indices = self.random_state.choice(list(range(len(X_min))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, n_neigh)),
                                                    n_to_sample)

        X_base = X_min[base_indices]
        X_neighbor = X_min[ind[base_indices, neighbor_indices]]

        samples = X_base + np.multiply(self.random_state.rand(n_to_sample,
                                                              1),
                                       X_neighbor - X_base)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class MWMOTE(OverSampling):
    
    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 k1=5,
                 k2=5,
                 k3=5,
                 M=10,
                 cf_th=5.0,
                 cmax=10.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            k1 (int): parameter of the NearestNeighbors component
            k2 (int): parameter of the NearestNeighbors component
            k3 (int): parameter of the NearestNeighbors component
            M (int): number of clusters
            cf_th (float): cutoff threshold
            cmax (float): maximum closeness value
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(k1, 'k1', 1)
        self.check_greater_or_equal(k2, 'k2', 1)
        self.check_greater_or_equal(k3, 'k3', 1)
        self.check_greater_or_equal(M, 'M', 1)
        self.check_greater_or_equal(cf_th, 'cf_th', 0)
        self.check_greater_or_equal(cmax, 'cmax', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.M = M
        self.cf_th = cf_th
        self.cmax = cmax
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'k1': [5, 9],
                                  'k2': [5, 9],
                                  'k3': [5, 9],
                                  'M': [4, 10],
                                  'cf_th': [5.0],
                                  'cmax': [10.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority = np.where(y == self.min_label)[0]

        # Step 1
        n_neighbors = min([len(X), self.k1 + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              n_jobs=self.n_jobs)
        nn.fit(X)
        dist1, ind1 = nn.kneighbors(X)

        # Step 2
        arr = [i for i in minority if np.sum(y[ind1[i][1:]] == self.min_label)]
        filtered_minority = np.array(arr)

        if len(filtered_minority) == 0:
            _logger.info(self.__class__.__name__ + ": " +
                         "filtered_minority array is empty")
            return X.copy(), y.copy()

        # Step 3 - ind2 needs to be indexed by indices of the lengh of X_maj
        nn_maj = NearestNeighbors(n_neighbors=self.k2, n_jobs=self.n_jobs)
        nn_maj.fit(X_maj)
        dist2, ind2 = nn_maj.kneighbors(X[filtered_minority])

        # Step 4
        border_majority = np.unique(ind2.flatten())

        # Step 5 - ind3 needs to be indexed by indices of the length of X_min
        n_neighbors = min([self.k3, len(X_min)])
        nn_min = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn_min.fit(X_min)
        dist3, ind3 = nn_min.kneighbors(X_maj[border_majority])

        # Step 6 - informative minority indexes X_min
        informative_minority = np.unique(ind3.flatten())

        def closeness_factor(y, x, cf_th=self.cf_th, cmax=self.cmax):
            """
            Closeness factor according to the Eq (6)

            Args:
                y (np.array): training instance (border_majority)
                x (np.array): training instance (informative_minority)
                cf_th (float): cutoff threshold
                cmax (float): maximum values

            Returns:
                float: closeness factor
            """
            d = np.linalg.norm(y - x)/len(y)
            if d == 0.0:
                d = 0.1
            if 1.0/d < cf_th:
                f = 1.0/d
            else:
                f = cf_th
            return f/cf_th*cmax

        # Steps 7 - 9
        _logger.info(self.__class__.__name__ + ": " +
                     'computing closeness factors')
        closeness_factors = np.zeros(
            shape=(len(border_majority), len(informative_minority)))
        for i in range(len(border_majority)):
            bm_i = border_majority[i]
            for j in range(len(informative_minority)):
                im_j = informative_minority[j]
                closeness_factors[i, j] = closeness_factor(X_maj[bm_i],
                                                           X_min[im_j])

        _logger.info(self.__class__.__name__ + ": " +
                     'computing information weights')
        information_weights = np.zeros(
            shape=(len(border_majority), len(informative_minority)))
        for i in range(len(border_majority)):
            norm_factor = np.sum(closeness_factors[i, :])
            for j in range(len(informative_minority)):
                cf_ij = closeness_factors[i, j]
                information_weights[i, j] = cf_ij**2/norm_factor

        selection_weights = np.sum(information_weights, axis=0)
        selection_probabilities = selection_weights/np.sum(selection_weights)

        # Step 10
        _logger.info(self.__class__.__name__ + ": " + 'do clustering')
        n_clusters = min([len(X_min), self.M])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.random_state)
        kmeans.fit(X_min)
        imin_labels = kmeans.labels_[informative_minority]

        clusters = [np.where(imin_labels == i)[0]
                    for i in range(np.max(kmeans.labels_)+1)]

        # Step 11
        samples = []

        # Step 12
        for i in range(n_to_sample):
            random_index = self.random_state.choice(informative_minority,
                                                    p=selection_probabilities)
            cluster_label = kmeans.labels_[random_index]
            cluster = clusters[cluster_label]
            random_index_in_cluster = self.random_state.choice(cluster)
            X_random = X_min[random_index]
            X_random_cluster = X_min[random_index_in_cluster]
            samples.append(self.sample_between_points(X_random,
                                                      X_random_cluster))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k1': self.k1,
                'k2': self.k2,
                'k3': self.k3,
                'M': self.M,
                'cf_th': self.cf_th,
                'cmax': self.cmax,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}



class Borderline_SMOTE1(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling Method
                                     in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 k_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): control parameter of the nearest neighbor
                                    technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor
                                    technique for sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'k_neighbors': [3, 5, 7]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining number of samples to be generated
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # fitting model
        X_min = X[y == self.min_label]

        n_neighbors = min([len(X), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X_min)

        # determining minority samples in danger
        noise = []
        danger = []
        for i in range(len(indices)):
            if self.n_neighbors == sum(y[indices[i][1:]] == self.maj_label):
                noise.append(i)
            elif mode(y[indices[i][1:]]) == self.maj_label:
                danger.append(i)
        X_danger = X_min[danger]
        X_min = np.delete(X_min, np.array(noise).astype(int), axis=0)

        if len(X_danger) == 0:
            _logger.info(self.__class__.__name__ +
                         ": " + "No samples in danger")
            return X.copy(), y.copy()

        # fitting nearest neighbors model to minority samples
        k_neigh = min([len(X_min), self.k_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=k_neigh, n_jobs=self.n_jobs)
        nn.fit(X_min)
        # extracting neighbors of samples in danger
        distances, indices = nn.kneighbors(X_danger)

        # generating samples near points in danger
        base_indices = self.random_state.choice(list(range(len(X_danger))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, k_neigh)),
                                                    n_to_sample)

        X_base = X_danger[base_indices]
        X_neighbor = X_min[indices[base_indices, neighbor_indices]]

        samples = X_base + \
            np.multiply(self.random_state.rand(
                n_to_sample, 1), X_neighbor - X_base)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'k_neighbors': self.k_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}




class SDSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{sdsmote,
                            author={Li, K. and Zhang, W. and Lu, Q. and
                                        Fang, X.},
                            booktitle={2014 International Conference on
                                        Identification, Information and
                                        Knowledge in the Internet of
                                        Things},
                            title={An Improved SMOTE Imbalanced Data
                                    Classification Method Based on Support
                                    Degree},
                            year={2014},
                            volume={},
                            number={},
                            pages={34-38},
                            keywords={data mining;pattern classification;
                                        sampling methods;improved SMOTE
                                        imbalanced data classification
                                        method;support degree;data mining;
                                        class distribution;imbalanced
                                        data-set classification;over sampling
                                        method;minority class sample
                                        generation;minority class sample
                                        selection;minority class boundary
                                        sample identification;Classification
                                        algorithms;Training;Bagging;Computers;
                                        Testing;Algorithm design and analysis;
                                        Data mining;Imbalanced data-sets;
                                        Classification;Boundary sample;Support
                                        degree;SMOTE},
                            doi={10.1109/IIKI.2014.14},
                            ISSN={},
                            month={Oct}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # fitting nearest neighbors model to find closest majority points to
        # minority samples
        nn = NearestNeighbors(n_neighbors=len(X_maj), n_jobs=self.n_jobs)
        nn.fit(X_maj)
        dist, ind = nn.kneighbors(X_min)

        # calculating the sum according to S3 in the paper
        S_i = np.sum(dist, axis=1)
        # calculating average distance according to S5
        S = np.sum(S_i)
        S_ave = S/(len(X_min)*len(X_maj))

        # calculate support degree
        def support_degree(x):
            return len(nn.radius_neighbors(x.reshape(1, -1),
                                           S_ave,
                                           return_distance=False))

        k = np.array([support_degree(X_min[i]) for i in range(len(X_min))])
        density = k/np.sum(k)

        # fitting nearest neighbors model to minority samples to run
        # SMOTE-like sampling
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(density)), p=density)
            random_neighbor_idx = self.random_state.choice(ind[idx][1:])
            X_a = X_min[idx]
            X_b = X_min[random_neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

class EditedNearestNeighbors(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, remove='both', n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            remove (str): class to remove from 'both'/'min'/'maj'
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_isin(remove, 'remove', ['both', 'min', 'maj'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.remove = remove
        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        if len(X) < 4:
            _logger.info(self.__class__.__name__ + ': ' +
                         "Not enough samples for noise removal")
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=4, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        to_remove = []
        for i in range(len(X)):
            if not y[i] == mode(y[indices[i][1:]]):
                if (self.remove == 'both' or
                    (self.remove == 'min' and y[i] == self.min_label) or
                        (self.remove == 'maj' and y[i] == self.maj_label)):
                    to_remove.append(i)

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)

    def get_params(self):
        """
        Get noise removal parameters

        Returns:
            dict: dictionary of parameters
        """
        return {'remove': self.remove}



class SMOTE_ENN(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_tomeklinks_enn,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA},
                    }

    Notes:
        * Can remove too many of minority samples.
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return SMOTE.parameter_combinations(raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        smote = SMOTE(self.proportion, self.n_neighbors,
                      n_jobs=self.n_jobs, random_state=self.random_state)
        X_new, y_new = smote.sample(X, y)

        enn = EditedNearestNeighbors(n_jobs=self.n_jobs)

        return enn.remove_noise(X_new, y_new)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

