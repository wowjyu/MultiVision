from scipy import stats
import numpy as np
import itertools
import math
import re
from dateutil.parser import parse
from typing import List

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        try:
            ## for int, only allows four-digit time (i.e. year)
            cast_int = int(string)
            if len(str(cast_int)) == 4:
                return True
            else:
                return False
        except:
            pass
        parse(str(string), fuzzy=fuzzy)
        return True

    except ValueError:
        return False
    
def is_number(l: List):
    """
        Return whether an input list is a number list

        :param l: List, list to check 
    """
    try: 
        [float(x) for x in l]
        return True
    except:
        return False
    
def is_string(l):
    """
        Return whether an input list is a string list

        :param l: List, list to check 
    """
    try: 
        [str(x) for x in l]
        return True
    except:
        return False    

def find_common_prefix(ini_strlist):
    # Finding commom prefix using Naive Approach
    res = ''.join(c[0] for c in itertools.takewhile(lambda x:
            all(x[0] == y for y in x), zip(*ini_strlist)))
    return str(res)

def compute_entropy(labels, base=2):
    value,counts = np.unique(labels, return_counts=True)
    return stats.entropy(counts, base=base)

def compute_incdec(numbers):
    prev = numbers[0]
    inc = 0
    dec = 0
    for i in range(1, len(numbers)):
        if numbers[i] > prev:
            inc = inc + 1
        elif numbers[i] < prev:
            dec = dec + 1
        prev = numbers[i]
    return max(inc, dec)

def compute_benford(values):
    ## BENFORD constants, see <see href="https://en.wikipedia.org/wiki/Benford%27s_law"/>.
    benfordstd = [0.0, 0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    
    first_digits = [str(abs(v))[0] for v in values] ## negative values contain -
    # first_digits = [v for v in values if v.isnumeric()] ## might be -

    value, counts = np.unique(first_digits, return_counts=True)

    skewing = 0
    for idx, count in enumerate(counts):
        delta = count / len(values) - benfordstd[idx]
        skewing = skewing + delta* delta
    return math.sqrt(skewing)
        
def get_equal_progression(values):
    diff = [x - values[i - 1] for i, x in enumerate(values) if i > 0]
    
    return 1 if len(set(diff)) == 1 else 0

def get_geo_progression(values):
    def norm_v(v):
        return v if v != 0 else 1
    values = [norm_v(v) for v in values]
    diff = [x / values[i - 1] for i, x in enumerate(values) if i > 0]

    return 1 if len(set(diff)) == 1 else 0

def get_gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def get_data_features(values: List):
    """
        Get the data feature of a input list (i.e., a data column)

        :param values: List, list to check 

        ----------------
        Additional Notes:
        Features like 'range', 'variance', 'cov', 'lengthStdDev' are in range [-inf, inf] or [0, inf].
        These features may cause problems in NN model and need to be normalized.
        We adopt normalization by distribution here.
        To take range as an example, this feature distributes in [0, inf]. We first square root this feature.
        Then examining the distribution (CDF) of the feature, we find that 99% square-rooted values less than 25528.5.
        Therefore, we normalize it by 25528.5. If the value is greater than this threshold (25528.5), they are set to 1.
    """
    dataType = 0 ## 'unknown'
    
    if len(values) == 0:
        return np.zeros(0, 48)
    
    if all(map(lambda x: is_date(x), values)):
        dataType = 3 ## datetime
        values = [str(v) for v in values] ## datetime is casted to strings for feature extraction
    elif is_number(values):
        dataType = 5 ## decimal
        values = [float(v) for v in values]
    elif is_string(values):
        dataType = 1 ## string
        values = [str(v) for v in values]
    
    ptp = 1 if dataType == 1 or dataType == 3 else np.ptp(values)
    N = len(values)
    
    data_features = {}
    if dataType == 1 or dataType == 3: ## string
        data_features['aggrPercentFormatted'] = len([x for x in values if '%' in x]) / N 
        data_features['norm_range'] = 1
        data_features['norm_var'] = 1
        data_features['norm_cov'] = 1
        data_features['skewness'] = 1
        data_features['kurtosis'] = 1
        data_features['gini'] = 1
        
        value_lengths = [len(v) for v in values]
        data_features['averageLogLength'] =sum(map(lambda x: min(1, 2*(1-(1/(math.log(max(x-1,1),10) + 1)))),value_lengths))

        
    elif dataType == 5: ## numbers
        data_features['aggr01Ranged'] = sum(1 for v in values if v >= 0 and v<= 1) / N
        data_features['aggr0100Ranged'] = sum(1 for v in values if v >= 0 and v<= 100) / N
        data_features['aggrIntegers']= sum(1 for v in values if v.is_integer()) / N
        data_features['aggrNegative']= sum(1 for v in values if v < 0) / N
        data_features['norm_range'] =  min(1.0, math.sqrt( np.ptp(values)) / 25528.5) 
        data_features['partialOrdered'] = compute_incdec(values) / max(N - 1, 1)

        data_features['norm_var'] = min(1.0, math.sqrt(np.var(values)) / 38791.2)
    
        raw_cov = np.cov(values)
        data_features['norm_cov'] = min(1.0, math.sqrt(raw_cov) / 55.2) if raw_cov >= 0 else \
            max(-1.0, -1.0 * math.sqrt(abs(raw_cov)) / 633.9)
        
        data_features['benford'] = compute_benford(values) 
        
        data_features['orderedConfidence'] = 1 if compute_incdec(values) == max(N - 1, 1) else 0
        data_features['equalProgressionConfidence'] = get_equal_progression(values)
        data_features['geometircProgressionConfidence'] = get_geo_progression(values)
        
        value_sum = np.sum(values)
        data_features['sumIn01'] = value_sum if value_sum >= 0 and value_sum <= 1 else 0
        data_features['sumIn0100'] = value_sum / 100 if value_sum >=0 and value_sum <= 100 else 0
        
        skewness_99ile = 3.844
        data_features['skewness'] = stats.skew(values) / skewness_99ile

        data_features['kurtosis'] = max(1, stats.kurtosis(values))
        data_features['gini'] = max(1, get_gini(values))
        
        
    ### shared features
    values_str = [str(v) for v in values]
    data_features['commonPrefix'] = len(find_common_prefix(values_str)) / max([len(x) for x in values_str])
    
    values_reversed_str = [s[::-1] for s in values_str]    
    data_features['commonSuffix'] = len(find_common_prefix(values_reversed_str)) / max([len(x) for x in values_reversed_str])
    
    data_features['keyEntropy'] = compute_entropy(values_str)
    
    chars = [list(s) for s in values_str]
    chars_flattened = [val for sublist in chars for val in sublist]
    data_features['charEntropy'] = compute_entropy(chars_flattened)
    
    data_features['changeRate'] = sum(1 for n in range(1,N) if values_str[n] != values_str[n-1]) / max(N - 1, 1)
    
    data_features['cardinality'] = len(set(values_str)) / N
    data_features['spread'] = data_features['cardinality'] / ptp

    value,counts = np.unique(values_str, return_counts=True)
    data_features['major'] = max(counts) / N
    
    value_lengths = [len(v) for v in values_str]
    data_features['medianLength'] = min(1, np.median(value_lengths) / 27.5) # median length of fields' records, 27.5 is 99% value
    data_features['lengthStdDev'] = min(1.0, np.std(value_lengths) / 10.0)
    
    data_features['nRows'] = N / 576 # Number of rows, 576 is 99% value
    
    data_features['absoluteCardinality'] = len(set(values_str)) / 344 #334 is 99% value
    
    features = [
        data_features.get('aggrPercentFormatted', 0),  # Proportion of cells having percent format
        data_features.get('aggr01Ranged', 0),  # Proportion of values ranged in 0-1
        data_features.get('aggr0100Ranged', 0),  # Proportion of values ranged in 0-100
        data_features.get('aggrIntegers', 0),  # Proportion of integer values
        data_features.get('aggrNegative', 0),  # Proportion of negative values
        data_features.get('aggrBayesLikeSum', 0),  # Aggregated Bayes feature
        data_features.get('dmBayesLikeDimension', 0),  # Bayes feature for dimension measure
        data_features['commonPrefix'],  # Proportion of most common prefix digit
        data_features['commonSuffix'],  # Proportion of most common suffix digit
        data_features['keyEntropy'],  # Entropy by values
        data_features['charEntropy'],  # Entropy by digits/chars
        data_features['norm_range'],  # data_features.get('range', 0),  # Values range
        data_features['changeRate'],  # Proportion of different adjacent values
        data_features.get('partialOrdered', 0),  # Maximum proportion of increasing or decreasing adjacent values
        data_features['norm_var'],  # data_features.get('variance', 0),  # Standard deviation
        data_features['norm_cov'],  # data_features.get('cov', 0),  # Coefficient of variation
        data_features['cardinality'],  # Proportion of distinct values
        data_features.get('spread', 0),  # Cardinality divided by range
        data_features['major'],  # Proportion of the most frequent value
        data_features.get('benford', 0),  # Distance of the first digit distribution to real-life average
        data_features.get('orderedConfidence', 0),  # Indicator of sequentiality
        data_features.get('equalProgressionConfidence', 0),  # confidence for a sequence to be equal progression
        data_features.get('geometircProgressionConfidence', 0),  # confidence for a sequence to be geometric progression
        data_features['medianLength'],  # median length of fields' records, 27.5 is 99% value
        data_features.get('lengthStdDev',0),  # transformed length stdDev of a sequence
        data_features.get('sumIn01', 0.0),  # Sum the values when they are ranged 0-1
        data_features.get('sumIn0100', 0.0),  # Sum the values when they are ranged 0-100
        data_features['absoluteCardinality'],  # Absolute Cardinality, 344 is 99% value
        data_features.get('skewness', 0),
        data_features.get('kurtosis', 0),
        data_features.get('gini', 0),
        data_features.get('nRows', 0.0),  # Number of rows, 576 is 99% value
        data_features.get('averageLogLength', 0.0)
    ]
        
    return dataType, features

def get_word_embeddings(text: str, word_embedding_dict: dict):
    """
        Get the embeddings for a phrase by averaging the embeddings of each word.
        
        :param text: (str). Input text
        :param word_embedding_dict: (dict). {text: embedding}. Read from pretrained model.
    """
    words = [w.lower() for w in re.split('[; _ , : -]', text)]
    embeddings = [word_embedding_dict.get(w, np.zeros(word_embedding_dict['apple'].shape)).astype(np.float) for w in words]
    
    return np.mean(embeddings, axis = 0)