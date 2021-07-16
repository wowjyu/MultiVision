import numpy as np
import math
from sklearn.preprocessing import minmax_scale
import itertools
import re

def cleanup_data_features_nn(data_features: dict):
    """
    Clean up data features that used in neural network models.
    Features like 'range', 'variance', 'cov', 'lengthStdDev' are in range [-inf, inf] or [0, inf].
    These features may cause problems in NN model and need to be normalized.
    We adopt normalization by distribution here.
    To take range as an example, this feature distributes in [0, inf]. We first square root this feature.
    Then examining the distribution (CDF) of the feature, we find that 99% square-rooted values less than 25528.5.
    Therefore, we normalize it by 25528.5. If the value is greater than this threshold (25528.5), they are set to 1.
    """
    # Normalize range, var and cov
    raw_range = data_features.get('range', 0.0)
    norm_range = 1 if isinstance(raw_range, str) else min(1.0, math.sqrt(raw_range) / 25528.5)
    raw_var = data_features.get('variance', 0.0)
    norm_var = 1 if isinstance(raw_var, str) else min(1.0, math.sqrt(raw_var) / 38791.2)
    raw_cov = data_features.get('cov', 0.0)
    if isinstance(raw_cov, str):
        norm_cov = 1
    else:
        norm_cov = min(1.0, math.sqrt(raw_cov) / 55.2) if raw_cov >= 0 else \
            max(-1.0, -1.0 * math.sqrt(abs(raw_cov)) / 633.9)
    # Use standard deviation rather than variance of feature 'lengthVariance'
    # 99% length stdDev of fields' records is less than 10
    lengthStdDev = min(1.0, math.sqrt(data_features.get('lengthVariance', 0.0)) / 10.0)

    # There are NAN or extremely large values in skewness and kurtosis, so we set:
    # skewness: NAN -> 0.0, INF/large values -> 1.0
    # kurtosis: NAN -> 0.0, INF/large values -> 1.0
    # skewness 99%ile = 3.844
    # kurtosis 99%ile = 0.7917 (no normalization)
    skewness_99ile = 3.844
    skewness = data_features.get('skewness', 0.0)
    if skewness == "NAN":
        skewness = 0.0
    elif isinstance(skewness, str) or abs(skewness) > skewness_99ile:
        skewness = skewness_99ile
    skewness = skewness / skewness_99ile

    kurtosis = data_features.get('kurtosis', 0.0)
    if kurtosis == "NAN":
        kurtosis = 0.0
    elif isinstance(kurtosis, str) or abs(kurtosis) > 1.0:
        kurtosis = 1.0

    gini = data_features.get('gini', 0.0)
    if gini == "NAN":
        gini = 0.0
    elif isinstance(gini, str) or abs(gini) > 1.0:
        gini = 1.0

    benford = data_features.get('benford', 0.0)
    if benford == "NAN":
        benford = 0.0
    elif isinstance(benford, str) or abs(benford) > 1.036061:
        benford = 1.036061

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
        norm_range,  # data_features.get('range', 0),  # Values range
        data_features['changeRate'],  # Proportion of different adjacent values
        data_features.get('partialOrdered', 0),  # Maximum proportion of increasing or decreasing adjacent values
        norm_var,  # data_features.get('variance', 0),  # Standard deviation
        norm_cov,  # data_features.get('cov', 0),  # Coefficient of variation
        data_features['cardinality'],  # Proportion of distinct values
        data_features.get('spread', 0),  # Cardinality divided by range
        data_features['major'],  # Proportion of the most frequent value
        benford,  # Distance of the first digit distribution to real-life average
        data_features.get('orderedConfidence', 0),  # Indicator of sequentiality
        data_features.get('equalProgressionConfidence', 0),  # confidence for a sequence to be equal progression
        data_features.get('geometircProgressionConfidence', 0),  # confidence for a sequence to be geometric progression
        min(1, data_features.get('medianLength', 0) / 27.5),  # median length of fields' records, 27.5 is 99% value
        lengthStdDev,  # transformed length stdDev of a sequence
        data_features.get('sumIn01', 0.0),  # Sum the values when they are ranged 0-1
        data_features.get('sumIn0100', 0.0) / 100,  # Sum the values when they are ranged 0-100
        min(1, data_features.get('absoluteCardinality', 0.0) / 344),  # Absolute Cardinality, 344 is 99% value
        skewness,
        kurtosis,
        gini,
        data_features.get('nRows', 0.0) / 576,  # Number of rows, 576 is 99% value
        data_features.get('averageLogLength', 0.0)
    ]
    for i, f in enumerate(features):
        if isinstance(f, str) or abs(f) > 10000:
            print("WARNING: feature[{}] is {}".format(i, f))
    return [0 if isinstance(f, str) else f for f in features]

def get_data_feature_by_column(columnIdx, dfJson):
    if dfJson == None:
        return np.array([])
    obj = next(x for x in dfJson['fields'] if x['index'] == columnIdx)
    features = [obj['index'], obj['chart_type'], *cleanup_data_features_nn(obj['dataFeatures']), *obj['cellTypeCounter'].values(), obj['inHeaderRegion'], 
               obj['isPercent'], obj['isCurrency'], obj['hasYear'], obj['hasMonth'], obj['hasDay']]
    features = [float(f) for f in features]
    return np.array(features)

# def get_embed_feature_by_column(columnIdx, emJson):
#     if emJson == None:
#         return np.array([])
#     return np.array(emJson[columnIdx]['0']['mean'])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


#######################################################################
#######################################################################
####       Provenance related                  ########################
#######################################################################

def get_all_charts_scores(charts):
    def zipChartsWithType(chart, score_normed):    
        results = []

        max_v = max([c['v'] for c in chart['chart_type']])
        if len(chart['indices']) == 1:
            [x for x in chart['chart_type'] if x["chart_type"] == 'bar' ][0]['v'] = max_v + 1
        elif len(chart['indices']) > 2:
            [x for x in chart['chart_type'] if x["chart_type"] == 'scatter' ][0]['v'] = max_v + 1

        v_normed = minmax_scale([t['v'] for t in chart['chart_type']])
        for idx, val in enumerate(chart['chart_type']):
            obj = {
                'indices': chart['indices'],
                'score': chart['score'],
                'score_normed': score_normed,
                'chart_type': val['chart_type'],
                'v': val['v'],
                'v_normed': v_normed[idx],
                's': score_normed * v_normed[idx]
            }
            results.append(obj)

        return results

    all_charts_scores_normed = minmax_scale([c['score'] for c in charts])
    all_charts_with_normed_score = []
    for idx, c in enumerate(charts):
        all_charts_with_normed_score.extend(zipChartsWithType(c, all_charts_scores_normed[idx]))
    return all_charts_with_normed_score

### chart_lists: {indices, chart_type} ###
def get_chart_lists(raw_lists):
    def get_chart(raw_chart):
        if raw_chart['markEncoding'] == 'arc':
            chart_type = 'pie'
        elif raw_chart['markEncoding'] == 'point':
            chart_type = 'scatter'
        else: 
            chart_type = raw_chart['markEncoding']
        
        return {"chart_type": chart_type, "indices": raw_chart['indices']}
    return [get_chart(c) for c in raw_lists]


def mvRecord_to_features(mvRecord, all_charts_with_normed_score):
    mv_charts = get_chart_lists(mvRecord['charts'])
    return charts_to_features(mv_charts, all_charts_with_normed_score)

def get_n_columns(chart_lists):
    return len(set(list(itertools.chain(*[x['indices'] for x in chart_lists]))))

def get_n_chartType(chart_lists):
    return len(set([x['chart_type'] for x in chart_lists]))

def get_n_complementary(chart_lists):
    combinations = list(itertools.combinations([x['indices'] for x in chart_lists], 2))
    
    def if_complementary(pair):
        return len(set(itertools.chain(*pair))) == sum([len(x) for x in pair])
    
    return sum(1 for x in combinations if if_complementary(x))
    
def get_n_decomposition(chart_lists):
    def if_decompose(c1, c2):
        return all(i in c1 for i in c2) or all(i in c2 for i in c1) 
    combinations = list(itertools.combinations([x['indices'] for x in chart_lists], 2))
    return sum(1 for c1, c2 in combinations if if_decompose(c1, c2))

def get_mean_scores(chart_lists, all_charts_with_normed_score):
    def get_score(chart):
        return [x for x in all_charts_with_normed_score if x['indices'] == chart['indices'] and x['chart_type'] == chart['chart_type']][0]['s']
    scores = [get_score(chart) for chart in chart_lists]
    return np.mean(scores)

def charts_to_features(chart_lists, all_charts_with_normed_score):
    features = [get_n_columns(chart_lists), get_n_chartType(chart_lists), get_n_complementary(chart_lists), 
            get_n_decomposition(chart_lists), get_mean_scores(chart_lists, all_charts_with_normed_score), len(chart_lists)]
    return features
