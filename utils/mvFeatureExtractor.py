import pandas as pd
import os
import glob as glob
import json
import matplotlib.pyplot as plt
import multiprocessing
import re
import itertools
import math
import numpy as np
from functools import partial
import random
from sklearn.preprocessing import minmax_scale

from utils.helper import charts_to_features

def chart_type_feature(chart):
    if (chart['chart_type'] == 'bar'):
        return [0,0,0,0,1]
    elif (chart['chart_type'] == 'pie'):
        return [0,0,0,1,0]
    elif (chart['chart_type'] == 'line'):
        return [0,0,1,0,0]
    elif (chart['chart_type'] == 'scatter'):
        return [0,1,0,0,0]
    elif (chart['chart_type'] == 'area'):
        return [1,0,0,0,0]

def compose_feature(chart, mv_charts):
    def if_decompose(c1, c2):
        return all(i in c2 for i in c1)  ## c1 in c2
    mv_charts = [c for c in  mv_charts if chart['indices'] != c['indices'] and chart['chart_type'] != c['chart_type']]
    return sum(1 for c in mv_charts if if_decompose(chart, c))

def complementary_feature(chart, mv_charts):
    def if_complementary(pair):
        return len(set(itertools.chain(*pair))) == sum([len(x) for x in pair])
    mv_charts = [c for c in mv_charts if chart['indices'] != c['indices'] and chart['chart_type'] != c['chart_type']]

    combinations = [(chart, c) for c in mv_charts]
    return sum(1 for x in combinations if if_complementary(x))

def chart_to_feature(chart, mv_charts, all_charts_with_normed_score):
    chart_type_n_columns_feature = chart_type_feature(chart)
    score = [x for x in all_charts_with_normed_score if x['indices'] == chart['indices'] and x['chart_type'] == chart['chart_type']][0]['final_score']
    
    # # print(len(chart['indices']), chart_type_feature(chart), compose_feature(chart,mv_charts), 
    # #         complementary_feature(chart,mv_charts), score)
    # print(chart)
    return [len(chart['indices']), *chart_type_feature(chart), compose_feature(chart,mv_charts), 
            complementary_feature(chart,mv_charts), score]

def charts_to_features_dl(mv_charts, all_charts_with_normed_score, seq_length = False):
    features = []
    for c in mv_charts:
        features.append(chart_to_feature(c, mv_charts, all_charts_with_normed_score))

    if seq_length != False:
        ## padding zero to match seq_length
        padding_zero = np.zeros((seq_length - len(features), len(features[0]))).tolist()
        features.extend(padding_zero)

    return features

def get_chart_lists(raw_lists):
    def get_chart(raw_chart):
        if raw_chart['markEncoding'] == 'arc':
            chart_type = 'pie'
        elif raw_chart['markEncoding'] == 'point':
            chart_type = 'scatter'
        else: 
            chart_type = raw_chart['markEncoding']
        
        return {'chart_type': chart_type, 'indices': raw_chart['indices']}
    return [get_chart(c) for c in raw_lists]


def mvRecord_to_features(mvRecord, all_charts_with_normed_score):
    charts = [c for c in mvRecord['charts'] if len(c['indices']) and not None in c['indices']]
    mv_charts = get_chart_lists(charts)
    return charts_to_features_dl(mv_charts, all_charts_with_normed_score)


def get_all_charts_scores(charts):
    def zipChartsWithType(chart, score_normed):    
        results = []
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