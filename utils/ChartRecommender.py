import pandas as pd
import numpy as np

import itertools
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from utils import featureExtractor, mvFeatureExtractor

from utils.helper import softmax
from sklearn.preprocessing import minmax_scale

class ChartRecommender:

    def __init__(self, csv_file: str, word_embedding_dict: dict, column_score_model, chart_type_model) -> None:
        """Init the recommender

        Args:
            csv_file (str): The path of input csv file
        """
        self.df = pd.read_csv(csv_file)
        self.feature_dict = self.compute_columns_feature(self.df, word_embedding_dict)

        ## enumerate possible charts 
        self.charts_indices = self.enumerate_chart(self.df)

        ## assess the quality of each chart
        self.charts = self.recommend_chart(self.charts_indices, self.feature_dict, column_score_model, chart_type_model)

    def recommend_chart(self, charts_indices, feature_dict, column_score_model, chart_type_model):
        """Evaluates the model and recommends single chart

        Args:
            charts_indices (List): a list of chart (describled as column indices). E.g., [[1,2],...] (the first chart encodes column 1 and column 2).
            feature_dict: [description]
            column_score_model: trained model 
            chart_type_model: trained model

        Returns:
            [type]: [description]
        """
        seq_length = 4 ## max number of columns per chart is 4

        ## compute the features of each chart by concatenating features of each column
        def columns_to_features(columns):
            feature = [feature_dict[c] for c in columns]
            padding_zero = np.zeros((seq_length - len(feature), len(feature[0]))).tolist()
            feature.extend(padding_zero)
            return feature
        chart_features = [columns_to_features(c) for c in charts_indices]

        ## eva the assessment/scoring model
        model_input = Variable(torch.Tensor(chart_features)).cuda().float() 
        max_seq_len = 4
        seq_len = np.full(model_input.shape[0], max_seq_len)
        model_input = pack_padded_sequence(model_input, seq_len, batch_first=True, enforce_sorted = False)
        chart_scores = torch.flatten(column_score_model(model_input)).tolist()
        chart_scores = minmax_scale(chart_scores) ## min-max norm

        ## eva the chart type prediction model
        model_input = Variable(torch.Tensor(chart_features)).cuda().float() 
        types = chart_type_model(model_input).tolist()
        classes = ['area', 'bar', 'bubble', 'line', 'pie', 'radar', 'scatter', 'stock', 'surface']
        def types_to_meaning(prob):
            """ prob - returned from model (softmax) """
            p = softmax(prob)
            ## only support the first five types (>99% training dataste)
            results = []
            for i, v in enumerate(prob):
                if classes[i] in ['area', 'bar', 'scatter', 'line', 'pie']:
                    results.append({'type': classes[i], 'v': v, 'p': p[i]})
            return results
        types = [types_to_meaning(t) for t in types]

        ## wrap the results
        charts = []
        for cidx, chart_indices in enumerate(charts_indices):
            for t in types[cidx]:
                charts.append({'indices': chart_indices, 'column_selection_score': chart_scores[cidx], 'chart_type': t['type'], 'chart_type_prob': t['p'], 'final_score': chart_scores[cidx] * t['p']})
        return charts

    def enumerate_chart(self, df):
        """Enumerate possible charts 

        Args:
            df (pandas.DataFrame): Input data table 
            feature_dict (dict):  {column_index: column_feature}

        Returns:
            List: a list of chart (describled as column indices). E.g., [[1,2],...] (the first chart encodes column 1 and column 2).
        """
        ## enumerate possible combinations of data columns.
        ## max number of columns is 4
        idxes = list(range(0, len(df.columns)))
        charts_indices = []
        for num_selected_columns in range(1,4):
            charts_indices.extend([p for p in list(itertools.combinations(idxes, num_selected_columns))])

        return charts_indices

    def compute_columns_feature(self, df, word_embedding_dict) -> dict:
        """Convert the data table into features of each column.

        Args:
            df (pandas.DataFrame): Input data table 
            word_embedding_dict (dict): The pre-trained word embedding model

        Returns:
            dict: {column_index: column_feature}
        """
        feature_dict = {}
        for cIdx, column_header in enumerate(df.columns):
            column_values = df[column_header].tolist()
            dataType, data_features = featureExtractor.get_data_features(column_values)
            embedding_features = featureExtractor.get_word_embeddings(column_header, word_embedding_dict)

            ## *np.zeros(11).tolist() contains 11 features that we do not reproduce yet
            column_idx_normed = min(1.0, cIdx / 50) # 99% tables have less than 50 columns
            dataType_normed = dataType / 5 # max is 5

            feature = [column_idx_normed, dataType_normed, *data_features, *np.zeros(11).tolist(), *embedding_features]

            feature = np.nan_to_num(np.array(feature), 0)
            feature_dict[cIdx] = feature.tolist()
        return feature_dict

    def recommend_mv(self, mv_model, current_mv = [], max_charts = 5, min_chart_type_prob = 0.2):
        """[summary]

        Args:
            mv_model (): the trained mv model
            current_mv (list, optional): The current mv. Defaults to [].
            max_charts (int, optional): The max number of charts. Defaults to 5.
            min_chart_type_prob (float, optional): Minimal chart_type_prob for fast computation. Defaults to 0.2.
        """
        def isExist(chart, charts):
            """Check if chart is in charts by the encoded data columns."""
            return any(chart['indices'] == c['indices'] for c in charts)

        def get_candidate_charts(current_mv):
            charts_pool = [x for x in self.charts if x['chart_type_prob'] > min_chart_type_prob] ## filter bad visual encoding for fast computation
            return [x for x in charts_pool if not isExist(x, current_mv)]

        while len(current_mv) < max_charts:
            candidate_charts = get_candidate_charts(current_mv)
            
            scores = []
            for candidate_chart in candidate_charts:
                ## add the candidate_chart to current_mv => new MV
                new_mv = [*current_mv, candidate_chart]
                
                ## use the trained model
                chart_feature_dl = mvFeatureExtractor.charts_to_features_dl(new_mv, self.charts, seq_length = 12) ## the max number of charts in an MV is 12
                mv_model_inut = Variable(torch.Tensor([chart_feature_dl])).cuda().float() 
                scores.append(mv_model(mv_model_inut).tolist()[0])

            ## get the highest score and select it (greedy algorithm)
            max_score_index = np.argmax(scores)
            max_score_chart = candidate_charts[max_score_index]
            
            ## add the chart into current_mv
            current_mv.append(max_score_chart)      

        return current_mv
