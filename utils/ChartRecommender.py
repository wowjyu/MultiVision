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

    def __init__(self, df, word_embedding_dict: dict, column_score_model, chart_type_model) -> None:
        """Init the recommender

        Args:
            df (pandas.DataFrame): the input data table
        """
        self.df = df 
        column_features = self.compute_columns_feature(self.df, word_embedding_dict)

        self.feature_dict = column_features['features']
        self.fields = column_features['fields']

        ## enumerate possible charts 
        self.charts_indices = self.enumerate_chart(self.df)

        ## assess the quality of each chart
        self.charts = self.recommend_chart(self.charts_indices, self.feature_dict, column_score_model, chart_type_model)
        self.charts = self.apply_hard_rule(self.charts)

    def recommend_chart(self, charts_indices, feature_dict, column_score_model, chart_type_model):
        """Evaluates the model and recommends single chart

        Args:
            charts_indices (List): a list of chart (describled as column indices). E.g., [[1,2],...] (the first chart encodes column 1 and column 2).
            feature_dict: [description]
            column_score_model: trained model 
            chart_type_model: trained model

        Returns:
            charts: a list of charts
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
                chart_obj = {'indices': chart_indices, 
                    'fields': [self.fields[idx] for idx in chart_indices], 
                    'column_selection_score': chart_scores[cidx], 
                    'chart_type': t['type'], 
                    'chart_type_prob': t['p'], 
                    'final_score': chart_scores[cidx] * t['p'], 
                    'n_column': len(chart_indices)}
                charts.append(chart_obj)
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

    def compute_columns_feature(self, df, word_embedding_dict):
        """Convert the data table into features of each column.

        Args:
            df (pandas.DataFrame): Input data table 
            word_embedding_dict (dict): The pre-trained word embedding model

        Returns:
            {
                'features': (dict) {column_index: column_feature},
                'fields': ('index': {"name","index","type"}). The head-name, column-index, data type of each data column.
            }
        """
        def field_type(idx):
            """ Convert the idx to field type. See get_data_features in featureExtractor.py """ 
            if (idx == 1):
                return "nominal";
            elif (idx ==3 or idx == 7):
                return "temporal";
            elif (idx == 5):
                return "quantitative";
            return "";

        feature_dict = {}
        fields = {}
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

            fields[cIdx] = {
                "name": column_header,
                "index": cIdx,
                "type": field_type(dataType)
            }

        return {"features": feature_dict, "fields": fields}
        

    def recommend_next(self, mv_model, current_mv = [], min_column_selection_score = 0.2):
        """Recommend the next one chart given the current MV

        Args:
            mv_model (): the trained mv model
            current_mv (list, optional): The current mv. Defaults to [].
            min_column_selection_score (float, optional): Minimal column_selection_score for fast computation. Defaults to 0.2.

        Returns:
            candidate_charts, scores: array-like. The candidate charts and their scores
        """        
        def isExist(chart, charts):
            """Check if chart is in charts by the encoded data columns."""
            return any(chart['indices'] == c['indices'] for c in charts)

        def get_candidate_charts(current_mv):
            charts_pool = [x for x in self.charts if x['column_selection_score'] > min_column_selection_score] ## filter bad visual encoding for fast computation
            return [x for x in charts_pool if not isExist(x, current_mv)]
        
        candidate_charts = get_candidate_charts(current_mv)
        
        scores = []
        for candidate_chart in candidate_charts:
            ## add the candidate_chart to current_mv => new MV
            new_mv = [*current_mv, candidate_chart]
            
            ## use the trained model
            chart_feature_dl = mvFeatureExtractor.charts_to_features_dl(new_mv, self.charts, seq_length = 12) ## the max number of charts in an MV is 12
            mv_model_inut = Variable(torch.Tensor([chart_feature_dl])).cuda().float() 
            scores.append(mv_model(mv_model_inut).tolist()[0])

        return candidate_charts, scores


    def recommend_mv(self, mv_model, current_mv = [], max_charts = 5, min_column_selection_score = 0.2):
        """[summary]

        Args:
            mv_model (): the trained mv model
            current_mv (list, optional): The current mv. Defaults to [].
            max_charts (int, optional): The max number of charts. Defaults to 5.
            min_column_selection_score (float, optional): Minimal column_selection_score for fast computation. Defaults to 0.2.
        """
        while len(current_mv) < max_charts:
            ## recommend the next chart 
            candidate_charts, scores = self.recommend_next(mv_model, current_mv, min_column_selection_score)

            ## get the highest score and select it (greedy algorithm)
            max_score_index = np.argmax(scores)
            max_score_chart = candidate_charts[max_score_index]
            
            ## add the chart into current_mv
            current_mv.append(max_score_chart)      

        return current_mv

    def apply_hard_rule(self, charts):
        """Filter the charts by hard rules/constraints

        Args:
            charts (List[dict]): a list of chart dicts
        """
        def count_field_types(chart, field_types):
            filtered = [x for x in chart['fields'] if x['type'] in field_types]
            return len(filtered)

        def is_valid(chart):
            if chart['chart_type'] == 'line' and count_field_types(chart, ['nominal']) >= 2:
                ## prevent a line chart from encoding two nominal fields
                return False
            elif chart['chart_type'] == 'line' and count_field_types(chart, ['temporal']) == 0:
                ## prevent a line chart without a temporal field 
                ## note: this rule can be softed by introducing aggregation such as binning
                return False
            elif chart['chart_type'] == 'pie' and chart['n_column'] > 1:
                ## prevent a pie chart encoding >1 column
                return False
            return True

        filtered_charts = [c for c in charts if is_valid(c)]
        return filtered_charts

