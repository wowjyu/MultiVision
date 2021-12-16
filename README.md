# MultiVision
This repo contains the source code for the work **MultiVision: Designing Analytical Dashboards with Deep Learning Based Recommendation**, accepted at IEEE VIS 2021.

Given a data table, MultiVision recommends a chart and/or a dashboard containing multiple charts for conducting data analysis..

[![Screenshot-2021-07-16-at-16-42-09.png](https://i.postimg.cc/g24nbgxb/Screenshot-2021-07-16-at-16-42-09.png)](https://postimg.cc/6829dLVM)

## Content   
This repo is under construction.
- [x] The trained model and demo
- [ ] Tutorial for running the scoring model
- [x] The visual encoding recommender 
- [ ] The interface
- [ ] The training script

For the training dataset, please refer to udpates from [Table2Charts](https://github.com/microsoft/Table2Charts).     

## How to run the model & benchmark?  
[Demo.ipynb](https://github.com/Franches/MultiVision/blob/master/Demo.ipynb) demonstrates how to run the trained model.    
- Input: a data table in CSV format
- Output: an MV, describled as a list of charts (as shown below, where the `indices' is the indices of data columns encoded by this chart.

```
## The first chart is a line chart encoding data columns 2, 3, 5. The second chart is a bar chart encoding data columns 0, 1, 6.
[{'indices': (2, 3, 5),
  'column_selection_score': 0.17087826005069967,
  'chart_type': 'line',
  'chart_type_prob': 0.9999961295181747,
  'final_score': 0.1708775986694998},
 {'indices': (0, 1, 6),
  'column_selection_score': 0.953993421295783,
  'chart_type': 'bar',
  'chart_type_prob': 0.9747984895572608,
  'final_score': 0.9299513461266928}]
```

VegaLiteRender.py provides a toolkit for rendering the above results into a Vega-Lite chart.
![Screenshot 2021-12-16 at 13 44 57](https://user-images.githubusercontent.com/14938532/146315135-95e2bdb1-d9f4-4a35-9f60-830c8e0433c8.png)


