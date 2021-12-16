class VegaLiteRender():
    '''
        Render a chart obj to Vega-Lite spec
    '''
    def __init__(self, chart_type, columns, data):
        self.chart_type = chart_type
        self.columns = columns
        
        ## assign mark
        mark = self.chart_type_to_mark(chart_type)
        vSpec = {
            'mark': mark
        }    
        
        ## assign data
        self.data = data
        vSpec['data'] = {
            'values': data
        }

        ## available channels
        ## channels are orderred according to the effectiveness rankings in Voyager
        if mark == 'pie':
            self.available_channels = {"theta", "color", "column", "row"}
            self.nominal_channels = ["color", "theta", "column", "row"]
            self.quantitative_channels = ["theta", "color", "column", "row"]
            self.temporal_channels = ["theta", "color", "column", "row"]
        else:
            self.available_channels = {"x", "y", "color", "size", "column", "row"}
            self.nominal_channels = ['x', 'y', "color", "column", "row"]
            self.quantitative_channels = ['x', 'y', "size", 'color']
            self.temporal_channels = ['x', 'y', "size", 'color']
            
        print(columns)

        if len(columns) == 1:
            vlEncoding = self.vis_single_column(chart_type, columns[0])
        else:
            vlEncoding = self.vis_multi_columns(chart_type, columns)

        vSpec['encoding'] = vlEncoding
            
        self.vSpec = vSpec

    def chart_type_to_mark(self, chart_type):
        """ Convert the chart type to mark type in Vega-Lite """
        if (chart_type == 'pie'):
            return 'arc'
        elif chart_type == 'scatter':
            return 'point'
        else:
            return chart_type ## 'area', 'bar', 'line'
        
    def vis_single_column(self, chart_type, field):
        print(field)

        if (chart_type == 'pie'):
            vlEncoding = {
                'color': {
                        'field': field['name'],
                        'bin': True,
                        'type': field['type']
                    },
                    'theta': {
                        'aggregate': 'count',
                        'type': 'quantitative'
                }
            }
        else:
            vlEncoding = {
                    'x': {
                        'field': field['name'],
                        'bin': True,
                        'type': field['type']
                    },
                    'y': {
                        'aggregate': 'count',
                        'type': 'quantitative'
                }
            }
        return vlEncoding
    
    def vis_multi_columns(self, chart_type, columns):
        ## assign encoding
        nominals = [c for c in columns if c['type'] == 'nominal']
        temporals = [c for c in columns if c['type'] == 'temporal']
        quantitatives = [c for c in columns if c['type'] == 'quantitative']

        def tryAssign(vlEncoding, fields, channels, available_channels):
            for f in fields:
                cs = [c for c in channels if c in available_channels]

                ## no available channels, skip this field
                if len(cs) == 0:
                    continue

                ## assign the field & channel
                c = cs[0]
                vlEncoding[c] = {
                    'field': f['name'],
                    'type': f['type']
                }

                available_channels.remove(c)

            return vlEncoding, available_channels

        vlEncoding = {}
        if len(quantitatives) == 0 and chart_type == 'bar':
            vlEncoding['y'] = {
                'aggregate': 'count',
                'type': 'quantitative'
            }
            self.available_channels.remove('y')
        vlEncoding, self.available_channels = tryAssign(vlEncoding, nominals, self.nominal_channels, self.available_channels)
        vlEncoding, self.available_channels = tryAssign(vlEncoding, quantitatives, self.quantitative_channels, self.available_channels)    
        vlEncoding, self.available_channels = tryAssign(vlEncoding, temporals, self.temporal_channels, self.available_channels)

        return vlEncoding