import logging

from .graphs import Graph
from .volleyball_feeder import Volleyball_Feeder
from .nba_feeder import NBA_Feeder
from .volleyball_individual_feeder import Volleyball_Individual_Feeder

__data_args = {
    'volleyball': {'class': 8, 'feeder': Volleyball_Feeder},
    'volleyball_individual': {'class': 10, 'feeder': Volleyball_Individual_Feeder},
    'nba': {'class': 9, 'feeder': NBA_Feeder},
}


def create(dataset, **kwargs):
    try:
        data_args = __data_args[dataset]
        num_class = data_args['class']
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    
    graph = Graph(dataset, **kwargs)
    del kwargs['graph']
    feeders = {
        'train': data_args['feeder'](phase='train', graph=graph, **kwargs),
        'eval': data_args['feeder'](phase='eval', graph=graph, **kwargs),
    }
    data_shape = feeders['train'].datashape

    return feeders, data_shape, num_class, graph.A, graph.parts
