import logging

from . import utils
from . import reader


class Generator():
    def __init__(self, args):
        utils.set_logging(args)
        self.dataset = args.dataset
        self.generator = reader.create(args)

    def start(self):
        logging.info('')
        logging.info('Starting generating ...')
        logging.info('Dataset: {}'.format(self.dataset))
        self.generator.start()
        logging.info('Finish generating!')
