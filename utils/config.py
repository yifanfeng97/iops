import sys
import os
import ConfigParser

class config():
    def __init__(self, cfg_file='config/config.cfg'):
        cfg = ConfigParser.SafeConfigParser()
        cfg.read(cfg_file)

        self.train_file = cfg.get('DEFAULT', 'train_file')
        self.val_file = cfg.get('DEFAULT', 'val_file')