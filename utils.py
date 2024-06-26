import numpy as np
import logging
from datetime import datetime
import os 
# from tensorboardX import SummaryWriter
import shutil

def read_and_parse_file(file_path):
    data_tbl = np.loadtxt(file_path, dtype=str)
    data, targets = data_tbl[:, 0], data_tbl[:, 1:].astype(np.int8)
    return data, targets


def set_logger(config):
    os.makedirs("./logs/", exist_ok=True)
    if config.notes:
        prefix = config.notes
    else:
        prefix = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_file = os.path.join('./logs/', prefix + '.log')

    log_directory = os.path.dirname(os.path.join('./logs/', prefix))
    os.makedirs(log_directory, exist_ok=True)
    # config.__dict__['checkpoint_root'] = os.path.join('./checkpoints/', prefix)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s.',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    if config.use_writer:
        writer_root = os.path.join('./logs/', prefix + '.writer')
        if os.path.exists(writer_root):
            shutil.rmtree(writer_root)
        writer = SummaryWriter(writer_root)        
    else:
        writer = None 
    return writer


class WarmUpAndCosineDecayScheduler:
    def __init__(self, optimizer, start_lr, base_lr, final_lr,
                 epoch_num, batch_num_per_epoch, warmup_epoch_num):
        self.optimizer = optimizer
        self.step_counter = 0
        warmup_step_num = batch_num_per_epoch * warmup_epoch_num
        decay_step_num = batch_num_per_epoch * (epoch_num - warmup_epoch_num)
        warmup_lr_schedule = np.linspace(start_lr, base_lr, warmup_step_num)
        cosine_lr_schedule = final_lr + 0.5 * \
            (base_lr - final_lr) * (1 + np.cos(np.pi *
                                               np.arange(decay_step_num) / decay_step_num))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # step at each mini-batch
    def step(self):
        curr_lr = self.lr_schedule[self.step_counter]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = curr_lr
        self.step_counter += 1
        return curr_lr
    

class Monitor:
    def __init__(self, max_patience=5, delta=1e-6):
        self.counter_ = 0
        self.best_value = 0
        self.max_patience = max_patience
        self.patience = max_patience
        self.delta = delta

    def update(self, cur_value):
        self.counter_ += 1
        is_break = False
        is_lose_patience = False
        if cur_value < self.best_value + self.delta:
            cur_value = 0
            self.patience -= 1
            logging.info("the monitor loses its patience to %d!" %
                         self.patience)
            is_lose_patience = True
            if self.patience == 0:
                self.patience = self.max_patience
                is_break = True
        else:
            self.patience = self.max_patience
            self.best_value = cur_value
            cur_value = 0
        return (is_break, is_lose_patience)

    @property
    def counter(self):
        return self.counter_