import os
import shutil


class SummaryWriter:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.log_name = 'log.csv'

        # remove
        shutil.rmtree(self.root_dir, ignore_errors=True)

    def add_scalars(self, name, data_dict, index):
        for k, v in data_dict.items():
            log_dir = os.path.join(self.root_dir, name, k)
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, self.log_name)
            with open(log_path, 'at') as wf:
                wf.write('{},{:.4f}\n'.format(index, v))
