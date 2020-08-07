import json
import os
import platform


def is_mac_or_pycharm():
    return is_my_mac() or is_pycharm()


def is_my_mac() -> bool:
    return platform.system() == 'Darwin'


def is_pycharm() -> bool:
    return 'PYTHONUNBUFFERED' in os.environ


def pretty_args(args, argv):
    s = f'PYTHONPATH=.. python {argv[0]}\n'
    for i, arg in enumerate(vars(args)):
        concat_mark = '\\' if i < len(vars(args)) - 1 else ''
        if isinstance(getattr(args, arg), str):
            s += f'''  --{arg} "{getattr(args, arg)}" {concat_mark}\n'''
        else:
            s += f'''  --{arg} {getattr(args, arg)} {concat_mark}\n'''
    return s


def get_model_filename(args):
    return f'{os.path.basename(os.getcwd())}.{os.path.basename(args.data_dir)}.{os.path.basename(args.config).replace(".json", "")}.batch.{args.batch}'


class Config(dict):
    """ load json file and create a dict instance. """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

    def __repr__(self):
        return json.dumps({key: getattr(self, key) for key in self.keys()}, indent=2)


if __name__ == '__main__':
    c = Config.load('bert/config.json')
    print(type(c))
    print(c)
    print(c.dropout)
    print(c['dropout'])
