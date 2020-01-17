import json
import platform

IS_MAC = True if platform.system() == 'Darwin' else False  # MacOS -> IS_DEBUG=True


class Config(dict):
    """ load json file and create a dict instance. """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


if __name__ == '__main__':
    c = Config.load('bert/config.json')
    print(type(c))
    print(c)
    print(c.dropout)
    print(c['dropout'])
