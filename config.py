import json


class Config(object):
    config_file = 'config.json'
    __instance = None

    @staticmethod
    def get_instance():
        if Config.__instance is None:
            Config()
        return Config.__instance

    def __init__(self):
        global __instance
        if Config.__instance is not None:
            raise Exception("Config initialized, invoke getInstance instance")
        else:
            Config.__instance = self
            with open(self.config_file) as f:
                self.config_data = json.load(f)
                print('Configuration loaded: ', self.config_data)

    def get(self, key):
        # return None and prevent exception
        if not key in self.config_data:
            return None
        return self.config_data[key]

