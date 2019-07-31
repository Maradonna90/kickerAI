import json

class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
    def save(self, filename):
        with open(filename+'.json', 'w') as fp:
            json.dump(self, fp, sort_keys=True, indent=4)
    def load(self, filename):
        with open(filename+'.json', 'r') as fp:
            self.clear()
            self.update(json.load(fp))
            self.inverse.clear()
            for key, value in self.items():
                self.inverse.setdefault(value,[]).append(key)
