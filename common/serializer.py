import pickle


def serialize(obj):
    return pickle.dumps(obj)


def deserialize(data):
    return pickle.loads(data)
