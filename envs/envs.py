




envs = {}
def register(id):
    def decorator(cls):
        envs[id] = cls
        return cls
    return decorator


def make(id, **kwargs):
    if id is None:
        return None
    env = envs[id](**kwargs)
    return env

