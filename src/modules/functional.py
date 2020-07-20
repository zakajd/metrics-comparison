import collections


def listify(p):
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    return p
