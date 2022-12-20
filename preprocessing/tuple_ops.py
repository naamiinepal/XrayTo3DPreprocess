import math


def add_tuple(a, b):
    return tuple(a_i + b_i for a_i, b_i in zip(a, b))


def subtract_tuple(a, b):
    return tuple(a_i - b_i for a_i, b_i in zip(a, b))


def divide_tuple_scalar(a, b):
    b = (b,) * len(a)
    return tuple(a_i / b_i for a_i, b_i in zip(a, b))


def divide_tuple(a, b):
    return tuple(a_i / b_i for a_i, b_i in zip(a, b))


def multiply_tuple(a, b):
    return tuple(a_i * b_i for a_i, b_i in zip(a, b))


def ceil_tuple(a):
    return tuple(math.ceil(a_i) for a_i in a)


def floor_tuple(a):
    return tuple(math.floor(a_i) for a_i in a)