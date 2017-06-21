def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y


def kld_coef(i):
    import math
    return (math.tanh((i - 13000) / 5000) + 1) / 2
