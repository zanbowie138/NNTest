def reLU(x):
    return max(0, x)

def reLU_deriv(x):
    if x > 0:
        return 1
    else:
        return 0
