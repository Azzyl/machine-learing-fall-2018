import numpy as np
a = 0
b = np.array([])
b = np.hstack((b,2))
print(b)
def find_w_b(alphas, y, x, sv_inds, kernel, thresh, C):
    w = 0
    b = 0
    if kernel == linear_kernel:
        for i in sv_inds:
            w += np.dot(alphas[i] * y[i], x[i])
        important_index = sv_inds[0]
        for i in sv_inds:
            if 0 < alphas[i] + thresh < C:
                important_index = i
                break
        b = y[important_index] - np.dot(w, x[important_index])
    elif kernel == polynomial_kernel:
        important_index = -1
        for i in sv_inds:
            if 0 < alphas[i] + thresh < C:
                important_index = i
                break
        some_sum = 0
        for i in sv_inds:
            some_sum += alphas[i] * y[i] * kernel(x[i], x[important_index])
        b = y[important_index] - some_sum
    if kernel == linear_kernel:
        return w, b
    elif kernel == polynomial_kernel:
        return None, b