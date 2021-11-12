import numpy as np
from scipy.stats import multivariate_normal


def pixel_to_distribution(kernel):
    kernel = normalize_kernel(kernel)
    h, w = kernel.shape
    data = []
    for i in range(h):
        for j in range(w):
            val = kernel[i, j]
            val = np.round(val * 1000).astype(np.uint16)
            data += [[i, j]] * val

    data = np.array(data)
    return data


def get_cov(kernel):
    k = normalize_kernel(kernel)
    h, w = kernel.shape
    x, y = np.mgrid[:h, :w]

    k = k * 1000
    k = np.round(k).astype(np.uint16)

    n = k.sum()
    e_x = ((x * k).sum() / n)
    e_y = ((y * k).sum() / n)
    ex2 = e_x ** 2
    ey2 = e_y ** 2
    e2x = ((x ** 2) * k).sum() / n
    e2y = ((y ** 2) * k).sum() / n
    e_xy = ((x * y) * k).sum() / n

    var_x = e2x - ex2
    var_y = e2y - ey2
    cov_xy = e_xy - e_x * e_y

    cov_mat = np.array([[var_x, cov_xy],
                        [cov_xy, var_y]])
    return cov_mat


def reconstruct_from_cov(cov, mean=(24, 24), size=(49, 49)):
    _size = size[0] * size[1]
    x, y = np.mgrid[:size[0], :size[1]]
    pos = np.dstack((x, y))
    try:
        mn = multivariate_normal(mean, cov)
        k = mn.pdf(pos)
    except:
        k = np.ones(size)

    k = k / k.sum()  # make sure it sums to one

    return k


def cov_to_eig(cov, sort=True):
    w, v = np.linalg.eig(cov)
    if sort:
        sort_idx = np.argsort(w)
        w = w[sort_idx]
        v = v.transpose((1, 0))[sort_idx].transpose((1, 0))

    return w, v


def eig_to_cov(w, v):
    return np.dot(np.dot(v, np.diagflat(w)), np.linalg.inv(v))


def normalize_vector(v):
    norm = np.linalg.norm(v)
    v = v / norm

    return v, norm


def normalize_kernel(k):
    _min = k.min()
    if _min < 0:
        k = k - _min
    _max = k.max()
    if _max != 0:
        k = (k / _max)

    return k


def get_main_v(eig_v):
    v = eig_v[:, 0]
    main_idx = 0

    if v[0] * v[1] < 0:
        v = eig_v[:, 1]
        main_idx = 1

    v = np.abs(v)

    return v, main_idx


def main_to_eig(v):
    main_v = v.copy()
    v = np.flip(v)
    if v[0] * v[1] < 0:
        v = np.abs(v)
    else:
        v[1] = -v[1]

    eig_v = np.stack([main_v, v], axis=1)
    return eig_v


def encode_cov(cov):
    w, v = cov_to_eig(cov)
    v, main_idx = get_main_v(v)
    if main_idx != 0:
        w = np.flip(w, 0)
    w, norm = normalize_vector(w)

    w_ratio = vector_2_ratio(w)
    v_ratio = vector_2_ratio(v)

    if v_ratio > 1:
        w_ratio = 1 / w_ratio
        v_ratio = 1 / v_ratio

    return np.array([norm, w_ratio, v_ratio])


def decode_to_cov(code):
    norm, w_ratio, v_ratio = code[0], code[1], code[2]

    w = ratio_2_vector(w_ratio)
    v = ratio_2_vector(v_ratio)

    w, _ = normalize_vector(w)
    v, _ = normalize_vector(v)
    w = norm * w
    v = main_to_eig(v)

    return eig_to_cov(w, v)


def vector_2_ratio(vector):
    return vector[0] / (vector[1] + 1e-8)


def ratio_2_vector(ratio):
    return np.array([ratio, 1])

