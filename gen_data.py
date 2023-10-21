import dataclasses

import numpy as np
import scipy.ndimage as sn
import tqdm
import numba


NUMBA_OPT = {"cache": True, "nopython": True}


@dataclasses.dataclass
class GenDataParam:
    """
    Parameters
    ----------
    num : int
        生成するデータ点の数

    size : tuple[int, ...]
        データ点分布の解像度

    power : float
        データ点分布の各スケールに乗する指数大きいほどより大きいスケールが優勢となる
        通常 -2 ~ +2 の範囲で使用する

    max_scale : int
        データ点分布の生成時に使用するガウシアンフィルターの最大サイズ
        大きいほど滑らかになり、計算負荷が増大する

    seed : int
        シード値デフォルト値は `None`

    use_float32 : bool
        `True` の場合、`np.float32` を使用する
        `False` の場合は `np.float64` を使用する
        デフォルト値は `False`

    attenuation_coef : float
        中心から遠い位置の分布を減衰させるときに使用する係数
        大きいほど強く減衰する。`0` で減衰しない
        デフォルト値は `0`
    """
    num: int
    size: tuple[int, ...]
    power: float
    max_scale: int
    attenuation_coef: float = 0.1
    use_float32: bool = True
    seed: int = None


@numba.jit(**NUMBA_OPT)
def _normalize_0_1(arr: np.ndarray) -> np.ndarray:
    val_range = np.max(arr) - np.min(arr)
    return (arr - np.min(arr)) / val_range


def _gen_limb_attenuation_dist(gen_data_param: GenDataParam) -> np.ndarray:
    """
    周辺減衰係数値を表す配列を返す

    Parameters
    ----------
    gen_data_param : GenDataParam
        データ点生成用のパラメータ

    Returns
    -------
    noise : np.ndarray
        周辺減衰係数値
    """
    if gen_data_param.use_float32:
        float_type = np.float32
    else:
        float_type = np.float64

    if gen_data_param.attenuation_coef == 0:
        return np.ones(gen_data_param.size, dtype=float_type)

    central = np.array(gen_data_param.size).astype(float_type) / 2
    grid_args = tuple([np.arange(s) for s in gen_data_param.size[::-1]])
    grid = np.array(np.meshgrid(*grid_args), dtype=float_type)
    distance_sq = np.zeros(gen_data_param.size, dtype=float_type)

    for d in range(len(gen_data_param.size)):
        dist_from_cen = grid[d] - central[d]
        distance_sq += dist_from_cen ** 2

    att_distr = np.exp(-distance_sq * gen_data_param.attenuation_coef / np.mean(gen_data_param.size))
    return att_distr


def _gen_noise_distribution(gen_data_param: GenDataParam) -> np.ndarray:
    """
    滑らかなノイズを生成する

    Parameters
    ----------
    gen_data_param : GenDataParam
        データ点生成用のパラメータ

    Returns
    -------
    noise : np.ndarray
        データ点分布
    """

    assert len(gen_data_param.size) < 31
    assert gen_data_param.attenuation_coef >= 0

    if gen_data_param.use_float32:
        float_type = np.float32
    else:
        float_type = np.float64

    if gen_data_param.seed is not None:
        np.random.seed(gen_data_param.seed)
    origin = np.random.random(gen_data_param.size).astype(float_type)
    noise = np.zeros(gen_data_param.size, dtype=float_type)

    max_scale = min(gen_data_param.max_scale, *gen_data_param.size)
    coef_arr = np.power(np.arange(1, max_scale + 1), gen_data_param.power).astype(float_type)
    coef_arr /= coef_arr.sum()
    for s, p in zip(range(1, max_scale + 1), coef_arr):
        noise += sn.gaussian_filter(origin, s) * p
    return noise


def _is_adopted(ref_val: float, point: np.ndarray, dist: np.ndarray, size: np.ndarray) -> bool:
    """
    与えられたデータ点 `point` を採用するかどうかを、データ点分布 `dist` と確率値 `ref_val` から判定する
    `point` での `dist` の値を簡易線形近似で算出して、その値が `ref_val` より大きければ `True` を返す

    Parameters
    ----------
    ref_val : float
        そのポイントを採用するかどうかの参照値

    point : np.ndarray
        サンプリングされた座標

    dist : np.ndarray
        データ分布を表す配列

    size : np.ndarray
        データ分布のサイズを表す配列

    Returns
    -------
    is_adopted : bool
    """
    scaled_point = point * (size - 1)
    floor_idx = np.floor(scaled_point).astype(np.int32)
    ceil_idx = np.ceil(scaled_point).astype(np.int32)
    floor_val = dist[tuple(floor_idx)]
    ceil_val = dist[tuple(ceil_idx)]
    apx_val = np.mean((ceil_idx - scaled_point) * ceil_val + (scaled_point - floor_idx) * floor_val)
    return ref_val < apx_val


@numba.jit(**NUMBA_OPT)
def _gen_dpd(noise: np.ndarray, limb_att_coef: np.ndarray) -> np.ndarray:
    """
    データ分布と周辺減衰を元にして最終的なデータ分布を返す

    Parameters
    ----------
    noise : np.ndarray
        データ分布
    limb_att_coef : np.ndarray
        周辺減衰

    Returns
    -------
    dpd : np.ndarray
    """
    dpd = _normalize_0_1(_normalize_0_1(noise) * _normalize_0_1(limb_att_coef))
    return dpd ** 2


def gen_data_points(gen_data_param: GenDataParam) -> tuple[np.ndarray, np.ndarray]:
    perlin = _gen_noise_distribution(gen_data_param)
    limb_att_coef = _gen_limb_attenuation_dist(gen_data_param)
    dpd = _gen_dpd(perlin, limb_att_coef)
    dim = len(gen_data_param.size)

    points = []
    size_arr = np.array(gen_data_param.size)
    with tqdm.tqdm(total=gen_data_param.num) as p_bar:
        while len(points) < gen_data_param.num:
            sampled_point = np.random.random(dim)

            if _is_adopted(np.random.random(), sampled_point, dpd, size_arr):
                points.append(sampled_point)
                p_bar.update(1)

    if gen_data_param.use_float32:
        return np.array(points, dtype=np.float32), dpd
    return np.array(points, dtype=np.float64), dpd


if __name__ == "__main__":
    param = GenDataParam(
        num=3000, size=(32, 32), power=0.6, max_scale=12, attenuation_coef=0.001, use_float32=True
    )

    sampling_points, distr = gen_data_points(param)
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(distr, origin="lower")
    # plt.plot(sampling_points[:, 0]*param.size[0], sampling_points[:, 1]*param.size[1], "o")

    plt.subplot(1, 2, 2)
    plt.hist2d(sampling_points[:, 1], sampling_points[:, 0])
    plt.show()
