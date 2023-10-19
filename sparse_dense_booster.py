import dataclasses

import numpy as np
import numba
import tqdm


NUMBA_OPT = {"cache": True, "nopython": True}


@dataclasses.dataclass
class SDBParam:
    """
    Parameters
    ----------
    iter_num : int
        疎密増幅の実行回数

    min_dist : float
        これより小さい距離では引力が一定値になる
        非常に接近した際の挙動が安定する

    pot_peak : float
        引力・斥力が切り替わる距離
        これより遠い距離では斥力が働くようになる

    amplitude : float
        引力・斥力に乗する値
    
    delta_t : float
        時間幅を表す量
        `iter_num` と `delta_t` の積が一定の場合、得られる結果が同じになり、後者の方がより安定した計算になる
        ただし、桁落ちが顕著になるほど `delta_t` を小さくすると結果が正確でなくなる
    """

    iter_num: int
    min_dist: float
    pot_peak: float
    amplitude: float
    delta_t: float

    def calc_shift(self) -> np.float32:
        return np.float32(self.amplitude / (self.pot_peak ** 2))

    def calc_force_uplim(self) -> np.float32:
        if self.min_dist <= 0:
            return np.float32("inf")

        return self.get_amp() * np.float32(1 / (self.min_dist ** 2) - 1 / (self.pot_peak ** 2))

    def get_amp(self) -> np.float32:
        return np.float32(self.amplitude)

    def get_delta_t(self) -> np.float32:
        return np.float32(self.delta_t)


@numba.jit(**NUMBA_OPT)
def calc_relative_pos(pos: np.ndarray) -> np.ndarray:
    """
    相対位置を計算する

    Parameters
    ----------
    pos : np.ndarray
        絶対位置を表す配列
        shape は `(N, D)`
        dtype は `np.float32`

    Returns
    -------
    relative_pos : np.ndarray
        相対位置を表す配列
        shape は `(N, N, D)`
        dtype は `np.float32`
        点 i に対する点 j の相対ベクトルの各成分 d は `relative_pos[i, j, d] で表される`
    """
    num = len(pos)
    dim = len(pos[0])

    hori = np.ones((num, num, dim), dtype=np.float32) * pos[np.newaxis, :]
    vert = np.ones((num, num, dim), dtype=np.float32) * pos[:, np.newaxis]
    return hori - vert


@numba.jit(**NUMBA_OPT)
def calc_dist_sq(relative_pos: np.ndarray) -> np.ndarray:
    """
    相対位置をもとに、ユークリッド距離の2乗を計算する

    Parameters
    ----------
    relative_pos : np.ndarray
        相対位置を表す配列
        shape は `(N, N, D)`
        dtype は `np.float32`

    Returns
    -------
    dist_sq : np.ndarray
        距離の2乗を表す配列
        shape は `(N, N)`
        dtype は `np.float32`
    """
    distance_sq_by_dim = relative_pos ** 2
    return distance_sq_by_dim.sum(axis=2)


@numba.jit(**NUMBA_OPT)
def calc_force(pos: np.ndarray, shift: np.float32, amp: np.float32, force_uplim: np.float32) -> np.ndarray:
    """
    各質点にかかる力を計算する
    
    Parameters
    ----------
    pos : np.ndarray
        絶対座標を表す配列
        shape は `(N, D)`
        dtype は `np.float32`
        
    shift : np.float32
        逆2乗則からのずれを表す量
        大きい程斥力が優勢になる
        
    amp : np.float32
        力に乗する値
    
    force_uplim : np.float32
        引力の上限値 

    Returns
    -------
    force : np.ndarray
        各質点にかかる力を成分ごとに表した配列
        shape は `(N, N, D)`
        dtype は `np.float32`
    """
    # (N, N, D)
    relative_pos = calc_relative_pos(pos)

    # (N, N)
    non_zero_dist_sq = (calc_dist_sq(relative_pos) + np.eye(len(relative_pos)))

    # (N, N)
    force_coef = np.minimum(amp / non_zero_dist_sq - shift, force_uplim)

    # (N, N, D)
    unit_rel_vec = relative_pos / np.sqrt(non_zero_dist_sq)[:, :, np.newaxis]

    return -unit_rel_vec * force_coef[:, :, np.newaxis]


@numba.jit(parallel=True, **NUMBA_OPT)
def normalize(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    分布を次元ごとに正規化する

    Parameters
    ----------
    pos : np.ndarray
        座標を表す配列
        shape は `(N, D)`

    Returns
    -------
    normalized_pos, mean, std : tuple[np.ndarray, np.ndarray, np.ndarray]
        正規化された座標と次元ごとの平均、次元ごとの標準偏差を表すタプル
        shape はそれぞれ `(N, D)`, `(1, D)`, `(1, D)`
        dtype は全て `np.float32`
    """
    std = np.zeros((1, pos.shape[1]), dtype=np.float32)
    mean = np.zeros((1, pos.shape[1]), dtype=np.float32)
    for j in range(pos.shape[1]):
        std[0, j] = pos[:, j].std()
        mean[0, j] = pos[:, j].mean()
    return (pos - mean) / std, mean, std


@numba.jit("Tuple((f4[:, :], f4[:, :]))(f4[:, :], f4[:, :], f4, f4, f4, f4)", **NUMBA_OPT)
def step(pos: np.ndarray,
         vel: np.ndarray,
         shift: np.float32,
         amp: np.float32,
         force_uplim: np.float32,
         delta_t: np.float32) -> tuple[np.ndarray, np.ndarray]:
    """
    疎密増幅の1ステップを実行する

    Parameters
    ----------
    pos : np.ndarray
        絶対座標を表す配列
        shape は `(N, D)`
        dtype は `np.float32`

    vel : np.ndarray
        速度を表す配列
        shape, dtype ともに `pos` と同じ

    shift : np.float32
        逆2乗則からのずれを表す量
        大きい程斥力が優勢になる

    amp : np.float32
        力に乗する値

    force_uplim : np.float32
        引力の上限値

    delta_t : float
        時間幅を表す量

    Returns
    -------
    pos, vel : tuple[np.ndarray, np.ndarray]
        絶対座標と速度を表す配列のペア
        shape は `(N, D)`
        dtype は `np.float32`
    """
    f = calc_force(pos, shift, amp, force_uplim).sum(axis=0)
    vel += f * delta_t
    pos += vel * delta_t
    return pos, vel


def boost(points: np.ndarray, sdb_param: SDBParam) -> np.ndarray:
    """
    疎密増幅を実行する

    Parameters
    ----------
    points : np.ndarray
        疎密増幅を実行したい質点の座標
        shape は `(N, D)` で `N` が質点の数、`D` が次元を表す
        dtype は `np.float32`

    sdb_param : SDBParam
        パラメータクラス

    Returns
    -------
    points : np.ndarray
        疎密増幅された質点の座標
        shape は `(N, D)`
        dtype は `np.float32`
    """
    points, m, s = normalize(points)
    _vel = np.zeros_like(points, dtype=np.float32)

    for _ in tqdm.tqdm(range(sdb_param.iter_num)):
        points, _vel = step(points, _vel,
                            sdb_param.calc_shift(), sdb_param.get_amp(),
                            sdb_param.calc_force_uplim(), sdb_param.get_delta_t())
        points, _, _ = normalize(points)
    return points * s + m


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(4224)
    param = SDBParam(
        iter_num=10,
        min_dist=0.05,
        pot_peak=0.6,
        amplitude=0.6,
        delta_t=0.01
    )

    _pos = np.random.random((500, 6)).astype(np.float16).astype(np.float32)
    plt.plot(_pos[:, 1], _pos[:, 0], "o", label="before")
    _pos = boost(_pos, param)
    plt.plot(_pos[:, 1], _pos[:, 0], "o", label="after")
    plt.legend()
    plt.show()
