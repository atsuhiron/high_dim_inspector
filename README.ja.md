# high_dim_inspector
高次元のデータ点の分布を調査するためのツール群です。

## 共通ルール
データは `numpy.ndarray` で表します。データセットの形状は `(N, D)` で `N` がデータ数、 `D` が次元数であることを要請します。

## 各ツールの説明
### gen_data
サンプルやその他のツールの評価用のデータを生成するためのスクリプトです。

```python
import gen_data

param = gen_data.GenDataParam(
    num=3000,               ## 生成するデータ点の数
    size=(32, 32),          ## データ生成に使用するデータ点分布関数の解像度
    power=0.6,              ## データ点分布関数の生成に使用する各スケールのノイズに乗する指数
    max_scale=12,           ## データ点分布関数の生成に使用するガウシアンフィルタの最大サイズ
    attenuation_coef=0.001, ## 周辺減衰の強度
    use_float32=True        ## np.float32 の使用
)

sampling_points, distribution = gen_data.gen_data_points(param)
```

`sampling_points` がデータ点を表す配列で、形状は `(num, len(size))` です。

`distribution` がデータ点分布関数で、形状は `size` と同じです。
> [!NOTE]
> `distribution` はあるグリッドでデータ点が生成される確率を表したものであり、統計学の文脈で使われる**確率分布関数**ではありません。(全空間を積分しても `1` にはなりません)

### sparse_dense_booster
データ点の分布の疎密を増幅させるためのスクリプトです。

```python
import numpy as np
import sparse_dense_booster as sdb

param = sdb.SDBParam(
        iter_num=10,      ## イテレーション回数
        min_dist=0.05,    ## 引力が最大値になる距離
        pot_peak=0.6,     ## 引力と斥力が切り替わる距離
        amplitude=0.6,    ## 力の強さを表す係数
        delta_t=0.01      ## 時間の刻み幅
    )

points = np.random.random((10, 2)).astype(np.float32)
boosted_points = sdb.boost(points, param)
```

`boosted_points` が疎密が増幅されたデータ点の座標です。
増幅作用は `SDBParam` の値に対してかなり鋭敏に反応するので、ほとんどの場合試行錯誤を繰り返すことになると思います。