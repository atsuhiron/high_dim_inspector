# high_dim_inspector
高次元のデータ点の分布を調査するためのツール群です。

## 1. Requirements
Python >= 3.10

## 2. 共通ルール
データは `numpy.ndarray` で表します。データセットの形状は `(N, D)` で `N` がデータ数、 `D` が次元数であることを要請します。

## 3. 各ツールの説明
### 3-1. gen_data
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

### 3-2. sparse_dense_booster
データ点の分布の疎密を増幅させるためのスクリプトです。

```python
import numpy as np
import sparse_dense_booster as sdb

points = np.random.random((10, 2)).astype(np.float32)

param = sdb.create_sdb_param(
    points,
    base_amp=0.6,    ## 増幅の強さを表す係数
    t_end=0.2        ## 増幅を作用させる時間
)
boosted_points = sdb.boost(points, param)
```

`boosted_points` が疎密が増幅されたデータ点の座標です。
増幅作用は `SDBParam` の値に対してかなり鋭敏に反応するので、ほとんどの場合試行錯誤を繰り返すことになると思います。
通常は上記のようなヘルパー関数を使用しますが、すべての値を手動で設定することも可能です。

```python
import sparse_dense_booster as sdb

param = sdb.SDBParam(
    iter_num=10,      ## イテレーション回数
    min_dist=0.05,    ## 引力が最大値になる距離
    pot_peak=0.6,     ## 引力と斥力が切り替わる距離
    amplitude=0.6,    ## 力の強さを表す係数
    delta_t=0.01      ## 時間の刻み幅
)
```

#### 3-2-1. 数理
それぞれのデータ点を質量の等しい質点とみなし、仮想的な力 $f$ を作用させて運動させます。  
時刻 $t+1$ での質点 $i$ の座標 $\vec{x}^{(t + 1)}_{i}$ は以下の式で表されます。

```math
\vec{x}^{(t + 1)}_{i} = \frac{1}{2} \sum_{j \neq i} \vec{f}^{(t)}_{ij} \Delta t^2 + \vec{x}^{(0)}_i
```

ここで $\Delta t$, $\vec{x}^{(0)}_i$ がそれぞれ時間の刻み幅、質点 $i$ の初期位置を表します。 

以下、簡単のため時間を表す添え字を省略します。作用させる力 $f$ は他の各質点との距離に応じて変化します。
二つの質点 $i, j$ の相対位置ベクトルを $\vec{r} _{ij} = \vec{x} _{j} - \vec{x} _{i}$ 、その $L^2$ ノルムを $r _{ij} = |\vec{r} _{ij}|$ とおけば、力は以下のように表されます。

```math
\vec{f}^{(t)}_{ij} = A \left( \frac{1}{r^2_{ij}} - \frac{1}{R^2_p}\right) \frac{\vec{r}_{ij}}{r_{ij}}
```

ここで $A$ 及び $R_p$ は定数です。 $A$ は力に乗する比例定数で、データの次元数 $d$ とすると、

```math
A = \sqrt{\frac{d}{2}} {\rm (base\_amp)}
```

で表されます。 $R_p$ は `pot_peak` の値がそのまま使用されます。  
$r_{ij}$ が $R_p$ の値を超える（つまり $i$ と $j$ が遠くに位置する）と $\vec{f}$ の符号が反転し、引力が斥力に変わります。
これにより、近いペアはより近くに移動し、遠いペアはより遠くに移動することになります。

また、最初の式によって計算された質点の移動量はそのまま使われず、以下の式による正規化が行われる。

```math
\hat{\vec{x}}_{(d)} = (\vec{x}_{(d)} - \vec{\mu}_{(d)}) \oslash \vec{\sigma}_{(d)}  
```

ここで $\vec{\mu}$, $\vec{\sigma}$ はそれぞれ平均と標準偏差を表すベクトルで、添え字の $(d)$ は各次元での値を表します
（つまり $\vec{x}_ {i}$ は `points[i, :]` で $\vec{x} _{(d)}$ は `points[:, d]` ）。
この正規化ステップを飛ばすと、`pot_peak` の値次第では質点が無限に拡散してしまいます。