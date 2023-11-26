import torch
import math
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    無駄な勾配計算を避けながら、テンソルを切断正規分布で初期化する内部関数
    """
    def norm_cdf(x):
        # 標準正規分布の累積分布関数を計算
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    # 平均値が許容範囲外にある場合に警告を発する
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # 勾配の計算を行わないコンテキスト内で処理

        # 切断された一様分布からの値を用いてテンソルを埋め、
        # 正規分布の逆累積分布関数を用いて変換する
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # 一様分布から得られた値を使ってテンソルを埋め、
        # 標準正規分布の逆累積分布関数で変換する
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        # エラー関数の逆関数を適用
        tensor.erfinv_()
        # 標準偏差と平均値を適用して変換
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        # 値が[a, b]の範囲内になるようにクランプ
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    """
    切断正規分布を使用して入力テンソルを値で埋める関数。
    この値は、通常の正規分布 \mathcal{N}(mean, std^2) から引き出され、
    [a, b] の範囲外の値は、範囲内になるまで再抽選される。
    このランダム値生成メソッドは、a <= mean <= b の時に最も効果的に機能する。

    引数:
        tensor: n次元の `torch.Tensor`
        mean: 正規分布の平均値
        std: 正規分布の標準偏差
        a: 最小切断値
        b: 最大切断値

    例:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b) # 内部関数を呼び出してテンソルを初期化