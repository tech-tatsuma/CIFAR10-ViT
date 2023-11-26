from torch import nn
from .modules import Attention, FeedForward, PreNorm

# Transformerの定義
class Transformer(nn.Module):
    def __init__(
        self,
        dim, # 埋め込み次元
        depth, # Transformerの層の深さ
        heads, # マルチヘッドアテンションのヘッド数
        mlp_ratio=4.0, # MLPレイヤーの比率
        attn_dropout=0.0, # アテンションドロップアウト率
        dropout=0.0, # ドロップアウト率
        qkv_bias=True, # QKVバイアスの有無
        revised=False, # 改訂されたフィードフォワードネットワークの使用有無
    ):
        super().__init__()
        self.layers = nn.ModuleList([]) # レイヤーを格納するためのモジュールリストの初期化

        # MLPレイヤーの比率が浮動小数点数であることを確認
        assert isinstance(
            mlp_ratio, float
        ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim) # MLPレイヤーの次元を計算

        for _ in range(depth):  # 指定された深さまで繰り返す
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate=dropout,),
                        )
                        if not revised
                        else FeedForward(
                            dim, mlp_dim, dropout_rate=dropout, revised=True,
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:  # 各レイヤーで処理を行う
            x = attn(x) + x  # アテンションレイヤーの適用と残差接続
            x = ff(x) + x # フィードフォワードレイヤーの適用と残差接続
        return x # 処理済みのデータを返す