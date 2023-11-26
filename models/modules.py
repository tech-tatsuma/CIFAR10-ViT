import torch
import torch.nn as nn


class PreNorm(nn.Module): # 正規化を行うPreNormクラス
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)  # レイヤー正規化を初期化
        self.fn = fn  # 関数fnを保存

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 正規化を適用した後に関数fnを適用


class Attention(nn.Module):  # 自己注意機構を行うAttentionクラス
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super(Attention, self).__init__()

        # 埋め込みの次元がヘッド数で割り切れることを確認
        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads  # ヘッド数を保存
        head_dim = dim // num_heads  # ヘッドごとの次元数
        self.scale = head_dim ** -0.5  # スケーリング係数

        # Query, Key, Valueを生成する線形レイヤー
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # アテンションドロップアウト
        self.proj = nn.Linear(dim, dim)  # 最終的な投影を行う線形レイヤー
        self.proj_drop = nn.Dropout(proj_drop)  # 投影ドロップアウト

    def forward(self, x):
        B, N, C = x.shape  # バッチサイズB、トークン数N、チャンネル数Cを取得
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
          # Query, Key, Valueを取得
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale # アテンションスコアの計算
        attn = attn.softmax(dim=-1)  # ソフトマックス関数を適用
        attn = self.attn_drop(attn) # ドロップアウトの適用

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # アテンションの適用と整形
        x = self.proj(x)  # 投影の適用
        x = self.proj_drop(x)  # ドロップアウトの適用
        return x # 出力を返す


class FeedForward(nn.Module):
    """
    Transformerのフィードフォワードネットワークを実装
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.0, revised=False):
        super(FeedForward, self).__init__()
        if not revised:
            """
            Original: https://arxiv.org/pdf/2010.11929.pdf
            """
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, dim),
            )
        else:
            """
            Scaled ReLU: https://arxiv.org/pdf/2109.03810.pdf
            """
            self.net = nn.Sequential(
                nn.Conv1d(dim, hidden_dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Conv1d(hidden_dim, dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            )

        self.revised = revised # 改訂フラグを保存
        self._init_weights()  # 重みの初期化

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.bias, std=1e-6)  # バイアスの初期化

    def forward(self, x):
        if self.revised:
            # 改訂されたネットワークの場合の処理
            x = x.permute(0, 2, 1)
            x = self.net(x)
            x = x.permute(0, 2, 1)
        else:
            # オリジナルのネットワークの場合の処理
            x = self.net(x)

        return x


class OutputLayer(nn.Module):  # 出力層を実装するクラス
    def __init__(
        self,
        embedding_dim,  # 埋め込み次元
        num_classes=1000,  # クラス数
        representation_size=None,  # 表現サイズ
        cls_head=False,  # クラスヘッドを使用するかのフラグ
    ):
        super(OutputLayer, self).__init__()

        self.num_classes = num_classes # クラス数を保存
        modules = []
        if representation_size:
            # 表現サイズが指定されている場合のレイヤー構成
            modules.append(nn.Linear(embedding_dim, representation_size))
            modules.append(nn.Tanh())
            modules.append(nn.Linear(representation_size, num_classes))
        else:
            # 表現サイズが指定されていない場合のレイヤー構成
            modules.append(nn.Linear(embedding_dim, num_classes))

        self.net = nn.Sequential(*modules)  # レイヤーのシーケンスを作成

        if cls_head:
            self.to_cls_token = nn.Identity()  # クラスヘッドがある場合、Identityレイヤーを使用

        self.cls_head = cls_head  # クラスヘッドフラグを保存
        self.num_classes = num_classes  # クラス数を保存
        self._init_weights()  # 重みの初期化

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                # 線形レイヤーの重みを初期化
                if module.weight.shape[0] == self.num_classes:
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.cls_head:
            # クラスヘッドがある場合の処理
            x = self.to_cls_token(x[:, 0])
        else:
            """
            Scaling Vision Transformer: https://arxiv.org/abs/2106.04560
            """
            # クラスヘッドがない場合の処理（Scaling Vision Transformer方式）
            x = torch.mean(x, dim=1)

        return self.net(x) # レイヤーシーケンスを通して出力を返す