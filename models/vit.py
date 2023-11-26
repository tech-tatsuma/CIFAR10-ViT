import torch.nn as nn

from .patch_embed import EmbeddingStem
from .transformer import Transformer
from .modules import OutputLayer

# Vision Transformerの定義
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224, # 画像サイズ
        patch_size=16, # パッチサイズ
        in_channels=3, # 入力チャンネル数(RGBの場合は３、グレースケールの場合は１)
        embedding_dim=768, # 埋め込み次元のデフォルト値を768に設定
        num_layers=12, # Transformerの層数のデフォルト値を12に設定
        num_heads=12, # マルチヘッドアテンションのヘッド数のデフォルト値を12に設定
        qkv_bias=True, # QKV（Query, Key, Value）バイアスを使用するかのデフォルト設定
        mlp_ratio=4.0, # MLPレイヤーのサイズ比率のデフォルト値を4.0に設定
        use_revised_ffn=False, # 改訂されたフィードフォワードネットワークを使用するかのデフォルト設定
        dropout_rate=0.0, # ドロップアウト率のデフォルト値を0.0に設定
        attn_dropout_rate=0.0, # アテンションドロップアウト率のデフォルト値を0.0に設定
        use_conv_stem=True, # 畳み込みステム層を使用するかのデフォルト設定
        use_conv_patch=False, # 畳み込みパッチを使用するかのデフォルト設定
        use_linear_patch=False, # 線形パッチを使用するかのデフォルト設定
        use_conv_stem_original=True, # オリジナルの畳み込みステムを使用するかのデフォルト設定
        use_stem_scaled_relu=False, # スケールされたReLUを使用するかのデフォルト設定
        hidden_dims=None, # 隠れ層の次元数のデフォルト設定
        cls_head=False, # クラスヘッドを使用するかのデフォルト設定
        num_classes=1000, # クラス数のデフォルト値を1000に設定
        representation_size=None, # 表現サイズのデフォルト設定
    ):
        super(VisionTransformer, self).__init__()

        # 埋め込み層の初期化
        self.embedding_layer = EmbeddingStem(
            image_size=image_size,
            patch_size=patch_size,
            channels=in_channels,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            conv_patch=use_conv_patch,
            linear_patch=use_linear_patch,
            conv_stem=use_conv_stem,
            conv_stem_original=use_conv_stem_original,
            conv_stem_scaled_relu=use_stem_scaled_relu,
            position_embedding_dropout=dropout_rate,
            cls_head=cls_head,
        )

        # Transformer層の初期化
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )
        self.post_transformer_ln = nn.LayerNorm(embedding_dim) # トランスフォーマー後のレイヤーノーマリゼーション層

        # 出力層の初期化
        self.cls_layer = OutputLayer(
            embedding_dim,
            num_classes=num_classes,
            representation_size=representation_size,
            cls_head=cls_head,
        )

    def forward(self, x):
        x = self.embedding_layer(x) # 入力xを埋め込み層に通す
        x = self.transformer(x)  # 埋め込み層の出力をトランスフォーマー層に通す
        x = self.post_transformer_ln(x)  # トランスフォーマー層の出力をレイヤーノーマリゼーション層に通す
        x = self.cls_layer(x)  # レイヤーノーマリゼーション層の出力をクラス出力層に通す
        return x  # 最終的な出力を返す