import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors import safe_open
from timm.layers import NormMlpClassifierHead
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from reconstruction import CustomNetwork

class AttentionWeighting(nn.Module):
    def __init__(self, feature_dim=4096, num_classifiers=3):
        super(AttentionWeighting, self).__init__()
        self.attention_net = nn.Linear(feature_dim, num_classifiers)

    def forward(self, features, classifiers_outputs):
        
        attention_weights = F.softmax(self.attention_net(features), dim=1)
        attention_weights = attention_weights.unsqueeze(1)

        stacked_outputs = torch.stack(classifiers_outputs, dim=1)

        weighted_outputs = torch.bmm(attention_weights, stacked_outputs)

        weighted_outputs = weighted_outputs.squeeze(1)
        return weighted_outputs


class Decoder(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super(Decoder, self).__init__()

        # 上采样层
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.bn5 = nn.BatchNorm2d(32)

        self.out_conv = nn.Conv2d(32, num_classes, 1)
        self.bn6 = nn.BatchNorm2d(2)

    def forward(self, x, output_size):


        x = F.gelu(self.bn1(self.up1(x))) # 512 * 14 * 14
        x = F.gelu(self.bn2(self.up2(x))) # 256 * 28 * 28
        x = F.gelu(self.bn3(self.up3(x))) # 128 * 56 * 56
        x = F.gelu(self.bn4(self.up4(x)))
        x = F.gelu(self.bn5(self.up5(x)))

        x = F.interpolate(x, size=(output_size, output_size), mode='bilinear', align_corners=False)

        x = F.gelu(self.bn6(self.out_conv(x)))
        
        
        return x

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_size=1024):
        super(SimpleSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, y):
        # x shape: (seq_len, embed_size), seq_len for your case is 2

        Q = self.query(x)  # Query
        K = self.key(y)    # Key
        V = self.value(y)  # Value

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.embed_size ** 0.5
        attention = F.softmax(attention_scores, dim=-1)
        weighted_value = torch.matmul(attention, V)

        return weighted_value


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size=1024, output_dim=2, ff_hidden_size=2048, dropout=0.3):
        super(TransformerEncoderLayer, self).__init__()
        self.model = timm.create_model('convnextv2_base', pretrained=False)
        self.self_attention_x = SimpleSelfAttention(embed_size)
        self.self_attention_y = SimpleSelfAttention(embed_size)
        self.r_model = CustomNetwork()

        self.norm1_x = nn.LayerNorm(embed_size)
        self.norm2_x = nn.LayerNorm(embed_size)
        self.norm1_y = nn.LayerNorm(embed_size)
        self.norm2_y = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embed_size)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2 * embed_size,
                nhead=4,
                dim_feedforward=ff_hidden_size
            ),
            num_layers=1
        )
        self.dropout = nn.Dropout(dropout)
        self.mse = nn.MSELoss()
        self.fc = nn.Linear(embed_size * 2, output_dim)
        self.loss = nn.CosineEmbeddingLoss()
        self.global_pool = SelectAdaptivePool2d(pool_type='avg')
        self.flatten = nn.Flatten(1)
        self.head = NormMlpClassifierHead(1024, 2, hidden_size=None, pool_type='avg',
                                          drop_rate=0, norm_layer='layernorm2d', act_layer='gelu')

        tensors = {}
        ## determined load pretarined parameters (by youself)
        with safe_open("model.safetensors", framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        self.model.load_state_dict(tensors, strict=False)
        self.decoder = Decoder()
        self.att = AttentionWeighting()

    def process_dual_images(self, x, y):
        input_size = x.shape[2]
        x = self.model.forward_features(x)
        y = self.model.forward_features(y)

        xx = x
        yy = y

        x1 = self.head(x)
        y1 = self.head(y)

        x = self.global_pool(x)
        y = self.global_pool(y)

        x = self.flatten(x)
        y = self.flatten(y)
        xf = x
        yf = y
        xxf = self.r_model(xf.detach())
        mse = self.mse(xxf, yf)
        attention_x = self.self_attention_x(x, y)
        x = self.norm1_x(attention_x + x)
        x = self.norm2_x(self.feed_forward(x) + x)

        # y的自注意力
        attention_y = self.self_attention_y(y, x)
        y = self.norm1_y(attention_y + y)
        y = self.norm2_y(self.feed_forward(y) + y)

        # 拼接
        concat_xy = torch.cat((x, y), dim=1)
        encoded = self.transformer_encoder(concat_xy)
        encoded = self.dropout(encoded)
        output = self.fc(encoded)

        xd = self.decoder(xx, input_size)
        yd = self.decoder(yy, input_size)

        ff = torch.cat([xf,encoded,yf],dim=1).detach()
        weight_out = self.att(ff,[x1,output,y1])


        return weight_out,xd,yd,mse

    def forward(self, x, y):
        output_dual, xd_dual, yd_dual, mse_dual = self.process_dual_images(x,y)

        return output_dual, xd_dual, yd_dual, mse_dual