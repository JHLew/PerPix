import torch.nn as nn
import torch


# normalizing (0, 1) to (-1, 1)
def preprocess(x):
    return x * 2 - 1


# denormalizing (-1, 1) to (0, 1)
def postprocess(x):
    return torch.clamp((x + 1) / 2, min=0, max=1)


# 3 dimensions for the encoded kernels;
# each in between the range of (0, 50), (0, 10), (0, 1)
code_max = torch.FloatTensor([[[[50]], [[10]], [[1]]]])


# default Upsampling block: uses Pixel Shuffling
class Upsample(nn.Module):
    def __init__(self, in_channels, scale_by, kernel_size=3, act='relu'):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_by ** 2), kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(scale_by)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'prelu':
            self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)

        return x


class SFT_layer(nn.Module):
    def __init__(self, n_channels, k_code=3):
        super(SFT_layer, self).__init__()
        self.v_size = k_code

        self.shared = nn.Sequential(
            nn.Conv2d(self.v_size + n_channels, n_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.beta_predictor = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
        )

        self.gamma_predictor = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
        )

    def forward(self, x, k_map):
        features = torch.cat([x, k_map], dim=1)
        shared_out = self.shared(features)
        beta = torch.sigmoid(self.beta_predictor(shared_out))
        gamma = self.gamma_predictor(shared_out)

        # we have mistakenly used the beta and gamma notations in the opposite way;
        # original beta and gamma values should be done in form of
        # _result = x * gamma + beta
        # for the original SFTMD model.

        _result = x * beta + gamma

        return _result


class SFT_res_block(nn.Module):
    def __init__(self, in_c, out_c, k_size):
        super(SFT_res_block, self).__init__()

        self.sft1 = SFT_layer(in_c, k_size)
        self.sft2 = SFT_layer(in_c, k_size)

        self.act_conv1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_c, in_c, 3, 1, 1)
        )

        self.act_conv2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_c, out_c, 3, 1, 1)
        )

    def forward(self, x, k_map):
        out_1 = self.sft1(x, k_map)
        out_1 = self.act_conv1(out_1)
        out_2 = self.sft2(out_1, k_map)
        out_2 = self.act_conv2(out_2)
        _result = out_2 + x

        return _result


class PerPix_SFTMD(nn.Module):
    def __init__(self, scale, code_len):
        super(PerPix_SFTMD, self).__init__()
        self.scale = scale
        self.k_code = code_len

        self.head = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

        body = [SFT_res_block(64, 64, self.k_code) for _ in range(16)]
        self.body = nn.Sequential(*body)
        self.final_sft = SFT_layer(64, self.k_code)
        self.tail = nn.Sequential(
            Upsample(64, scale_by=self.scale),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x, k_map):
        x = preprocess(x)

        head_result = self.head(x)
        body_result = head_result
        for i in range(16):
            body_result = self.body[i](body_result, k_map)

        out = head_result + body_result
        out = self.final_sft(out, k_map)
        _result = self.tail(out)

        _result = postprocess(_result)

        return _result


class Predictor(nn.Module):
    def __init__(self, scale, code_len=3):
        super(Predictor, self).__init__()
        self.scale = scale
        self.code_len = code_len

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, code_len, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = preprocess(x)
        _result = self.layers(x)
        _result = _result * code_max.cuda()

        return _result