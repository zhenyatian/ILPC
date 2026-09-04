import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from geoopt.manifolds.stereographic import PoincareBall
from models.AtlasNet.atlasnet import Atlasnet
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from einops import rearrange, repeat
#from tutel import moe as tutel_moe


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def Poincare_dist(x, y, c=1.0):
    """
    Computes Poincare distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    manifold = PoincareBall(c=c)

    # x = manifold.projx(x)
    # y = manifold.projx(y)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    hx = manifold.projx(x)
    hy = manifold.projx(y)

    return manifold.dist2(hx, hy)


class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = torch.cat([knn_x, position_embed],dim=1)
        # knn_x_w *= position_embed

        return knn_x_w


class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        # self.geo_extract = PosE_Geo(3, 72, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x): #b*n*3, b*n*d, b*n*k*3, b*n*k*d
        knn_x = knn_x.permute(0,2,1,3)
        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=2)
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        # knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        # knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        # knn_x = knn_x.permute(0, 3, 1, 2)
        # knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x#knn_x_w.permute(0, 2, 3, 1)

class SGW(nn.Module):
    def __init__(self):
        super().__init__()
        self.group = 32
        #self.register_buffer('running_mean', torch.zeros(self.group, 1))
        #self.register_buffer('running_projection', torch.eye(self.group))

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        B, K, N, C = knn_x.shape

        knn_x = knn_x.permute(0,2,1,3)
        #mean_x = lc_x.unsqueeze(dim=2)
        #knn_x = (knn_x - mean_x).view(B, N, K, self.group, C // self.group).reshape(-1, C)
        knn_x = knn_x.view(B, N, K, self.group, C // self.group).reshape(-1, C)

        new_idx = torch.randperm(C)
        knn_x = knn_x.t()[new_idx].t()
        
        knn_x = knn_x.reshape(-1, self.group, C // self.group)
        knn_x = (knn_x - knn_x.mean(dim=0, keepdim=True)).transpose(0, 1)
        covs = knn_x.transpose(1, 2).bmm(knn_x) / knn_x.shape[0]
        eig, u = torch.linalg.eigh(covs, UPLO='U')
        W = u.bmm(eig.rsqrt().diag_embed()).bmm(u.transpose(1, 2))
        knn_x = knn_x.bmm(W)
        #knn_x = knn_x.transpose(0, 1).flatten(1).view(B, N, K, C)
        knn_x = knn_x.transpose(1, 2).flatten(0, 1)[torch.argsort(new_idx)].t().view(B, N, K, C)
        return knn_x

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        #self.fc_v = nn.Linear(d_model, h * d_v)
        #self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
    #    nn.init.xavier_uniform_(self.fc_v.weight)
    #    nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
    #    nn.init.constant_(self.fc_v.bias, 0)
    #    nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, mode='known'):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[0]

        # dot
        # q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        #cos
        #keys = (keys).unsqueeze(0).repeat(b_s, 1, 1)
        #q = l2_normalize(queries)  # (b_s, h, nq, d_k)
        #k = l2_normalize(keys).permute(0, 2, 1)  # (b_s, h, d_k, nk)
        #values = (values).unsqueeze(0).repeat(b_s, 1, 1)
        #att = torch.matmul(q, k) #/ np.sqrt(self.d_k)  # (b_s, h, nq, nk) #FIXME cos similarity

        # # hyperbolic
        q = queries.view(-1, self.d_model)  # (b_s, h, nq, d_k)
        k = keys.view(-1, self.d_model) # (b_s, h, d_k, nk)
        q = self.fc_q(q)
        k = self.fc_k(k)
        att = Poincare_dist(q, k)

        #att2 = att / 0.01
        att2 = att
        #_, mask_inds = att.topk(100, dim=2, largest=False)
        #att2 = att2.scatter(dim=2, index=mask_inds, value=-1e9)
        _, mask_inds = att.topk(int(nk * 0.6), dim=1, largest=False)
        att2 = att2.scatter(dim=1, index=mask_inds, value=-1e9) 
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att2 = att2.masked_fill(attention_mask, -np.inf)
        att2 = torch.softmax(att2, -1)
        out = torch.matmul(att2, values).contiguous()# (b_s, nq, h*d_v)
        # out = self.fc_o(out)  # (b_s, nq, d_model)
        att2 = att2.view(b_s, nq, -1)
        out = out.view(b_s, nq, self.d_model)
        return out, att2, None


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
                #tutel_moe.moe_layer(
                #gate_type={'type': 'cosine_top', 'k': 2, 'fp32_gate': True, 'gate_noise': 1.0, 'capacity_factor': 1.5},
                #experts={'type': 'ffn', 'count_per_node': 16, 'hidden_size_per_expert': dim * 2, 'activation_fn': lambda x: F.gelu(x)},#self.moe_drop(F.gelu(x))},
                #model_dim=dim,
                #batch_prioritized_routing=True,
                #is_gshard_loss=False,
                #)
            ]))

    def forward(self, x):
        l_aux = 0
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            #l_aux += ff.l_aux * 0.1

        return x, l_aux


class ClusteringQuantize(nn.Module):
    def __init__(self):
        super(ClusteringQuantize, self).__init__()
        self.beta = 0.25
        self.decay = 0.99
        self.num_code = 500
        self.code_prob = torch.zeros(self.num_code).cuda()

    def forward(self, z_from_encoder, codebook):
        bs, n, dim_z = z_from_encoder.shape
        z_flattened = z_from_encoder.view(-1, dim_z)
        # cosine distances from z to embeddings e_j
        normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
        normed_codebook = F.normalize(codebook, dim=1)
        d = torch.einsum('bd,dn->bn', normed_z_flattened, normed_codebook.transpose(0, 1))
        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_code, device=z_from_encoder.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        # quantise and unflatten
        z_q = torch.matmul(encodings, codebook).view(z_from_encoder.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z_from_encoder) ** 2) + torch.mean((z_q - z_from_encoder.detach()) ** 2)
        # preserve gradients
        z_q = z_from_encoder + (z_q - z_from_encoder).detach()

        avg_probs = torch.mean(encodings, dim=0)
        if self.training:
            # calculate the average usage of code entries
            self.code_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # closest sampling
            sort_distance, indices = d.sort(dim=0)
            random_feat = z_flattened.detach()[indices[-1, :]]
            # decay parameter based on the average usage
            decay = torch.exp(-(self.code_prob * self.num_code * 10) / (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, dim_z)
            codebook = codebook * (1 - decay) + random_feat * decay

            # contrastive loss
            sort_distance, indices = d.sort(dim=0)
            dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.num_code)):, :].mean(dim=0, keepdim=True)
            dis_neg = sort_distance[:int(sort_distance.size(0) * 1 / 2), :]
            dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
            contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
            loss += contra_loss

        return z_q, loss, encodings.reshape(bs, n, -1)


class Quantize(nn.Module):
    def __init__(self):
        super(Quantize, self).__init__()

    def forward(self, z_from_encoder, codebook):
        bs, n, dim_z = z_from_encoder.shape
        z_from_encoder= F.normalize(z_from_encoder, p=2, dim=-1)
        z_flat = z_from_encoder.view(-1, dim_z)
        codebook = codebook.transpose(0, 1)
        codebook_norm = F.normalize(codebook, p=2, dim=0)
        dist = (z_flat.pow(2).sum(1, keepdim=True) - 2 * z_flat @ codebook_norm + codebook_norm.pow(2).sum(0, keepdim=True))
        _, codebook_inds = (-dist).max(1)
        codebook_inds = codebook_inds.view(*z_from_encoder.shape[:-1])
        quantize = self.embed_code(codebook_inds, codebook)

        att = torch.zeros(codebook_inds.shape[0] * codebook_inds.shape[1], codebook.shape[1], device='cuda')
        att.scatter_(1, codebook_inds.reshape(codebook_inds.shape[0] * codebook_inds.shape[1], 1), 1.0)
        att = att.reshape(bs, n, -1)
        diff = 0.25 * (quantize.detach() - z_from_encoder).pow(2).mean() + (quantize - z_from_encoder.detach()).pow(2).mean()
        quantize = z_from_encoder + (quantize - z_from_encoder).detach()

        return quantize, att, None, diff

    def embed_code(self, embed_id, codebook):
        return F.embedding(embed_id, codebook.transpose(0, 1))


class GeneralizedFeatureExtractor(nn.Module):
    def __init__(self, channel=3):
        super(GeneralizedFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channel, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
        )

    def forward(self, xyz, pn=None):
        batch_size = xyz.shape[0]
        point_num = xyz.shape[2]
        feat = self.encoder(xyz)
        if pn is not None:
            index = torch.arange(point_num).reshape(1, -1).repeat(batch_size, 1).to('cuda')
            mask = (index < pn.reshape(-1, 1)).unsqueeze(1)
            feat = feat.masked_fill(torch.bitwise_not(mask).repeat(1, 1024, 1), 0)
        return feat


class GeneralizedPointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(GeneralizedPointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class SpecializedCls(nn.Module):
    def __init__(self, mode):
        super(SpecializedCls, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        #if mode == 0:
        #    self.fc3 = nn.Linear(256, 25)
        #    self.feature_dim = 25
        #else:
        #    self.fc3 = nn.Linear(256, 5)
        #    self.feature_dim = 5
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.feature_dim = 256

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        #x = self.fc3(x)
        return x


class PointNetCls(nn.Module):
    def __init__(self):
        super(PointNetCls, self).__init__()
        self.feat = GeneralizedPointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        #if mode == 0:
        #    self.fc3 = nn.Linear(256, 25)
        #    self.feature_dim = 25
        #else:
        #    self.fc3 = nn.Linear(256, 5)
        #    self.feature_dim = 5
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.feature_dim = 256
        self.out_dim = 256

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        #x = self.fc3(x)
        return x#, trans_feat


class GeneralizedPointNet2Encoder(nn.Module):
    def __init__(self, normal_channel=False):
        super(GeneralizedPointNet2Encoder, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        '''
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3 + normal_channel,
                                             [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 256])
        self.fp1 = PointNetFeaturePropagation(256, [256, 256, 256])
        '''
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        '''
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        '''
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        #l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l2_points.transpose(1, 2)

        return x


class GeneralizedDGCNNEncoder(nn.Module):
    def __init__(self, emb_dims):
        self.k = 20
        super(GeneralizedDGCNNEncoder, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))


    def forward(self, x, pn=None):
        batch_size = x.size(0)
        point_num = x.shape[2]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        if pn is not None:
            index = torch.arange(point_num).reshape(1, -1).repeat(batch_size, 1).to('cuda')
            mask = (index < pn.reshape(-1, 1)).unsqueeze(1)
            x = x.masked_fill(torch.bitwise_not(mask).repeat(1, 1024, 1), 0)
        return x


class GeneralizedPointNet2Decoder(nn.Module):
    def __init__(self, normal_channel=False):
        super(GeneralizedPointNet2Decoder, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.fp3 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128])
        self.fp2 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 64])
        self.fp1 = PointNetFeaturePropagation(in_channel=64, mlp=[64, 32])
        self.fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, points, xyz_and_feats):
        l2_points = self.fp3(xyz_and_feats[2][0], xyz_and_feats[3][0], None, points)
        l1_points = self.fp2(xyz_and_feats[1][0], xyz_and_feats[2][0], None, l2_points)
        l0_points = self.fp1(xyz_and_feats[0][0], xyz_and_feats[1][0], None, l1_points)
        return self.fc(l0_points.transpose(1, 2))


class GeneralizedPointTransformerEncoder(nn.Module):
    def __init__(self, d_points=3, nblocks=1, nneighbor=32, npoints=1024, transformer_dim=64):
        super(GeneralizedPointTransformerEncoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            #nn.ReLU(),
            #nn.Linear(32, 32), 
            #nn.ReLU(),
            #nn.Linear(64, 128)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor // 4 ** i, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor // 4 ** i))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        B, N, C = x.shape
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class GeneralizedPointTransformerDecoder(nn.Module):
    def __init__(self, d_points=3, nblocks=1, nneighbor=64, npoints=1024, transformer_dim=64):
        super(GeneralizedPointTransformerDecoder, self).__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor // 4)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor // 4 ** (nblocks - i - 1)))
        self.fc3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, points, xyz_and_feats):
        xyz = xyz_and_feats[-1][0]
        B, N, C = points.shape
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], None)
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]

        B, N, C = points.shape
        return self.fc3(points)


class GeneralizedPartConceptLearning(nn.Module):
    def __init__(self, k=8, num_points=512, emb_dim=256):
        super(GeneralizedPartConceptLearning, self).__init__()
        self.PosEm = LGA(emb_dim * 2, 1000, 1000)
        #self.PosEm = SGW()
        self.part_projection = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        #self.VQVAE = ClusteringQuantize()
        #self.decoder = GeneralizedPointTransformerDecoder()
        #self.decoder = Atlasnet(number_points=num_points, nb_primitives=k)
        self.CrossAttn = ScaledDotProductAttention(d_model=emb_dim, d_k=emb_dim, d_v=emb_dim, h=1)
        self.k = k
        self.num_points = num_points

    def forward(self, xyz_and_feats, feat, partcode, cent_xyz=None, original=False):
        #'''
        B = feat.shape[0]
        xyz = xyz_and_feats.transpose(1, 2)
        feat = feat.transpose(1, 2)
        if cent_xyz == None:
            cent_index = farthest_point_sample(xyz, self.k)
            cent_xyz = index_points(xyz, cent_index)
        id = knn2(xyz, cent_xyz, self.num_points)
        part_feat = index_points(feat, id)
        center_feat = index_points(feat, cent_index)
        
        part_xyz = index_points(xyz, id)
        part_xyz = part_xyz.transpose(1, 2).contiguous()
        part_feat_max = torch.max(part_feat, 1)[0]

        part_related = self.PosEm(cent_xyz, center_feat, part_xyz, part_feat)
        part_related = self.part_projection(part_related.reshape(part_related.shape[0] * self.k * self.num_points,-1)).reshape(B, self.k, self.num_points, -1).contiguous()
        part_related = torch.max(part_related, 2)[0]
        #'''
        
        #part_related = feat.contiguous()
        #transformed_part_feat, loss, att = self.VQVAE(part_related, partcode)
        
        transformed_part_feat, att2, part_target = self.CrossAttn(part_related, partcode, partcode)
        
        #recon_points = self.decoder(transformed_part_feat.permute(1, 0, 2), train=True)
        #recon_points = recon_points.transpose(2, 3).contiguous()
        #recon_points = recon_points.view(B, -1, 3)
        #recon_points = self.decoder(transformed_part_feat, xyz_and_feats)
        
        return transformed_part_feat, part_related, att2, cent_xyz, None
        #return transformed_part_feat, part_related, att2, cent_xyz, id
        
        #return part_related, part_related, None, cent_xyz
        #return part_related, None, None, None, att2, None, part_related, diff, None
        #return transformed_part_feat, None, None, None, att2, None, None, None, None
        #return transformed_part_feat, None, None, None, att, None, None, loss, recon_points
        #return transformed_part_feat, None, None, None, att, None, None, None, None
        #return transformed_part_feat, cent_xyz, part_feat_max, part_xyz, att2, part_target, part_related, None, None#, diff, recon_points


class SpecializedTransformer(nn.Module):
    def __init__(self, mode, dim=512, depth=2, heads=8, mlp_dim=512, dim_head=64, dropout=0., emb_dropout=0.):
        super(SpecializedTransformer, self).__init__()
        if mode == 0:
            self.prototypes = nn.Parameter(torch.zeros(25, dim), requires_grad=True)
            self.feature_dim = 25
        else:
            self.prototypes = nn.Parameter(torch.zeros(5, dim), requires_grad=True)
            self.feature_dim = 5
        torch.nn.init.xavier_normal_(self.prototypes)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.l_aux = None

        #self.projection = nn.Sequential(
        #    nn.Linear(dim, dim),
        #    nn.BatchNorm1d(dim),
        #    nn.ReLU(),
        #    nn.Linear(dim, dim),
        #    nn.BatchNorm1d(dim),
        #    nn.ReLU(),
        #    nn.Linear(dim, dim),
        #)
        #self.transformers = nn.ModuleList()
        #for i in range(3):
        #    self.transformers.append(TransformerBlock(dim, mlp_dim, 16))

    def transformer_encoder(self, xyz, feat, part_pos_emb, part_related):
        B, N, D = feat.shape
        feat_o = torch.cat([feat, part_related], dim=-1)

        #feat_m = self.projection(feat_o.reshape(B*N, -1)).reshape(B, N, -1)
        #pooled_x_max = torch.max(feat_m, 1)[0]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)
        ########feat = torch.cat([cls_tokens, feat_o], dim=1)
        feat = torch.cat([cls_tokens, feat_o + self.pos_embedding(xyz)], dim=1)
        feat, l_aux = self.transformer(feat)
        self.l_aux = l_aux

        #for i in range(3):
        #    feat = self.transformers[i](xyz, feat)[0]

        #pooled_x_max = torch.max(feat, 1)[0]
        #pooled_x_mean = torch.mean(feat, 1)
        #return pooled_x_mean
        ########x_cls = feat[:, 0]
        x_cls = feat.max(dim=1)[0]
        return x_cls #torch.cat([x_cls, pooled_x_max], dim=-1)

    def forward(self, xyz, x, part_pos_emb, part_related, biases, i, att, return_feats=False):
        feat_x = self.transformer_encoder(xyz, x, part_pos_emb, part_related)
        feat_x = l2_normalize(feat_x)
        if biases is None:
            out = torch.einsum('bc,kc->bk', feat_x, l2_normalize(self.prototypes))
        else:
            _bias = l2_normalize(self.prototypes)
            for j in range(len(biases)):
                if j >= i:
                    _bias = biases[j][i](_bias, att, bias=True)
            out = torch.einsum('bc,kc->bk', feat_x, _bias)
        if return_feats:
            if biases is not None:
                for j in range(len(biases)):
                    if j >= i:
                        feat_x = biases[j][i](feat_x, bias=True)
            return feat_x
        else:
            return out


class SpecializedMLP(nn.Module):
    def __init__(self, mode, dim=256):
        super(SpecializedMLP, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim * 2, dim),
            #nn.TransformerEncoderLayer(dim * 2, 4, dim, 0.5, batch_first=True),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            #nn.TransformerEncoderLayer(dim, 4, dim, 0.5, batch_first=True),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            #nn.TransformerEncoderLayer(dim, 4, dim, 0.5, batch_first=True),
        )
        if mode == 0:
            self.prototypes = nn.Parameter(torch.zeros(25, dim), requires_grad=True)
            self.feature_dim = 25
        else:
            self.prototypes = nn.Parameter(torch.zeros(5, dim), requires_grad=True)
            self.feature_dim = 5
        torch.nn.init.xavier_normal_(self.prototypes)
        #self.mlp = tutel_moe.moe_layer(
        #    gate_type={'type': 'cosine_top', 'k': 1, 'fp32_gate': True, 'gate_noise': 1.0, 'capacity_factor': 1.5},
        #    experts={'type': 'ffn', 'count_per_node': 4, 'hidden_size_per_expert': dim * 2, 'activation_fn': lambda x: F.relu(x)},#self.moe_drop(F.gelu(x))},
        #    model_dim=dim * 2,
        #    batch_prioritized_routing=True,
        #    is_gshard_loss=False,
        #)

    def multi_pool(self, feat, part_pos_emb, part_related):
        B, N, D = feat.shape
        feat = torch.cat([feat, part_related], dim=-1).reshape(B * N, -1)
        #feat = feat.reshape(B * N, -1)
        feat = self.projection(feat).reshape(B, N, -1)
        #feat = self.mlp(feat).reshape(B, N, -1)
        pooled_x_max = torch.max(feat, 1)[0]
        #feat = torch.mean(feat, 1)
        #feat = self.projection(feat)
        return pooled_x_max

    def forward(self, xyz, x, part_pos_emb, part_related, biases, i, att, return_feats=False):
        feat_x = self.multi_pool(x, part_pos_emb, part_related)
        feat_x = l2_normalize(feat_x)        
        if biases is None:
            out = torch.einsum('bc,kc->bk', feat_x, l2_normalize(self.prototypes))
        else:
            _bias = l2_normalize(self.prototypes)#.unsqueeze(0).expand(feat_x.shape[0], -1, -1)
            for j in range(len(biases)):
                if j >= i:
                    _bias = biases[j][i](_bias, att, bias=True)
            out = torch.einsum('bc,kc->bk', feat_x, _bias)
        if return_feats:
            #if biases is not None:
            #    for j in range(len(biases)):
            #        if j >= i:
            #            feat_x = biases[j][i](feat_x, bias=True)
            return feat_x
        else:
            return out


class SpecializedPointNetMLP(nn.Module):
    def __init__(self):
        super(SpecializedPointNetMLP, self).__init__()
        self.fc1 = nn.Linear(256, 512)
        #self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        #######self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.feature_dim = 256

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        #x = self.fc3(x)
        #x = F.log_softmax(x, dim=1)
        return x


class SpecializedPartRelationEncoder(nn.Module):
    def __init__(self, k=8, num_points=512, emb_dim=256):
        super(SpecializedPartRelationEncoder, self).__init__()
        self.PartFormer = nn.TransformerEncoderLayer(emb_dim, 4, emb_dim, 0.5, batch_first=True)
        self.feature_dim = 256
        #self.feature_dim = 1024
        self.num_points = num_points

    def forward(self, part_feat_max):
        part_relate_emb = self.PartFormer(part_feat_max)

        return part_relate_emb


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn2(x, y, k):
    inner = -2 * torch.matmul(x, y.transpose(1, 2))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=1)[1]  # (batch_size, num_points, k)
    return idx


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def distributed_sinkhorn_topk(out, sinkhorn_iterations=3, epsilon=0.03, sparsity=2):
    Q = torch.exp(out / epsilon).t() # K x B
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)+ 1e-10
        Q /= sum_of_rows
        Q /= K

        # apply top-k soft thresholding to each row
        Q = Q.t() # B,K
        topk_values, _ = torch.topk(Q, k=sparsity, dim=1)
        Q[Q < topk_values[:, [-1]]] = 0
        # Q[Q > topk_values[:, [-1]]] = 1/sparsity
        Q = Q.t()

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)+ 1e-10
        Q /= B

    Q *= B # the columns must sum to 1 so that Q is an assignment
    Q = Q.t()

    return Q


def index_points_pt(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def farthest_point_sample_pt(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample_pt(xyz, npoint) # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        #feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1# + feats2


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points_pt(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points_pt(self.w_ks(x), knn_idx), index_points_pt(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

'''
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points
'''
'''
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
'''
