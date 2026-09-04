import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
from convs.partpointnet_VQ_VAE import GeneralizedFeatureExtractor, GeneralizedPartConceptLearning, SpecializedMLP, l2_normalize
from convs.partpointnet_VQ_VAE import GeneralizedPointNetEncoder, GeneralizedPointNet2Encoder, GeneralizedFeatureExtractor, GeneralizedPartConceptLearning, SpecializedTransformer, SpecializedMLP, SpecializedPointNetMLP, SpecializedPartRelationEncoder, l2_normalize
from convs.partpointnet_VQ_VAE import GeneralizedPointTransformerEncoder, GeneralizedDGCNNEncoder
from convs.vqvae import VQ_VAE, AE
from convs.atlasnet import Atlas
#from models_open.backbone.pointnext import PointNextEncoder
#from convs.partpointnet import GeneralizedPointTransformerEncoder


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        #self.alpha = nn.Linear(125, 1)
        #self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, attn, bias=True):
        #attn = attn.argmax(-1)
        #attn_weight = torch.zeros((attn.shape[0], 125)).cuda()
        #for i in range(attn.shape[0]):
        #    unique_indices, counts = attn[i].unique(return_counts=True)
        #    attn_weight[i, unique_indices] = counts.float() / attn.shape[1]
        #attn_alpha = self.alpha(attn_weight).unsqueeze(-1)
        #ret_x = (attn_alpha + 1) * x
        ret_x = (self.alpha+1) * x
        #if bias:
        #    ret_x = ret_x + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class PointNet(nn.Module):
    def __init__(self, init_cls=25, normal_channel=True):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.TaskAgnosticExtractor = GeneralizedPointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        #self.TaskAgnosticExtractor = GeneralizedFeatureExtractor(channel=channel)
        #self.TaskAgnosticExtractor = GeneralizedPointNet2Encoder()
        #self.TaskAgnosticExtractor = PointNextEncoder()
        #self.TaskAgnosticExtractor = GeneralizedDGCNNEncoder(emb_dims=256)
        #self.TaskAgnosticExtractor = GeneralizedPointTransformerEncoder()
        self.TaskAgnosticPartLearner = GeneralizedPartConceptLearning(k=256, num_points=64, emb_dim=256)
        #self.TaskAgnosticPartLearner = GeneralizedPartConceptLearning(k=32, num_points=8, emb_dim=256)
        #self.AdaptivePartLearners = nn.ModuleList()
        self.AdaptiveLearnerSelectors = nn.ModuleList()
        self.TaskAgnosticExtractor.train()
        self.TaskAgnosticPartLearner.train()
        #self.AdaptiveRelationLearners = nn.ModuleList()
        self.AdaptiveExtractors = nn.ModuleList()  # Specialized Blocks
        #part_gen = part_generation(5 * init_cls, 256)
        #self.part_prototypes = nn.Parameter(torch.Tensor(part_gen), requires_grad=True)
        #####self.part_prototypes = nn.Parameter(torch.zeros(5 * 25, 256), requires_grad=True)
        #####torch.nn.init.xavier_normal_(self.part_prototypes)
        self.part_prototype_list = nn.ParameterList()
        for i in range(6):
            if i == 0:
                part_prototypes = nn.Parameter(torch.zeros(1 * 25, 256), requires_grad=True)
                torch.nn.init.xavier_normal_(part_prototypes)
                self.part_prototype_list.append(part_prototypes)
            else:
                part_prototypes = nn.Parameter(torch.zeros(1 * 5, 256), requires_grad=True)
                torch.nn.init.xavier_normal_(part_prototypes)
                self.part_prototype_list.append(part_prototypes)
        #self.part_prototypes = []
        self.out_dim1 = None
        self.out_dim2 = None
        # self.fc1 = None
        # self.fc2 = None
        #self.aux_fc = None
        self.task_sizes = []
        self.biases = nn.ModuleList()

    #@property
    #def feature_dim1(self):
    #    if self.out_dim1 is None:
    #        return 0
    #    return self.out_dim1 * len(self.AdaptiveRelationLearners)

    @property
    def feature_dim2(self):
        if self.out_dim2 is None:
            return 0
        return sum([learner.feature_dim for learner in self.AdaptiveExtractors])

    def extract_vector(self, x):
        B = x.shape[0]
        base_feature_map = self.TaskAgnosticExtractor(x)
        #base_feature_map, base_x = self.TaskAgnosticExtractor(x.transpose(1, 2))
        q1, part_related, att, base_x, _ = self.TaskAgnosticPartLearner(x, base_feature_map, torch.cat([self.part_prototype_list[i] for i in range(len(self.AdaptiveExtractors))], 0))
        #part_xyz = part_xyz.view(B, -1, 3)
        part_score = att.sum(1)
        '''
        if self.training:
            task_id = 0 #len(self.AdaptiveExtractors) - 1
            q1, part_related, att, base_x, part_id = self.AdaptivePartLearners[task_id](x, base_feature_map, self.part_prototypes)
            part_score = att.sum(1)
        else:
            recon_feats = x # base_feature_map
            base_x_list, part_related_list, q1_list, att_list, router_list = [], [], [], [], []
            for i in range(len(self.AdaptiveExtractors)):
                outputs = self.AdaptiveLearnerSelectors[i](recon_feats)
                outputs = ((recon_feats - outputs) ** 2).sum(-1).sum(-1)
                router_list.append(outputs.unsqueeze(0))
            router_scores = torch.cat(router_list, dim=0).transpose(0, 1)
            #for i in range(B):
            #    task_id = router_scores[i].argmin()
            #    q1, part_related, att, base_x, part_id = self.AdaptivePartLearners[task_id](x[i].unsqueeze(0), base_feature_map[i].unsqueeze(0), self.part_prototypes)
            #    part_related_list.append(part_related)
            #    q1_list.append(q1)
            #    att_list.append(att)
            #part_related = torch.cat(part_related_list, 0)
            #q1 = torch.cat(q1_list, 0)
            #part_score = torch.cat(att_list, 0).sum(1)
            task_id = 0 #router_scores.transpose(0, 1).sum(-1).argmin()
            q1, part_related, att, base_x, part_id = self.AdaptivePartLearners[task_id](x, base_feature_map, self.part_prototypes)
            part_score = att.sum(1)
        '''
        features = self.AdaptiveExtractors[-1](base_x, q1, None, part_related, None, len(self.AdaptiveExtractors) - 1, att)
        for i in range(len(self.AdaptiveExtractors) - 2, -1, -1):
            features = torch.cat([self.AdaptiveExtractors[i](base_x, q1, None, part_related, self.biases, i, att), features], 1)
        return features

    def forward(self, x):
        B = x.shape[0]
        base_feature_map = self.TaskAgnosticExtractor(x) # B, D, N
        #base_feature_map, base_x = self.TaskAgnosticExtractor(x.transpose(1, 2))
        q1, part_related, att, base_x, part_id = self.TaskAgnosticPartLearner(x, base_feature_map, torch.cat([self.part_prototype_list[i] for i in range(len(self.AdaptiveExtractors))], 0))
        #part_xyz = part_xyz.view(B, -1, 3)
        part_score = att.sum(1)
        '''
        if self.training:
            task_id = 0 #len(self.AdaptiveExtractors) - 1
            q1, part_related, att, base_x, part_id = self.AdaptivePartLearners[task_id](x, base_feature_map, self.part_prototypes)
            part_score = att.sum(1)
        else:
            recon_feats = x # base_feature_map
            base_x_list, part_related_list, q1_list, att_list, router_list = [], [], [], [], []
            for i in range(len(self.AdaptiveExtractors)):
                outputs = self.AdaptiveLearnerSelectors[i](recon_feats)
                outputs = ((recon_feats - outputs) ** 2).sum(-1).sum(-1)
                router_list.append(outputs.unsqueeze(0))
            router_scores = torch.cat(router_list, dim=0).transpose(0, 1)
            #for i in range(B):
            #    task_id = router_scores[i].argmin()
            #    q1, part_related, att, base_x, part_id = self.AdaptivePartLearners[task_id](x[i].unsqueeze(0), base_feature_map[i].unsqueeze(0), self.part_prototypes)
            #    part_related_list.append(part_related)
            #    q1_list.append(q1)
            #    att_list.append(att)
            #part_related = torch.cat(part_related_list, 0)
            #q1 = torch.cat(q1_list, 0)
            #part_score = torch.cat(att_list, 0).sum(1)
            task_id = 0 #router_scores.transpose(0, 1).sum(-1).argmin()
            q1, part_related, att, base_x, part_id = self.AdaptivePartLearners[task_id](x, base_feature_map, self.part_prototypes)
            part_score = att.sum(1)
        '''
        features = self.AdaptiveExtractors[-1](base_x, q1, None, part_related, None, len(self.AdaptiveExtractors) - 1, att)
        for i in range(len(self.AdaptiveExtractors) - 2, -1, -1):
            features = torch.cat([self.AdaptiveExtractors[i](base_x, q1, None, part_related, self.biases, i, att), features], 1)

        #before_features = self.AdaptiveExtractors[-1](base_x, q1, None, part_related, None, len(self.AdaptiveExtractors) - 1)
        #for i in range(len(self.AdaptiveExtractors) - 2, -1, -1):
        #    before_features = torch.cat([self.AdaptiveExtractors[i](base_x, q1, None, part_related, None, i), before_features], 1)
        
        out = {'logits': features}
        out.update({'part_logits': att})
        out.update({'base_feats': part_related})
        out.update({'part_scores': part_score})
        out.update({'backbone_feats': x})
        out.update({'feats': q1})
        out.update({'part_related': part_related})
        out.update({'attns': att})
        #out.update({'part_xyz': part_id})
        #out.update({'xyz': x.transpose(1, 2)})
        #out.update({'before_logits': before_features})
        #out.update({'part_prototypes': l2_normalize(self.part_prototypes)})
        #out.update({'part_targets': part_target})
        #out.update({'part_diff': diff})
        #out.update({'part_xyz': x.transpose(1, 2)})
        #out.update({'recon_points': recon_points})
        #out.update({'feats': feats_q1})
        #out.update({'aux_logits': aux_logits})
        return out

        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

    def update_fc(self, nb_classes):
        if len(self.AdaptiveExtractors) > 0:
            _new_bias = nn.ModuleList([BiasLayer() for i in range(len(self.task_sizes))])
            self.biases.append(_new_bias)
        if len(self.AdaptiveExtractors) == 0:
            #_new_extractor = SpecializedMLP(0)
            _new_extractor = SpecializedTransformer(0)
        else:
            #_new_extractor = SpecializedMLP(len(self.AdaptiveExtractors))
            _new_extractor = SpecializedTransformer(len(self.AdaptiveExtractors))
        #if len(self.part_prototypes) == 0:
        #    _new_part_prototypes = nn.Parameter(torch.zeros(5 * 25, 256).cuda(), requires_grad=True)
        #    torch.nn.init.xavier_normal_(_new_part_prototypes)
        #else:
        #    _new_part_prototypes = nn.Parameter(torch.zeros(1 * 5, 256).cuda(), requires_grad=True)
        #    torch.nn.init.xavier_normal_(_new_part_prototypes)
        if len(self.AdaptiveExtractors) == 0:
            self.AdaptiveExtractors.append(_new_extractor)
        #    self.part_prototypes.append(_new_part_prototypes)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
        #    self.part_prototypes.append(_new_part_prototypes)

        #PartLearner = GeneralizedPartConceptLearning(k=256, num_points=64, emb_dim=256).train()
        #self.AdaptivePartLearners.append(PartLearner)
        #LearnerSelector = AE().train()
        LearnerSelector = Atlas().train()
        self.AdaptiveLearnerSelectors.append(LearnerSelector)

        if self.out_dim2 is None:
            logging.info(self.AdaptiveExtractors[-1])
            self.out_dim2 = self.AdaptiveExtractors[-1].feature_dim
        # fc1 = self.generate_fc(self.feature_dim1, 256)
        # fc2 = self.generate_fc(self.feature_dim2, nb_classes)
        #if self.fc1 is not None:
        #    nb_output = self.fc1.out_features
        #    weight = copy.deepcopy(self.fc1.weight.data)
        #    bias = copy.deepcopy(self.fc1.bias.data)
        #    fc1.weight.data[:nb_output, :self.feature_dim1 - self.out_dim1] = weight
        #    fc1.bias.data[:nb_output] = bias
        #if self.fc2 is not None:
        #    nb_output = self.fc2.out_features
        #    weight = copy.deepcopy(self.fc2.weight.data)
        #    bias = copy.deepcopy(self.fc2.bias.data)
        #    fc2.weight.data[:nb_output, :self.feature_dim2 - self.out_dim2] = weight
        #    fc2.bias.data[:nb_output] = bias

        #del self.fc1
        #self.fc1 = fc1
        #del self.fc2
        #self.fc2 = fc2

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        #self.aux_fc = self.generate_fc(self.out_dim2, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = nn.Linear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        a=1
        #weights1 = self.fc1.weight.data
        #newnorm1 = (torch.norm(weights1[-increment:, :], p=2, dim=1))
        #oldnorm1 = (torch.norm(weights1[:-increment, :], p=2, dim=1))
        #meannew1 = torch.mean(newnorm1)
        #meanold1 = torch.mean(oldnorm1)
        #gamma1 = meanold1 / meannew1
        #print('alignweights,gamma1=', gamma1)
        #self.fc1.weight.data[-increment:, :] *= gamma1

        #weights2 = self.fc2.weight.data
        #newnorm2 = (torch.norm(weights2[-increment:, :], p=2, dim=1))
        #oldnorm2 = (torch.norm(weights2[:-increment, :], p=2, dim=1))
        #meannew2 = torch.mean(newnorm2)
        #meanold2 = torch.mean(oldnorm2)
        #gamma2 = meanold2 / meannew2
        #print('alignweights,gamma2=', gamma2)
        #self.fc2.weight.data[-increment:, :] *= gamma2

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['convnet']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k: v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k: v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class uniform_loss(nn.Module):
    def __init__(self, t=0.07):
        super(uniform_loss, self).__init__()
        self.t = t

    def forward(self, x):
        return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()


def part_generation(num_part, emd_size, N_iter=1000):
    #print("N =", num_part)
    #print("M =", emd_size)
    criterion = uniform_loss()
    x = Variable(torch.randn(num_part, emd_size).float(), requires_grad=True)
    optimizer = optim.Adam([x], lr=1e-1)
    min_loss = 100
    optimal_target = None
    for i in range(N_iter):
        optimizer.zero_grad()
        x_norm = F.normalize(x, dim=1)
        loss = criterion(x_norm)
        if i % 10 == 0:
            print(i, loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            optimal_target = x_norm
        loss.backward()
        optimizer.step()

    #np.save('models/optimal_{}_{}.npy'.format(num_part, emd_size), optimal_target.detach().numpy())

    #print("optimal loss = ", criterion(optimal_target).item())
    return optimal_target.detach()

