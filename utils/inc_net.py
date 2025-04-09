import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
from convs.partpointnet import GeneralizedPointNetEncoder, GeneralizedPartConceptLearning, SpecializedTransformer, l2_normalize


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, attn, bias=True):
        ret_x = (self.alpha + 1) * x
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
        self.TaskAgnosticPartLearner = GeneralizedPartConceptLearning(k=256, num_points=64, emb_dim=256)
        self.TaskAgnosticExtractor.train()
        self.TaskAgnosticPartLearner.train()
        self.AdaptiveExtractors = nn.ModuleList()  # Specialized Blocks
        self.part_prototype_list = nn.ParameterList()
        for i in range(6):
            if i == 0:
                part_prototypes = nn.Parameter(torch.zeros(5 * 25, 256), requires_grad=True)
                torch.nn.init.xavier_normal_(part_prototypes)
                self.part_prototype_list.append(part_prototypes)
            else:
                part_prototypes = nn.Parameter(torch.zeros(5 * 5, 256), requires_grad=True)
                torch.nn.init.xavier_normal_(part_prototypes)
                self.part_prototype_list.append(part_prototypes)
        self.out_dim1 = None
        self.out_dim2 = None
        self.task_sizes = []
        self.biases = nn.ModuleList()


    @property
    def feature_dim2(self):
        if self.out_dim2 is None:
            return 0
        return sum([learner.feature_dim for learner in self.AdaptiveExtractors])

    def extract_vector(self, x):
        B = x.shape[0]
        base_feature_map = self.TaskAgnosticExtractor(x)
        q1, part_related, att, base_x, _ = self.TaskAgnosticPartLearner(x, base_feature_map, torch.cat([self.part_prototype_list[i] for i in range(len(self.AdaptiveExtractors))], 0))
        part_score = att.sum(1)
        features = self.AdaptiveExtractors[-1](base_x, q1, None, part_related, None, len(self.AdaptiveExtractors) - 1, att)
        for i in range(len(self.AdaptiveExtractors) - 2, -1, -1):
            features = torch.cat([self.AdaptiveExtractors[i](base_x, q1, None, part_related, self.biases, i, att), features], 1)
        return features

    def forward(self, x):
        B = x.shape[0]
        base_feature_map = self.TaskAgnosticExtractor(x) # B, D, N
        q1, part_related, att, base_x, part_id = self.TaskAgnosticPartLearner(x, base_feature_map, torch.cat([self.part_prototype_list[i] for i in range(len(self.AdaptiveExtractors))], 0))
        part_score = att.sum(1)
        features = self.AdaptiveExtractors[-1](base_x, q1, None, part_related, None, len(self.AdaptiveExtractors) - 1, att)
        for i in range(len(self.AdaptiveExtractors) - 2, -1, -1):
            features = torch.cat([self.AdaptiveExtractors[i](base_x, q1, None, part_related, self.biases, i, att), features], 1)
        
        out = {'logits': features}
        out.update({'part_logits': att})
        out.update({'base_feats': part_related})
        out.update({'part_scores': part_score})
        out.update({'backbone_feats': x})
        out.update({'feats': q1})
        out.update({'part_related': part_related})
        out.update({'attns': att})
        return out


    def update_fc(self, nb_classes):
        if len(self.AdaptiveExtractors) > 0:
            _new_bias = nn.ModuleList([BiasLayer() for i in range(len(self.task_sizes))])
            self.biases.append(_new_bias)
        if len(self.AdaptiveExtractors) == 0:
            _new_extractor = SpecializedTransformer(0)
        else:
            _new_extractor = SpecializedTransformer(len(self.AdaptiveExtractors))
        if len(self.AdaptiveExtractors) == 0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)

        if self.out_dim2 is None:
            logging.info(self.AdaptiveExtractors[-1])
            self.out_dim2 = self.AdaptiveExtractors[-1].feature_dim

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

    def generate_fc(self, in_dim, out_dim):
        fc = nn.Linear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

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


