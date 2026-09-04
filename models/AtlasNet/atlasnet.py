import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from models.AtlasNet.model_blocks import Mapping2Dto3D, Identity
from models.AtlasNet.template import get_template


class Atlasnet(nn.Module):

    def __init__(self, number_points=128, nb_primitives=8):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt: 
        """
        super(Atlasnet, self).__init__()
        #self.opt = opt
        # self.device = opt.device

        self.number_points = number_points
        self.nb_primitives = nb_primitives
        # Define number of points per primitives
        self.nb_pts_in_primitive = number_points
        self.nb_pts_in_primitive_eval = number_points

        # if opt.remove_all_batchNorms:
        #     torch.nn.BatchNorm1d = Identity
        #     print("Replacing all batchnorms by identities.")

        # Initialize templates
        self.template = [get_template("SQUARE") for i in range(0, nb_primitives)]

        # Intialize deformation networks
        self.decoder = nn.ModuleList([Mapping2Dto3D() for i in range(0, nb_primitives)])

    def forward(self, latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, K, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        # Sample points in the patches
        # input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive,
        #                                                     device=latent_vector.device)
        #                 for i in range(self.opt.nb_primitives)]
        if train:
            input_points = [self.template[i].get_random_points(
                torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)), # 8 [1，3，256]
                latent_vector.device) for i in range(self.nb_primitives)]
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=latent_vector.device)
                            for i in range(self.nb_primitives)]

        # Deform each patch
        output_points = torch.cat([self.decoder[i](input_points[i], latent_vector[i].unsqueeze(2)).unsqueeze(1) for i in
                                   range(0, self.nb_primitives)], dim=1)

        # Return the deformed pointcloud
        return output_points.contiguous()  # batch, nb_prim, num_point, 3

    def generate_mesh(self, latent_vector):
        assert latent_vector.size(0)==1, "input should have batch size 1!"
        import pymesh
        input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive, latent_vector.device)
                        for i in range(self.nb_primitives)]
        input_points = [input_points[i] for i in range(self.nb_primitives)]

        # Deform each patch
        output_points = [self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).squeeze() for i in
                         range(0, self.nb_primitives)]

        output_meshes = [pymesh.form_mesh(vertices=output_points[i].transpose(1, 0).contiguous().cpu().numpy(),
                                          faces=self.template[i].mesh.faces)
                         for i in range(self.nb_primitives)]

        # Deform return the deformed pointcloud
        mesh = pymesh.merge_meshes(output_meshes)

        return mesh
