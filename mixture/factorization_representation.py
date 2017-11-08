import random

import torch

import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

from torch.autograd import Variable

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


class MixtureNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 projection_scale=1.0,
                 num_components=4):

        super(MixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components
        self.projection_scale = projection_scale

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

        self.taste_projection = nn.Linear(embedding_dim,
                                          embedding_dim * self.num_components, bias=False)
        self.attention_projection = nn.Linear(embedding_dim,
                                              embedding_dim * self.num_components, bias=False)

        for layer in (self.taste_projection, self.attention_projection):
            torch.nn.init.xavier_normal(layer.weight, self.projection_scale)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        batch_size, embedding_size = item_embedding.size()

        user_tastes = (self.taste_projection(user_embedding)
                       .resize(batch_size,
                               self.num_components,
                               embedding_size))
        user_attention = (self.attention_projection(user_embedding)
                          .resize(batch_size,
                                  self.num_components,
                                  embedding_size))
        user_attention = user_attention #  * user_embedding.unsqueeze(1).expand_as(user_tastes)

        attention = (F.softmax((user_attention *
                                item_embedding.unsqueeze(1).expand_as(user_attention))
                               .sum(2)).unsqueeze(2).expand_as(user_attention))
        weighted_preference = (user_tastes * attention).sum(1)

        dot = (weighted_preference * item_embedding).sum(1)

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        # if random.random() < 0.005:
            # print('User tastes', user_tastes[0][0].abs().mean().data[0])
            # print('Tastes', weighted_preference.abs().mean().data[0])
            # assert False
        #     print('Attention', (user_attention * item_embedding).sum(2).max().data[0])
        #     print('Softmax', attention.max(1)[0].mean().data[0])
        #     print('Preference', preference.max(1)[0].mean().data[0])
        #     print('Prediction', weighted_preference.mean().data[0])
        #     print('Biases', user_bias.max().data[0], item_bias.max().data[0])

        return dot + user_bias + item_bias


class MixtureComponent(nn.Module):

    def __init__(self, embedding_dim, num_components):

        super(MixtureComponent, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        self.fc_1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.taste_projection = nn.Linear(embedding_dim,
                                          embedding_dim * num_components,
                                          bias=False)
        self.attention_projection = nn.Linear(embedding_dim,
                                              embedding_dim * num_components,
                                              bias=False)

    def forward(self, x):

        batch_size, embedding_size = x.size()

        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        user_tastes = (self.taste_pnrojection(x)
                       .resize(batch_size,
                               self.num_components,
                               embedding_size))
        user_attention = (self.attention_projection(x)
                          .resize(batch_size,
                                  self.num_components,
                                  embedding_size))

        return user_tastes, user_attention


class NonlinearMixtureNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 num_components=4):

        super(NonlinearMixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

        self.mixture = MixtureComponent(embedding_dim, num_components)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        batch_size, embedding_size = item_embedding.size()

        user_tastes, user_attention = self.mixture(user_embedding)
        item_embedding = item_embedding.unsqueeze(1).expand_as(user_attention)

        attention = F.softmax((user_attention * item_embedding).sum(2))

        preference = ((user_tastes * item_embedding)
                      .sum(2))
        weighted_preference = (attention * preference).sum(1).squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return weighted_preference + user_bias + item_bias


class BilinearNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 user_embedding_layer=None, item_embedding_layer=None, sparse=False):

        super(BilinearNet, self).__init__()

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   sparse=sparse)

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        # if random.random() < 0.01:
        #       print('Tastes', user_embedding.abs().mean().data[0])

        # if random.random() < 0.01:
        #     print('Attention', (user_attention * item_embedding).sum(2).max().data[0])
        #     print('Softmax', attention.max(1)[0].mean().data[0])
        #     print('Preference', preference.max(1)[0].mean().data[0])
        #     print('Prediction', dot.mean().data[0])
        #     print('Biases', user_bias.max().data[0], item_bias.max().data[0])

        return dot + user_bias + item_bias


class NonlinearMixtureNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 num_components=4):

        super(NonlinearMixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

        self.mixture = MixtureComponent(embedding_dim, num_components)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        batch_size, embedding_size = item_embedding.size()

        user_tastes, user_attention = self.mixture(user_embedding)
        item_embedding = item_embedding.unsqueeze(1).expand_as(user_attention)

        attention = F.softmax((user_attention * item_embedding).sum(2))

        preference = ((user_tastes * item_embedding)
                      .sum(2))
        weighted_preference = (attention * preference).sum(1).squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return weighted_preference + user_bias + item_bias


class EmbeddingMixtureNet(nn.Module):
    """
    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 num_components=4):

        super(EmbeddingMixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        self.taste_embeddings = ScaledEmbedding(num_users, embedding_dim * num_components)
        self.attention_embeddings = ScaledEmbedding(num_users, embedding_dim * num_components)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        item_embedding = self.item_embeddings(item_ids)

        batch_size, embedding_size = item_embedding.size()

        user_tastes = (self.taste_embeddings(user_ids)
                       .resize(batch_size,
                               self.num_components,
                               embedding_size))
        user_attention = (self.attention_embeddings(user_ids)
                          .resize(batch_size,
                                  self.num_components,
                                  embedding_size))

        attention = (F.softmax((user_attention *
                                item_embedding.unsqueeze(1).expand_as(user_attention))
                               .sum(2)).unsqueeze(2).expand_as(user_attention))
        weighted_preference = (user_tastes * attention).sum(1)

        dot = (weighted_preference * item_embedding).sum(1)

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return dot + user_bias + item_bias
