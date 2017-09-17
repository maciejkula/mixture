"""
This module contains prototypes of various ways of representing users
as functions of the items they have interacted with in the past.
"""

import numpy as np
import torch
import random

from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


PADDING_IDX = 0


def gaussian_kl_divergence(x_mu, x_sigma_sq, y_mu, y_sigma_sq):

    x_sigma = torch.sqrt(x_sigma_sq)
    y_sigma = torch.sqrt(y_sigma_sq)

    return (torch.log(y_sigma / x_sigma) +
            (x_sigma_sq + (x_mu - y_mu) ** 2) / (2 * y_sigma_sq) -
            0.5)


class GaussianLSTMNet(nn.Module):
    """
    Module representing users through running a recurrent neural network
    over the sequence, using the hidden state at each timestep as the
    sequence representation, a'la [2]_

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and aross time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of hidden
        units in the LSTM layer.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [2] Hidasi, Balazs, et al. "Session-based recommendations with
       recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
    """

    def __init__(self, num_items, embedding_dim=32,
                 item_embedding_layer=None, sparse=False):

        super(GaussianLSTMNet, self).__init__()

        self.embedding_dim = embedding_dim

        # if item_embedding_layer is not None:
        #     self.item_embeddings = item_embedding_layer
        # else:
        #     self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
        #                                            padding_idx=PADDING_IDX,
        #                                            sparse=sparse)

        self.mu_item = ScaledEmbedding(num_items, embedding_dim,
                                       padding_idx=PADDING_IDX,
                                       sparse=sparse)
        self.sigma_item = ScaledEmbedding(num_items, embedding_dim,
                                          padding_idx=PADDING_IDX,
                                          sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.mu_lstm = nn.LSTM(batch_first=True,
                               input_size=embedding_dim,
                               hidden_size=embedding_dim)
        self.sigma_lstm = nn.LSTM(batch_first=True,
                                  input_size=embedding_dim,
                                  hidden_size=embedding_dim)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """

        embedding_mu = self.mu_item(item_sequences).permute(0, 2, 1)
        embedding_sigma = self.sigma_item(item_sequences).permute(0, 2, 1)

        # if self.training:
        #     draw = torch.zeros(embedding_mu.size())
        #     draw.normal_()

        #     if embedding_mu.is_cuda:
        #         draw = draw.cuda()

        #     sequence_embeddings = Variable(draw) * torch.exp(embedding_sigma) + embedding_mu
        # else:
        #     sequence_embeddings = embedding_mu

        sequence_embeddings = embedding_mu

        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        mu, _ = self.mu_lstm(sequence_embeddings)
        sigma, _ = self.sigma_lstm(sequence_embeddings)

        mu = mu.permute(0, 2, 1)
        sigma = sigma.permute(0, 2, 1)

        if self.training:
            draw = torch.zeros(mu.size())
            draw.normal_()

            if mu.is_cuda:
                draw = draw.cuda()

            user_representations = Variable(draw) * torch.exp(sigma) + mu
        else:
            user_representations = mu

        # user_representations, _ = self.lstm(sequence_embeddings)
        # user_representations = user_representations.permute(0, 2, 1)

        return user_representations[:, :, :-1], user_representations[:, :, -1]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            A minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        embedding_mu = self.mu_item(targets).permute(0, 2, 1)
        embedding_sigma = self.sigma_item(targets).permute(0, 2, 1)

        # if self.training:
        #     draw = torch.zeros(embedding_mu.size())
        #     draw.normal_()

        #     if embedding_mu.is_cuda:
        #         draw = draw.cuda()
            
        #     target_embedding = Variable(draw) * torch.exp(embedding_sigma) + embedding_mu
        # else:
        #     target_embedding = embedding_mu

        target_embedding = embedding_mu

        # target_embedding = (self.item_embeddings(targets)
        #                     .permute(0, 2, 1)
        #                     .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding.squeeze())
               .sum(1)
               .squeeze())

        return target_bias + dot


class GaussianKLLSTMNet(nn.Module):
    """
    Module representing users through running a recurrent neural network
    over the sequence, using the hidden state at each timestep as the
    sequence representation, a'la [2]_

    During training, representations for all timesteps of the sequence are
    computed in one go. Loss functions using the outputs will therefore
    be aggregating both across the minibatch and aross time in the sequence.

    Parameters
    ----------

    num_items: int
        Number of items to be represented.
    embedding_dim: int, optional
        Embedding dimension of the embedding layer, and the number of hidden
        units in the LSTM layer.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.

    References
    ----------

    .. [2] Hidasi, Balazs, et al. "Session-based recommendations with
       recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
    """

    def __init__(self, num_items, embedding_dim=32,
                 item_embedding_layer=None, sparse=False):

        super(GaussianKLLSTMNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.mu_item = ScaledEmbedding(num_items, embedding_dim,
                                       padding_idx=PADDING_IDX,
                                       sparse=sparse)
        self.sigma_item = ScaledEmbedding(num_items, embedding_dim,
                                          padding_idx=PADDING_IDX,
                                          sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.mu_lstm = nn.LSTM(batch_first=True,
                               input_size=embedding_dim,
                               hidden_size=embedding_dim)
        self.sigma_lstm = nn.LSTM(batch_first=True,
                                  input_size=embedding_dim,
                                  hidden_size=embedding_dim)

    def user_representation(self, item_sequences):
        """
        Compute user representation from a given sequence.

        Returns
        -------

        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.
        """

        embedding_mu = self.mu_item(item_sequences).permute(0, 2, 1)
        embedding_sigma = self.sigma_item(item_sequences).permute(0, 2, 1)

        # if self.training:
        #     draw = torch.zeros(embedding_mu.size())
        #     draw.normal_()

        #     if embedding_mu.is_cuda:
        #         draw = draw.cuda()
            
        #     sequence_embeddings = Variable(draw) * torch.exp(embedding_sigma) + embedding_mu
        # else:
        #     sequence_embeddings = embedding_mu

        sequence_embeddings = embedding_mu

        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        mu, _ = self.mu_lstm(sequence_embeddings)
        sigma, _ = self.sigma_lstm(sequence_embeddings)

        mu = mu.permute(0, 2, 1)
        sigma = sigma.permute(0, 2, 1)

        ret = torch.cat([mu, sigma], 1)

        return ret[:, :, :-1], ret[:, :, -1]

    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------

        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            A minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------

        predictions: tensor
            of shape (minibatch_size, sequence_length)
        """

        embedding_mu = self.mu_item(targets).permute(0, 2, 1).squeeze()
        embedding_sigma = self.sigma_item(targets).permute(0, 2, 1).squeeze()

        target_bias = self.item_biases(targets).squeeze()

        split_idx = self.embedding_dim
        user_mu, user_sigma = (user_representations[:, :split_idx],
                               user_representations[:, split_idx:])

        kl = gaussian_kl_divergence(user_mu,
                                    torch.exp(user_sigma),
                                    embedding_mu,
                                    torch.exp(embedding_sigma))
        return (-kl.sum(1)
                .squeeze())

        dot = ((user_representations * target_embedding.squeeze())
               .sum(1)
               .squeeze())

        return target_bias + dot
