import torch

import torch.nn as nn

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


def softmax(x, dim):

    numerator = torch.exp(x)
    denominator = numerator.sum(dim, keepdim=True).expand_as(numerator)

    return numerator / denominator


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
                 num_components=4):

        super(MixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)

        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

        self.taste_projection = nn.Linear(embedding_dim,
                                          embedding_dim * self.num_components)
        self.attention_projection = nn.Linear(embedding_dim,
                                              embedding_dim * self.num_components)

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
        item_embedding = item_embedding.unsqueeze(1).expand_as(user_attention)
        attention = (softmax((user_attention * item_embedding).sum(2), 1))
        preference = ((user_tastes * item_embedding)
                      .sum(2))
        weighted_preference = (attention * preference).sum(1).squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return weighted_preference + user_bias + item_bias
