import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from spotlight.layers import ScaledEmbedding, ZeroEmbedding


PADDING_IDX = 0


def softmax(x, dim):

    numerator = torch.exp(x)
    denominator = numerator.sum(dim, keepdim=True).expand_as(numerator)

    return numerator / denominator


class LSTMNet(nn.Module):
    """
    Not clear why this is here.
    """

    def __init__(self, num_items, embedding_dim=32,
                 item_embedding_layer=None, sparse=False):

        super(LSTMNet, self).__init__()

        self.embedding_dim = embedding_dim

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
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

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)

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

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class MixtureLSTMNet(nn.Module):
    """
    LSTM with a layer on top that projects the last hidden state
    into mus (interest vectors) and categorical alphas for taste
    selection.
    """

    def __init__(self, num_items, embedding_dim=32,
                 num_components=4,
                 item_embedding_layer=None, sparse=False):

        super(MixtureLSTMNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)
        self.projection = nn.Conv1d(embedding_dim,
                                    embedding_dim * self.num_components * 2,
                                    kernel_size=1)

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

        batch_size, sequence_length = item_sequences.size()

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)
        user_representations = self.projection(user_representations)
        user_representations = user_representations.resize(batch_size,
                                                           self.num_components * 2,
                                                           self.embedding_dim,
                                                           sequence_length + 1)

        return user_representations[:, :, :, :-1], user_representations[:, :, :, -1:]

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

        user_components = user_representations[:, :self.num_components, :, :]
        mixture_vectors = user_representations[:, self.num_components:, :, :]

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze()

        mixture_weights = (mixture_vectors * target_embedding
                           .unsqueeze(1)
                           .expand_as(user_components))
        mixture_weights = (softmax(mixture_weights.sum(2), 1)
                           .unsqueeze(2)
                           .expand_as(user_components))
        weighted_user_representations = (mixture_weights * user_components).sum(1)

        dot = ((weighted_user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class Mixture2LSTMNet(nn.Module):
    """
    Version of mixture net that has a second set of embeddings for every item
    that are used together with the attention vectors from user representation
    to define the taste mixture.
    """

    def __init__(self, num_items, embedding_dim=32,
                 num_components=4,
                 item_embedding_layer=None, sparse=False):

        super(Mixture2LSTMNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)
            self.item_interest_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                            padding_idx=PADDING_IDX,
                                                            sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)
        self.projection = nn.Conv1d(embedding_dim,
                                    embedding_dim * self.num_components * 2,
                                    kernel_size=1)

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

        batch_size, sequence_length = item_sequences.size()

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)
        user_representations = self.projection(user_representations)
        user_representations = user_representations.resize(batch_size,
                                                           self.num_components * 2,
                                                           self.embedding_dim,
                                                           sequence_length + 1)

        return user_representations[:, :, :, :-1], user_representations[:, :, :, -1:]

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

        user_components = user_representations[:, :self.num_components, :, :]
        mixture_vectors = user_representations[:, self.num_components:, :, :]

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_interest_embedding = (self.item_interest_embeddings(targets)
                                     .permute(0, 2, 1))

        target_bias = self.item_biases(targets).squeeze()

        mixture_weights = (mixture_vectors * target_interest_embedding
                           .unsqueeze(1)
                           .expand_as(user_components))
        mixture_weights = (softmax(mixture_weights.sum(2), 1)
                           .unsqueeze(2)
                           .expand_as(user_components))
        weighted_user_representations = (mixture_weights * user_components).sum(1)

        dot = ((weighted_user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class LinearMixtureLSTMNet(nn.Module):
    """
    Average of different tastes; not really a mixture model.

    For ablation analysis.
    """

    def __init__(self, num_items, embedding_dim=32,
                 num_components=4,
                 item_embedding_layer=None, sparse=False):

        super(LinearMixtureLSTMNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)
        self.projection = nn.Conv1d(embedding_dim,
                                    embedding_dim * self.num_components,
                                    kernel_size=1)

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

        batch_size, sequence_length = item_sequences.size()

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)
        user_representations = self.projection(user_representations)
        user_representations = user_representations.resize(batch_size,
                                                           self.num_components,
                                                           self.embedding_dim,
                                                           sequence_length + 1)

        return user_representations[:, :, :, :-1], user_representations[:, :, :, -1:]

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

        user_components = user_representations[:, :, :, :]

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze()

        weighted_user_representations = user_components.mean(1)

        dot = ((weighted_user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class DiversifiedMixtureLSTMNet(nn.Module):
    """
    Like MixtureNet, but with a self-similarity penalty on user tastes.
    """

    def __init__(self, num_items, embedding_dim=32,
                 num_components=4,
                 diversity_penalty=1.0,
                 item_embedding_layer=None, sparse=False):

        super(DiversifiedMixtureLSTMNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components
        self._diversity_penalty = diversity_penalty

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)
        self.projection = nn.Conv1d(embedding_dim,
                                    embedding_dim * self.num_components * 2,
                                    kernel_size=1)

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

        batch_size, sequence_length = item_sequences.size()

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)
        user_representations = self.projection(user_representations)
        user_representations = user_representations.resize(batch_size,
                                                           self.num_components * 2,
                                                           self.embedding_dim,
                                                           sequence_length + 1)

        return user_representations[:, :, :, :-1], user_representations[:, :, :, -1:]

    def diversity_penalty(self, representations):

        representations = representations[:, :self.num_components, :, :]
        batch_size, num_components, dim, seq_len = representations.size()

        # Unroll the sequence into the minibatch dimension
        representations = representations.permute(0, 3, 1, 2)
        representations = representations.resize(batch_size * seq_len,
                                                 num_components,
                                                 dim)

        # Normalize: we want to penalize similarity but not norm
        norm = torch.norm(representations, p=2, dim=2).unsqueeze(2)
        representations = representations / norm.expand_as(representations)

        # Do AA' matrix multiply. The result will be a square matrix
        # with 1s on the diagonal, off-diagonal entries express the
        # correlation of tastes.
        taste_similarity = torch.bmm(representations,
                                     representations.permute(0, 2, 1))

        # Let's remove the diagonal
        identity = Variable(torch.eye(num_components).repeat(
            batch_size * seq_len, 1, 1))

        if taste_similarity.is_cuda:
            identity = identity.cuda()

        taste_similarity = taste_similarity - identity

        # Want want to penalize both positive and negative correlations,
        # and promote orthogonality.
        taste_similarity = taste_similarity ** 2

        return self._diversity_penalty * taste_similarity.sum()

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

        user_components = user_representations[:, :self.num_components, :, :]
        mixture_vectors = user_representations[:, self.num_components:, :, :]

        self.diversity_penalty(user_components)

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze()

        mixture_weights = (mixture_vectors * target_embedding
                           .unsqueeze(1)
                           .expand_as(user_components))
        mixture_weights = (softmax(mixture_weights.sum(2), 1)
                           .unsqueeze(2)
                           .expand_as(user_components))
        weighted_user_representations = (mixture_weights * user_components).sum(1)

        dot = ((weighted_user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot
