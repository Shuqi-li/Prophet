from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn

from ...utils.torch_utils import generate_fully_connected

from  munch import DefaultMunch
import torch.nn.functional as F
import torch.fft
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.Conv_Blocks import Inception_Block_V1
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import ProbAttention, AttentionLayer
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import (
    Encoder as AutoEncoder, 
    Decoder as AutoDecoder, 
    EncoderLayer  as AutoEncoderLayer, 
    DecoderLayer as AutoDecoderLayer, 
    my_Layernorm, 
    series_decomp)
from .layers.ETSformer_EncDec import (
    EncoderLayer as ETSEncoderLayer, 
    Encoder as ETSEncoder, 
    DecoderLayer as ETSDecoderLayer, 
    Decoder as ETSDecoder, 
    Transform
)


class TemporalFGNNIwithESTformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self,
        group_mask: torch.Tensor,
        lag: int,
        configs,
        pre_len:int=1):

        super().__init__()
        self.seq_len = lag
        self.pred_len = pre_len
        self.num_nodes, _ = group_mask.shape
        configs['pred_len'] = pre_len
        configs['seq_len'] = lag
        self.label_len = int(lag/2)
        configs['label_len'] = int(self.label_len)
        configs['enc_in'] = self.num_nodes
        configs['dec_in'] = self.num_nodes
        configs['c_out'] = self.num_nodes
        configs = DefaultMunch.fromDict(configs)
        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = ETSEncoder(
            [
                ETSEncoderLayer(
                    configs.d_model, configs.n_heads, configs.enc_in, configs.seq_len, self.pred_len, configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )
        # Decoder
        self.decoder = ETSDecoder(
            [
                ETSDecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, self.pred_len,
                    dropout=configs.dropout,
                ) for _ in range(configs.d_layers)
            ],
        )
        self.transform = Transform(sigma=0.2)

    def predict(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        if len(X.shape) == 2:
            X = X.unsqueeze(0)  # [1, lag, processed_dim_all]
        W_total = W_adj.sum(-1) # batch lag nodes
        W_total = torch.where( W_total > 0, 1, 0)
        X = X*W_total
        return self.feed_forward(X)



    def forecast(self, x_enc, x_dec, x_mark_enc=None, x_mark_dec=None):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def feed_forward(self, x_enc):
        
        batch, lag, nodes = x_enc.shape
        x_dec_ = torch.zeros(batch, self.pred_len, nodes).to(x_enc.device)
        x_dec = torch.cat([x_enc[:, -self.label_len:], x_dec_],dim=1) 
        dec_out = self.forecast(x_enc, x_dec)
        return dec_out[:, -self.pred_len:, :]






class TemporalFGNNIwithAutoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self,
        group_mask: torch.Tensor,
        lag: int,
        configs,
        pre_len:int=1):

        super().__init__()
        self.seq_len = lag
        self.pred_len = pre_len
        self.num_nodes, _ = group_mask.shape
        configs['pred_len'] = pre_len
        configs['seq_len'] = lag
        self.label_len = int(lag/2)
        configs['label_len'] = int(self.label_len)
        configs['enc_in'] = self.num_nodes
        configs['dec_in'] = self.num_nodes
        configs['c_out'] = self.num_nodes
        configs = DefaultMunch.fromDict(configs)


        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = AutoEncoder(
            [
                AutoEncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.decoder = AutoDecoder(
            [
                AutoDecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def predict(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        if len(X.shape) == 2:
            X = X.unsqueeze(0)  # [1, lag, processed_dim_all]
        W_total = W_adj.sum(-1) # batch lag nodes
        W_total = torch.where( W_total > 0, 1, 0)
        X = X*W_total
        return self.feed_forward(X)


    def forecast(self, x_enc,x_dec, x_mark_enc=None, x_mark_dec=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init,  x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def feed_forward(self, x_enc):
        
        batch, lag, nodes = x_enc.shape
        x_dec_ = torch.zeros(batch, self.pred_len, nodes).to(x_enc.device)
        x_dec = torch.cat([x_enc[:, -self.label_len:], x_dec_],dim=1) 
        dec_out = self.forecast(x_enc, x_dec)
        return dec_out[:, -self.pred_len:, :]




class ContractiveInvertibleGNN(nn.Module):
    """
    Given x, we can easily compute the exog noise z that generates it.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        device: torch.device,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        embedding_size: Optional[int] = None,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: Device used.
        """
        super().__init__()
        self.group_mask = group_mask.to(device)
        self.num_nodes, self.processed_dim_all = group_mask.shape
        self.device = device
        self.W = self._initialize_W()
        self.f = FGNNI(
            self.group_mask,
            self.device,
            norm_layer=norm_layer,
            res_connection=res_connection,
            layers_g=encoder_layer_sizes,
            layers_f=decoder_layer_sizes,
            embedding_size=embedding_size,
        )

    def _initialize_W(self) -> torch.Tensor:
        """
        Creates and initializes the weight matrix for adjacency.

        Returns:
            Matrix of size (num_nodes, num_nodes) initialized with zeros.

        Question: Initialize to zeros??
        """
        W = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        return nn.Parameter(W, requires_grad=True)

    def get_weighted_adjacency(self) -> torch.Tensor:
        """
        Returns the weights of the adjacency matrix.
        """
        W_adj = self.W * (1.0 - torch.eye(self.num_nodes, device=self.device))  # Shape (num_nodes, num_nodes)
        return W_adj

    def predict(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        Gives the prediction of each variable given its parents.

        Args:
            X: Output of the GNN after reaching a fixed point, batched. Array of size (batch_size, processed_dim_all).
            W_adj: Weighted adjacency matrix, possibly normalized.

        Returns:
            predict: Predictions, batched, of size (B, n) that reconstructs X using the SEM.
        """
        return self.f.feed_forward(X, W_adj)  # Shape (batch_size, processed_dim_all)

    def simulate_SEM(
        self,
        Z: torch.Tensor,
        W_adj: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
        gumbel_max_regions: Optional[List[List[int]]] = None,
        gt_zero_region: Optional[List[int]] = None,
    ):
        """
        Given exogenous noise Z, computes the corresponding set of observations X, subject to an optional intervention
        generates it.
        For discrete variables, the exogeneous noise uses the Gumbel max trick to generate samples.

        Args:
            Z: Exogenous noise vector, batched, of size (B, n)
            W_adj: Weighted adjacency matrix, possibly normalized. (n, n) if a single matrix should be used for all batch elements. Otherwise (B, n, n)
            intervention_mask: torch.Tensor of shape (num_nodes) optional array containing binary flag of nodes that have been intervened.
            intervention_values: torch.Tensor of shape (processed_dim_all) optional array containing values for variables that have been intervened.
            gumbel_max_regions: a list of index lists `a` such that each subarray X[a] represents a one-hot encoded discrete random variable that must be
                sampled by applying the max operator.
            gt_zero_region: a list of indices such that X[a] should be thresholded to equal 1, if positive, 0 if negative. This is used to sample
                binary random variables. This also uses the Gumbel max trick implicitly

        Returns:
             X: Output of the GNN after reaching a fixed point, batched. Array of size (batch_size, processed_dim_all).
        """

        X = torch.zeros_like(Z)

        for _ in range(self.num_nodes):
            if intervention_mask is not None and intervention_values is not None:
                X[:, intervention_mask] = intervention_values.unsqueeze(0)
            X = self.f.feed_forward(X, W_adj) + Z
            if gumbel_max_regions is not None:
                for region in gumbel_max_regions:
                    maxes = X[:, region].max(-1, keepdim=True)[0]
                    X[:, region] = (X[:, region] >= maxes).float()
            if gt_zero_region is not None:
                X[:, gt_zero_region] = (X[:, gt_zero_region] > 0).float()

        if intervention_mask is not None and intervention_values is not None:
            if intervention_values.shape == X.shape:
                X[:, intervention_mask] = intervention_values
            else:
                X[:, intervention_mask] = intervention_values.unsqueeze(0)
        return X


class TemporalContractiveInvertibleGNN(nn.Module):
    """
    This class implements the temporal version of the Contractive Invertible GNN, which supports the temporal adjacency matrix with shape [lag+1, node, node]
    or batched version [N_batch, lag+1, num_node, num_node].
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        lag: int,
        device: torch.device,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        embedding_size: Optional[int] = None,
        pre_len: int = 1
    ):
        """
        Init method for TemporalContractiveInvertibleGNN.
        Args:
            group_mask: A mask of shape [num_nodes, proc_dims] such that group_mask[i, j] = 1 when col j is in group i.
            lag: The model specified lag
            norm_layer: Normalization layer to use.
            res_connection: Whether to use residual connection
            encoder_layer_sizes: List of layer sizes for the encoder.
            decoder_layer_sizes: List of layer sizes for the decoder.
            embedding_size: The size of embeddings in Temporal FGNNI.
        """
        super().__init__()
        self.group_mask = group_mask.to(device)
        self.num_nodes, self.processed_dim_all = group_mask.shape
        self.device = device
        assert lag > 0, "Lag must be greater than 0"
        self.lag = lag
        self.embedding_size = embedding_size
        self.pre_len = pre_len

        # Initialize the associated weights by calling self.W = self._initialize_W(). self.W has shape [lag+1, num_nodes, num_nodes]
        self.W = self._initialize_W()
        # Initialize the network self.f, which is an instance of TemporalFGNNI.
        self.f = TemporalFGNNI(
            self.group_mask,
            self.device,
            self.lag,
            norm_layer=norm_layer,
            res_connection=res_connection,
            layers_g=encoder_layer_sizes,
            layers_f=decoder_layer_sizes,
            embedding_size=self.embedding_size,
            pre_len = self.pre_len
        )

    def _initialize_W(self) -> torch.Tensor:
        """
        Initializes the associated weight with shape [lag+1, num_nodes, num_nodes]. Currently, initialize to zero.
        Returns: the initialized weight with shape [lag+1, num_nodes, num_nodes]
        """
        W = torch.zeros(self.lag, self.num_nodes, self.num_nodes, device=self.device)
        return nn.Parameter(W, requires_grad=True)

    def get_weighted_adjacency(self) -> torch.Tensor:
        """
        This function returns the weights for the temporal adjacency matrix. Note that we need to disable the diagonal elements
        corresponding to the instantaneous adj matrix (W[0,...]), and keep the full matrix for the lagged adj matrix (W[1,...]).
        Returns:
            Weight with shape [lag, num_nodes, num_nodes]
        """
        return self.W

    def predict(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        This function gives the predicts of each variable based on its parents (both instantaneous and lagged parents)
        specified by weighted adjacency matrix W_adj. The functionality is similar to the one in ContractiveInvertibleGNN,
        but with the support of temporal data X and temporal weighted adjacency matrix W_adj.
        Args:
            X: The data tensor with shape [lag+1, proc_dim] or [batch_size, lag+1, proc_dims].
            W_adj: The weighted adjacency matrix with shape [lag+1, num_nodes, num_nodes] or [N_batch, lag+1, num_nodes, num_nodes].
        Returns:
            The prediction with shape [proc_dims] or [N_batch, proc_dims].
        """
        # Directly calling the feed_forward of TemporalFGNNI.
        # Requirement: self.f.feed_backward(X, W_adj). If W_adj has shape [lag+1, num_nodes, num_nodes], then it is applied for all batches
        # in X. If W_adj has shape [N_batch, lag+1, num_nodes, num_nodes], then each W_adj[i, ...] is applied for X[i, ...].
        W_adj = W_adj*self.get_weighted_adjacency().unsqueeze(0)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)  # [1, lag, processed_dim_all]
        return self.f.feed_forward(X, W_adj).squeeze(0).transpose(-1,-2)  # batch  pre_len nodes 

    # #change
    # def simulate_SEM_conditional(
    #     self,
    #     conditional_dist: nn.Module,
    #     Z: torch.Tensor,
    #     W_adj: torch.Tensor,
    #     X_history: torch.Tensor,
    #     gumbel_max_regions: Optional[List[List[int]]] = None,
    #     gt_zero_region: Optional[List[List[int]]] = None,
    #     intervention_mask: Optional[torch.Tensor] = None,
    #     intervention_values: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     This simulates the SEM with history dependent noise. This is achieved in a similar logic as simulate_SEM. The difference is
    #     that at each time step, we need to generate history dependent noise from the based noise Z for all cts nodes at that time step.
    #     This is achieved by calling conditional_dist.transform_noise(...).
    #     Args:
    #         conditional_dist: nn.Module, the conditional distributions used for sampling noise.
    #         Z: the noise from the base distribution with shape [batch_size, time_span, proc_dim_all]
    #         W_adj: Weighted adjacency matrix with shape [lag+1, num_node, num_node]. Note that this cannot be batched.
    #         X_history: The history observations with shape [batch, lag, proc_dim_all]
    #         gumbel_max_regions: a list of index lists `a` such that each subarray X[a] represents a one-hot encoded discrete random variable that must be
    #             sampled by applying the max operator.
    #         gt_zero_region: a list of indices such that X[a] should be thresholded to equal 1, if positive, 0 if negative. This is used to sample
    #             binary random variables. This also uses the Gumbel max trick implicitly
    #         intervention_mask: a binary mask of shape [max_intervention_ahead_time, proc_dim_all] that indicates which variables are intervened on.
    #         intervention_values: a tensor of shape [total_intervention_dim] that contains the intervened values.


    #     Returns:
    #         The history+simulated observations with shape [batch, history_length+time_span, processed_dim_all].
    #         The simulated observations with shape [batch, time_span, processed_dim_all].
    #     """
    #     # Assert the input noise should have shape [batch_size, time_span, proc_dim_all]
    #     assert len(Z.shape) == 3
    #     time_span = Z.shape[1]
    #     assert time_span > 0, "The time span must be >0"
    #     assert X_history.shape[0] == Z.shape[0], "The batch size of the history must match input noise batch"
    #     assert (
    #         W_adj.dim() == 3
    #     ), "The weighted adjacency matrix must have shape [lag+1, num_nodes, num_nodes], it should not be batched."
    #     # Create batched W_adj_batch
    #     W_adj_batch = W_adj.expand(Z.shape[0], -1, -1, -1)  # [batch, lag+1, num_nodes, num_nodes]
    #     # get continuous node index
    #     cts_node = conditional_dist.cts_node

    #     # Create tensor to store the simulated observations
    #     X_all = torch.cat(
    #         [X_history, torch.zeros_like(Z)], dim=1
    #     )  # shape [batch_size, time_span+history_length, processed_dim_all]
    #     size_history = X_history.shape[1]

    #     if intervention_mask is not None and intervention_values is not None:
    #         assert (
    #             Z.shape[1] >= intervention_mask.shape[0]
    #         ), "The future ahead time for observation generation must be >= the ahead time for intervention"
    #         # Convert the time_length in intervention mask to be compatible with X_all
    #         false_matrix_conditioning = torch.full(X_history.shape[1:], False, dtype=torch.bool, device=self.device)
    #         false_matrix_future = torch.full(
    #             (Z.shape[1] - intervention_mask.shape[0], Z.shape[2]), False, dtype=torch.bool, device=self.device
    #         )
    #         intervention_mask = torch.cat(
    #             (false_matrix_conditioning, intervention_mask, false_matrix_future), dim=0
    #         )  # shape [history_length+ time_span, processed_dim_all]

    #     for time in range(time_span):
    #         # Loop over num_nodes to propagate the instantaneous effect
    #         history_start_idx = size_history + time - self.lag
    #         inst_end_idx = size_history + time + 1
    #         # conditional noise for cts node.

    #         Z[:, time, cts_node] = conditional_dist.transform_noise(  # type:ignore
    #             Z=Z[:, time, cts_node], X_history=X_all[:, history_start_idx : inst_end_idx - 1, :], W=W_adj
    #         )  # shape [batch_size, cts_dim]
    #             # Add intervention logic here.
    #         if intervention_mask is not None and intervention_values is not None:
    #             # Assign the intervention values to the corresponding indices in X_all, specified by intervention_mask
    #             X_all[..., intervention_mask] = intervention_values
    #         # Generate the observations based on the history (given history + previously-generated observations)
    #         # and exogenous noise Z.
    #         generated_observations = (
    #             self.f.feed_forward(X_all[:, history_start_idx:inst_end_idx, :], W_adj_batch) + Z[:, time, :]
    #         )  # shape [batch_size, processed_dim_all]

    #         # Logic for processing discrete and binary variables. This is similar to static version.
    #         if gumbel_max_regions is not None:
    #             for region in gumbel_max_regions:
    #                 maxes = generated_observations[:, region].max(-1, keepdim=True)[0]  # shape [batch_size, 1]
    #                 generated_observations[:, region] = (generated_observations[:, region] >= maxes).float()
    #         if gt_zero_region is not None:
    #             generated_observations[:, gt_zero_region] = (generated_observations[:, gt_zero_region] > 0).float()

    #         X_all[:, size_history + time, :] = generated_observations
    #     # Add intervention logic here to make sure the generated observations respect the intervened values.
    #     if intervention_mask is not None and intervention_values is not None:
    #         # assign intervention_values to X_all at corresponding index specified by intervention_mask
    #         X_all[..., intervention_mask] = intervention_values

    #     X_simulate = X_all[:, size_history:, :].clone()  # shape [batch_size, time_span, proc_dim]

    #     return X_all, X_simulate

    # def simulate_SEM(
    #     self,
    #     Z: torch.Tensor,
    #     W_adj: torch.Tensor,
    #     X_history: torch.Tensor,
    #     gumbel_max_regions: Optional[List[List[int]]] = None,
    #     gt_zero_region: Optional[List[List[int]]] = None,
    #     intervention_mask: Optional[torch.Tensor] = None,
    #     intervention_values: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     This method simulates the SEM of the AR-DECI model, given forward time span, conditioned history, and exogenous noise Z.
    #     Currently, it does not support interventions, but it can be easily extended by adding few lines in the corresponding places.
    #     The simulation proceeds as follows: we loop over the entire simulation time span. For each forward time t, we call the TemporalFGNNI network
    #     to predict the observations based on the conditioned history and the generated observations at previous steps. Then, the observations are
    #     prediction + exogenous noise.

    #     We also support the discrete variables by specifying gumbel_max_regions and gt_zero_region.
    #     Args:
    #         Z: The exogenous noise variable with shape [batch_size, time_span, processed_dim_all].
    #         W_adj: The weighted adjacency matrix with shape [lag+1, num_nodes, num_nodes] or batched [batch_size, lag+1, num_nodes, num_nodes].
    #         X_history: The conditioned history observations. If None, it means the simulation starts at the very beginning.
    #             It has the shape [batch_size, history_length, processed_dim_all].
    #         gumbel_max_regions: a list of index lists `a` such that each subarray X[a] represents a one-hot encoded discrete random variable that must be
    #             sampled by applying the max operator.
    #         gt_zero_region: a list of indices such that X[a] should be thresholded to equal 1, if positive, 0 if negative. This is used to sample
    #             binary random variables. This also uses the Gumbel max trick implicitly
    #         intervention_mask: A mask of the interventions with shape [time_length, processed_dim_all]. If None or all elements are False, it means no interventions.
    #         intervention_values: The values of the interventions with shape [proc_dim] (Note proc_dim is not the same as proc_dim_all since proc_dim depends on the num_intervened_variables).
    #         If None, it means no interventions.
    #     Returns:
    #         The history+simulated observations with shape [batch_size, history_length+time_span, processed_dim_all].
    #         The simulated observations with shape [batch_size, time_span, processed_dim_all].


    #     """
    #     # Expansion dimension for W_adj if it does not have batch info
    #     assert len(Z.shape) == 3
    #     time_span = Z.shape[1]
    #     assert time_span > 0, "The time span must be >0"
    #     if len(W_adj.shape) == 3:
    #         W_adj = W_adj.unsqueeze(0)  # [1, lag+1, num_nodes, num_nodes]
    #     # Assertion checks. This will check the time span >0 and exogenous noise Z mathches the span.
    #     # Assertion checks for matching batch_size
    #     assert (
    #         X_history.shape[0] == W_adj.shape[0] or W_adj.shape[0] == 1
    #     ), "The batch size of the history and the adjacency matrix must match"

    #     # Assert X batch history has same batch size as Z noise batch
    #     assert (
    #         X_history.shape[0] == Z.shape[0]
    #     ), f"The batch size of history ({X_history.shape[0]}) and exogeneous noise Z ({Z.shape[0]}) must match "

    #     # Initialize the tensor for storing the observations. We use X_all to store the history and the generated observations.
    #     # We use X_simulate to store just the generated observations. size_history is to store the starting time for auoregressive simulation.
    #     X_all = torch.cat(
    #         [X_history, torch.zeros_like(Z)], dim=1
    #     )  # shape [batch_size, time_span+history_length, processed_dim_all]
    #     size_history = X_history.shape[1]
    #     # Loop over the entire time span to generate the observations for each time step. At each time step, we generate the observations
    #     # and update X_all and X_simulate accordingly. At the next time step, we use the updated X_all to generate the observations.
    #     # For future support of interventions, one can add the intervention logic before each generation, similar to static version.

    #     if intervention_mask is not None and intervention_values is not None:
    #         assert (
    #             Z.shape[1] >= intervention_mask.shape[0]
    #         ), "The future ahead time for observation generation must be >= the ahead time for intervention"
    #         # Convert the time_length in intervention mask to be compatible with X_all
    #         false_matrix_conditioning = torch.full(
    #             (X_history.shape[1], X_history.shape[2]), False, dtype=torch.bool, device=self.device
    #         )
    #         false_matrix_future = torch.full(
    #             (Z.shape[1] - intervention_mask.shape[0], Z.shape[2]), False, dtype=torch.bool, device=self.device
    #         )
    #         intervention_mask = torch.cat(
    #             (false_matrix_conditioning, intervention_mask, false_matrix_future), dim=0
    #         )  # shape [history_length+ time_span, processed_dim_all]

    #     for time in range(time_span):
    #         # Loop over num_nodes to propagate the instantaneous effect
    #         for _ in range(self.num_nodes):
    #             # Add intervention logic here.
    #             if intervention_mask is not None and intervention_values is not None:
    #                 # Assign the intervention values to the corresponding indices in X_all, specified by intervention_mask
    #                 X_all[..., intervention_mask] = intervention_values
    #             # Generate the observations based on the history (given history + previously-generated observations)
    #             # and exogenous noise Z.
    #             start_idx = size_history + time - self.lag
    #             end_idx = size_history + time + 1
    #             generated_observations = (
    #                 self.f.feed_forward(X_all[:, start_idx:end_idx, :], W_adj) + Z[:, time, :]
    #             )  # shape [batch_size, processed_dim_all]

    #             # Logic for processing discrete and binary variables. This is similar to static version.
    #             if gumbel_max_regions is not None:
    #                 for region in gumbel_max_regions:
    #                     maxes = generated_observations[:, region].max(-1, keepdim=True)[0]  # shape [batch_size, 1]
    #                     generated_observations[:, region] = (generated_observations[:, region] >= maxes).float()
    #             if gt_zero_region is not None:
    #                 generated_observations[:, gt_zero_region] = (generated_observations[:, gt_zero_region] > 0).float()

    #             X_all[:, size_history + time, :] = generated_observations

    #     # Add intervention logic here to make sure the generated observations respect the intervened values.
    #     if intervention_mask is not None and intervention_values is not None:
    #         # assign intervention_values to X_all at corresponding index specified by intervention_mask
    #         X_all[..., intervention_mask] = intervention_values

    #     X_simulate = X_all[:, size_history:, :].clone()  # shape [batch_size, time_span, proc_dim]

    #     return X_all, X_simulate


class FGNNI(nn.Module):
    """
    Defines the function f for the SEM. For each variable x_i we use
    f_i(x) = f(e_i, sum_{k in pa(i)} g(e_k, x_k)), where e_i is a learned embedding
    for node i.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        device: torch.device,
        embedding_size: Optional[int] = None,
        out_dim_g: Optional[int] = None,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = False,
        layers_g: Optional[List[int]] = None,
        layers_f: Optional[List[int]] = None,
        pre_len: int = 1,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: Device used.
            embedding_size: Size of the embeddings used by each node. If none, default is processed_dim_all.
            out_dim_g: Output dimension of the "inner" NN, g. If none, default is embedding size.
            layers_g: Size of the layers of NN g. Does not include input not output dim. If none, default
                      is [a], with a = max(2 * input_dim, embedding_size, 10).
            layers_f: Size of the layers of NN f. Does not include input nor output dim. If none, default
                      is [a], with a = max(2 * input_dim, embedding_size, 10)
        """
        super().__init__()
        self.group_mask = group_mask
        self.num_nodes, self.processed_dim_all = group_mask.shape
        self.device = device
        # Initialize embeddings
        self.embedding_size = embedding_size or self.processed_dim_all
        self.embeddings = self.initialize_embeddings()  # Shape (input_dim, embedding_size)
        # Set value for out_dim_g
        out_dim_g = out_dim_g or self.embedding_size
        # Set NNs sizes
        a = max(4 * self.processed_dim_all, self.embedding_size, 64)
        layers_g = layers_g or [a, a]
        layers_f = layers_f or [a, a]
        in_dim_g = self.embedding_size + self.processed_dim_all
        in_dim_f = self.embedding_size + out_dim_g
        self.g = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=out_dim_g,
            hidden_dims=layers_g,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )
        # self.f = [generate_fully_connected(
        #     input_dim=in_dim_f,
        #     output_dim=1,
        #     hidden_dims=layers_f,
        #     non_linearity=nn.LeakyReLU,
        #     activation=nn.Identity,
        #     device=self.device,
        #     normalization=norm_layer,
        #     res_connection=res_connection,
        # ) for i in range(self.num_nodes)]
        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=pre_len,
            hidden_dims=layers_f,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )

    def feed_forward(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        Computes non-linear function f(X, W) using the given weighted adjacency matrix.

        Args:
            X: Batched inputs, size (batch_size, processed_dim_all).
            W_adj: Weighted adjacency matrix, size (processed_dim_all, processed_dim_all) or size (batch_size, n, n).
        """
        if len(W_adj.shape) == 2:
            W_adj = W_adj.unsqueeze(0)

        # g takes inputs of size (*, embedding_size + processed_dim_all) and outputs (*, out_dim_g)
        # the input will be appropriately masked to correspond to one variable group
        # f takes inputs of size (*, embedding_size + out_dim_g) and outputs (*, processed_dim_all)
        # the ouptut is then masked to correspond to one variable

        # Generate required input for g (concatenate X and embeddings)
        X = X.unsqueeze(1)  # Shape (batch_size, 1, processed_dim_all)
        # Pointwise multiply X : shape (batch_size, 1, processed_dim_all) with self.group_mask : shape (num_nodes, processed_dim_all)
        X_masked = X * self.group_mask  # Shape (batch_size, num_nodes, processed_dim_all)
        E = self.embeddings.expand(X.shape[0], -1, -1)  # Shape (batch_size, num_nodes, embedding_size)
        X_in_g = torch.cat([X_masked, E], dim=2)  # Shape (batch_size, num_nodes, embedding_size + processed_dim_all)
        X_emb = self.g(X_in_g)  # Shape (batch_size, num_nodes, out_dim_g)
        # Aggregate sum and generate input for f (concatenate X_aggr and embeddings)
        X_aggr_sum = torch.matmul(W_adj.transpose(-1, -2), X_emb)  # Shape (batch_size, num_nodes, out_dim_g)
        # return vmap(torch.mm, in_dims=(None, 0))(W_adj.t(), X_emb)  # Shape (batch_size, num_nodes, out_dim_g)
        X_in_f = torch.cat([X_aggr_sum, E], dim=2)  # Shape (batch_size, num_nodes, out_dim_g + embedding_size)
        # Run f
        X_rec = self.f(X_in_f)  # Shape (batch_size, num_nodes, processed_dim_all)
        # Mask and aggregate
        X_rec = X_rec * self.group_mask  # Shape (batch_size, num_nodes, processed_dim_all)
        return X_rec.sum(1)  # Shape (batch_size, processed_dim_all, seq_len)

    def initialize_embeddings(self) -> torch.Tensor:
        """
        Initialize the node embeddings.
        """
        aux = torch.randn(self.num_nodes, self.embedding_size, device=self.device) * 0.01  # (N, E)
        return nn.Parameter(aux, requires_grad=True)


class TemporalFGNNI(FGNNI):
    """
    This defines the temporal version of FGNNI, which supports temporal adjacency matrix. The main difference is the modification of
    the feed_forward method, which generates the predictions based on the given parents (simultantanous + lagged). Additionally,
    we also need to override the method initialize_embeddings() in FunctionSEM so that it is consistent with the temporal data format.

    For now, since we use ANM for both simultaneous and lagged effect, we share the network parameters, and they only differ by the input embeddings.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        device: torch.device,
        lag: int,
        embedding_size: Optional[int] = None,
        out_dim_g: Optional[int] = None,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = False,
        layers_g: Optional[List[int]] = None,
        layers_f: Optional[List[int]] = None,
        pre_len:int=1
    ):
        """
        This initalize the temporal version of FGNNI.

        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: The device to use.
            lag: The lag for the model, should be >0.
            embedding_size: The embedding size to use. Thus, the generated embeddings will be of shape [lag+1, num_nodes, embedding_size].
            out_dim_g: The output dimension of the g function.
            norm_layer: The normalization layer to use.
            res_connection: Whether to use residual connection.
            layers_g: The hidden layers of the g function.
            layers_f: The hidden layers of the f function.
        """
        self.lag = lag
        self.pre_len = pre_len
        # Call init of the parent class. Note that we need to overwrite the initialize_embeddings() method so that
        # it is consistent with the temporal data format.
        super().__init__(
            group_mask=group_mask,
            device=device,
            embedding_size=embedding_size,
            out_dim_g=out_dim_g,
            norm_layer=norm_layer,
            res_connection=res_connection,
            layers_g=layers_g,
            layers_f=layers_f,
            pre_len=pre_len
        )

    def initialize_embeddings(self) -> torch.Tensor:
        """
        This overwrites the method in FunctionSEM. It will initialize the node embeddings with shape [lag+1, num_nodes, embedding_size].
        """
        # Initialize the embeddings.
        aux = (
            torch.randn(self.lag + 1, self.num_nodes, self.embedding_size, device=self.device) * 0.01
        )  # shape (lag+1, num_nodes, embedding_size)
        return nn.Parameter(aux, requires_grad=True)

    def feed_forward(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        This method overwrites the one in FGNNI and computes the SEM children = f(parents) specified by the temporal W_adj. The implementation strategy is similar to
        the static version.
        Args:
            X: Data from data loader with shape [batch_size, lag+1, processed_dim_all].
            W_adj: The temporal adjacency matrix with shape [lag+1, num_nodes, num_nodes] or [batch_size, lag+1, num_nodes, num_nodes].
        """

        # Assert tht if W_adj has batch dimension and >1 and X.shape[0]>1, then W_adj.shape[0] must match X.shape[0].
        # Assert X must have batch dimension.
        # Expand the weighted adjacency matrix dims for later matmul operation.
        if len(W_adj.shape) == 3:
            W_adj = W_adj.unsqueeze(0)  # shape (1, lag+1, num_nodes, num_nodes)
        assert len(X.shape) == 3, "The shape of X must be [batch, lag, proc_dim]"
        assert (
            W_adj.shape[1] == X.shape[1]
        ), f"The lag of W_adj ({W_adj.shape[1]}) is inconsistent to the lag of X ({X.shape[1]})"
        assert (
            W_adj.shape[0] == 1 or W_adj.shape[0] == X.shape[0]
        ), "The batch size of W_adj is inconsistent with X batch size"

        # For network g input, we mask the input with group mask, and concatenate it with the node embeddings.
        # Transform through g function. Output has shape shape (batch_size, lag+1, num_nodes, out_dim_g)
        X = X.unsqueeze(-2)  # shape (batch_size, lag+1, 1, processed_dim_all)
        X_masked = X * self.group_mask  # shape (batch_size, lag, num_nodes, processed_dim_all)
        E = self.embeddings.expand(
            X_masked.shape[0], -1, -1, -1
        )  # shape (batch_size, lag+1, num_nodes, embedding_size)
        E_lag = E[:, 1:]
        X_in_g = torch.cat(
            [X_masked, E_lag], dim=-1
        )  # shape (batch_size, lag, num_nodes, embedding_size+processed_dim_all)
        X_emb = self.g(X_in_g)  # shape (batch_size, lag, num_nodes, out_dim_g)
        # Aggregate the output from g network s.t. child value is generated by its parents, specified by W_adj.
        # This can be done by matrix multiplication with W_adj.transpose(-1, -2).flip([1]), followed by a summation over the lag dimension.
        # The flip is needed because W_adj[:,0,...] is the adj for instantaneous effect, but X[:, -1,...] is the data at current time step.
        # Output will have shape [batch_size, lag+1, num_nodes, out_dim_g]
        # Summation is done by summing over the lag dimension.


        X_aggr_sum = torch.einsum("klij,klio->kjo", W_adj.flip([1]), X_emb)  # shape (batch_size, predict_num_nodes, out_dim_g)

        # For network f input, we concatenate the results from the previous step  with the node embeddings, and feed it to f.
        # Output has shape (batch_size, num_nodes, processed_dim_all)
        X_in_f = torch.cat(
            [X_aggr_sum, E[:, 0, :, :]], dim=-1
        )  # shape (batch_size, predict_num_nodes, embedding_size+out_dim_g)
        X_rec = self.f(X_in_f)  # batch predict_nodes pre_len
        # X_rec = torch.stack([self.f[i](X_in_f[:,i]) for i in range(self.num_nodes)],dim=1) # shape (batch_size, predict_num_nodes, 1)

        # Masked the output with group_mask, followed by summation num_nodes to get correct node values.
        # output has shape (batch_size, processed_dim_all)
        # X_rec *= self.group_mask  # shape (batch_size, num_nodes, processed_dim_all)
        return X_rec # shape (batch_size, processed_dim_all, pre_len)

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TemporalFGNNIwithTimesNet(nn.Module):
    """
    This defines the temporal version of FGNNI, which supports temporal adjacency matrix. The main difference is the modification of
    the feed_forward method, which generates the predictions based on the given parents (simultantanous + lagged). Additionally,
    we also need to override the method initialize_embeddings() in FunctionSEM so that it is consistent with the temporal data format.

    For now, since we use ANM for both simultaneous and lagged effect, we share the network parameters, and they only differ by the input embeddings.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        lag: int,
        configs,
        pre_len:int=1
    ):
        """
        This initalize the temporal version of FGNNI.

        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: The device to use.
            lag: The lag for the model, should be >0.
            embedding_size: The embedding size to use. Thus, the generated embeddings will be of shape [lag+1, num_nodes, embedding_size].
            out_dim_g: The output dimension of the g function.
            norm_layer: The normalization layer to use.
            res_connection: Whether to use residual connection.
            layers_g: The hidden layers of the g function.
            layers_f: The hidden layers of the f function.
        """
        self.lag = lag
        self.pre_len = pre_len
        self.num_nodes, _ = group_mask.shape
        configs['pred_len'] = pre_len
        configs['seq_len'] = lag
        configs = DefaultMunch.fromDict(configs)
        # Call init of the parent class. Note that we need to overwrite the initialize_embeddings() method so that
        # it is consistent with the temporal data format.
        super().__init__()
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(self.num_nodes, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        self.predict_linear = nn.Linear(
            self.lag, self.lag+self.pre_len)
        self.projection = nn.Linear(
            configs.d_model, self.num_nodes, bias=True)
        # if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        #     self.projection = nn.Linear(
        #         configs.d_model, configs.c_out, bias=True)
        # if self.task_name == 'classification':
        #     self.act = F.gelu
        #     self.dropout = nn.Dropout(configs.dropout)
        #     self.projection = nn.Linear(
        #         configs.d_model * configs.seq_len, configs.num_class)
    
    def predict(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        if len(X.shape) == 2:
            X = X.unsqueeze(0)  # [1, lag, processed_dim_all]
        return self.feed_forward(X, W_adj)


    def feed_forward(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        This method overwrites the one in FGNNI and computes the SEM children = f(parents) specified by the temporal W_adj. The implementation strategy is similar to
        the static version.
        Args:
            X: Data from data loader with shape [batch_size, lag+1, processed_dim_all].
            W_adj: The temporal adjacency matrix with shape [lag+1, num_nodes, num_nodes] or [batch_size, lag+1, num_nodes, num_nodes].
        """

        # Assert tht if W_adj has batch dimension and >1 and X.shape[0]>1, then W_adj.shape[0] must match X.shape[0].
        # Assert X must have batch dimension.
        # Expand the weighted adjacency matrix dims for later matmul operation.
        if len(W_adj.shape) == 3:
            W_adj = W_adj.unsqueeze(0)  # shape (1, lag+1, num_nodes, num_nodes)
        assert len(X.shape) == 3, "The shape of X must be [batch, lag, proc_dim]"
        assert (
            W_adj.shape[1] == X.shape[1]
        ), f"The lag of W_adj ({W_adj.shape[1]}) is inconsistent to the lag of X ({X.shape[1]})"
        assert (
            W_adj.shape[0] == 1 or W_adj.shape[0] == X.shape[0]
        ), "The batch size of W_adj is inconsistent with X batch size"

        # For network g input, we mask the input with group mask, and concatenate it with the node embeddings.
        # Transform through g function. Output has shape shape (batch_size, lag+1, num_nodes, out_dim_g)
        #   X shape (batch_size, lag, self.nodes)
        #   W shape batch, lag nodes nodes
        W_total = W_adj.sum(-1) # batch lag nodes
        W_total = torch.where( W_total > 0, 1, 0)
        X = X*W_total
        dec_out = self.forecast(X)

        # Masked the output with group_mask, followed by summation num_nodes to get correct node values.
        # output has shape (batch_size, processed_dim_all)
        # X_rec *= self.group_mask  # shape (batch_size, num_nodes, processed_dim_all)
        return dec_out # shape (batch_size, pre_len, processed_dim_all, pre_len)

    def forecast(self, x_enc, x_mark_enc=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pre_len + self.lag, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pre_len + self.lag, 1))
        return dec_out[:, -self.pre_len:,]#batch pre_len nodes

    def imputation(self, x_enc, x_mark_enc, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output


class TemporalFGNNIwithInformer(nn.Module):
    """
    This defines the temporal version of FGNNI, which supports temporal adjacency matrix. The main difference is the modification of
    the feed_forward method, which generates the predictions based on the given parents (simultantanous + lagged). Additionally,
    we also need to override the method initialize_embeddings() in FunctionSEM so that it is consistent with the temporal data format.

    For now, since we use ANM for both simultaneous and lagged effect, we share the network parameters, and they only differ by the input embeddings.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        lag: int,
        configs,
        pre_len:int=1
    ):
        """
        This initalize the temporal version of FGNNI.

        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: The device to use.
            lag: The lag for the model, should be >0.
            embedding_size: The embedding size to use. Thus, the generated embeddings will be of shape [lag+1, num_nodes, embedding_size].
            out_dim_g: The output dimension of the g function.
            norm_layer: The normalization layer to use.
            res_connection: Whether to use residual connection.
            layers_g: The hidden layers of the g function.
            layers_f: The hidden layers of the f function.
        """

        self.lag = lag
        self.pre_len = pre_len
        self.num_nodes, _ = group_mask.shape
        configs['pred_len'] = pre_len
        configs['seq_len'] = lag
        self.label_len = int(lag/2)
        configs['label_len'] = int(self.label_len)
        configs = DefaultMunch.fromDict(configs)
        # Call init of the parent class. Note that we need to overwrite the initialize_embeddings() method so that
        # it is consistent with the temporal data format.
        super().__init__()

        
        # Embedding
        self.enc_embedding = DataEmbedding(self.num_nodes, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(self.num_nodes, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            # [
            #     ConvLayer(
            #         configs.d_model
            #     ) for l in range(configs.e_layers - 1)
            # ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, self.num_nodes, bias=True)
        )
        
    def predict(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        if len(X.shape) == 2:
            X = X.unsqueeze(0)  # [1, lag, processed_dim_all]
        W_total = W_adj.sum(-1) # batch lag nodes
        W_total = torch.where( W_total > 0, 1, 0)
        X = X*W_total
        return self.feed_forward(X)

    
   
    def long_forecast(self, x_enc,  x_dec,  x_mark_enc=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_dec,  x_mark_enc=None):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def feed_forward(self, x_enc):
        batch, lag, nodes = x_enc.shape
        x_dec_ = torch.zeros(batch, self.pre_len, nodes).to(x_enc.device)
        x_dec = torch.cat([x_enc[:, -self.label_len:], x_dec_],dim=1)  # B l+p D
        # if self.task_name == 'long_term_forecast':
        dec_out = self.long_forecast(x_enc, x_dec)
        return dec_out[:, -self.pre_len:, :]  # [B, L, D]
        # if self.task_name == 'short_term_forecast':
        #     dec_out = self.short_forecast(x_enc,  x_dec, x_mask)
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        


class TemporalHyperNet(nn.Module):
    """
    This hypernet class is for predicting the spline flow parameters with lagged parents
    """

    def __init__(
        self,
        cts_node: List[int],
        group_mask: torch.Tensor,
        device: torch.device,
        lag: int,
        param_dim: List[int],
        embedding_size: Optional[int] = None,
        out_dim_g: Optional[int] = None,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = False,
        layers_g: Optional[List[int]] = None,
        layers_f: Optional[List[int]] = None,
    ):
        """
        This initialize the temporal hypernet instances. The hypernet has the form:
        param_i = f(g(G,X), e_i) where G is the temporal graph, X has shape [n_batch, lag+1, proc_dim_all] and e_i is the
        embedding for node i.
        Args:
            cts_node: A list of node idx specifies the cts variables.
            group_mask: A mask of shape [num_nodes, proc_dims] such that group_mask[i, j] = 1 when col j is in group i.
            device: Device to use
            lag: The specified lag for the temporal SEM model.
            param_dim: A list of ints that specifies the output parameters dims from the hypernet.
                For conditional spline flow, the output dims dependes on the order = linear or order = quadratic.
            embedding_size: The embedding size for the node embeddings.
            out_dim_g: The output dims of the inner g network.
            norm_layer: Whether to use layer normalization
            res_connection: whether to use residual connection
            layers_g: the hidden layers of the g
            layers_f: the hidden layers of the f
        """
        super().__init__()
        self.cts_node = cts_node
        self.lag = lag
        self.param_dim = param_dim
        self.total_param = sum(self.param_dim)
        self.group_mask = group_mask
        self.num_nodes, self.processed_dim_all = group_mask.shape
        self.device = device
        # Initialize embeddings
        self.embedding_size = embedding_size or self.processed_dim_all
        self.embeddings = self.initialize_embeddings()  # Shape (num_node, embedding_size)
        self.init_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # Set value for out_dim_g
        out_dim_g = out_dim_g or max(
            8 * 4, self.embedding_size
        )  # default num_bins for conditional flow is 8, and we need f to output 4 of them.
        # Set NNs sizes
        a = max(4 * self.processed_dim_all, self.embedding_size, 64)
        layers_g = layers_g or [a, a]
        layers_f = layers_f or [a, a]
        in_dim_g = self.processed_dim_all + self.embedding_size
        in_dim_f = self.embedding_size + out_dim_g
        self.g = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=out_dim_g,
            hidden_dims=layers_g,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )
        # self.f = [generate_fully_connected(
        #     input_dim=in_dim_f,
        #     output_dim=self.total_param,
        #     hidden_dims=layers_f,
        #     non_linearity=nn.LeakyReLU,
        #     activation=nn.Identity,
        #     device=self.device,
        #     normalization=norm_layer,
        #     res_connection=res_connection,
        # )for i in range(self.num_nodes)]
        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=self.total_param,
            hidden_dims=layers_f,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )
        # Initialize the associated weights by calling self.W = self._initialize_W(). self.W has shape [lag+1, num_nodes, num_nodes]
        # self.W = self._initialize_W() #[lag, num_nodes, num_nodes]

    # def _initialize_W(self) -> torch.Tensor:
    #     """
    #     Initializes the associated weight with shape [lag, num_nodes, num_nodes]. Currently, initialize to zero.
    #     Returns: the initialized weight with shape [lag, num_nodes, num_nodes]
    #     """
    #     W = torch.zeros(self.lag , self.num_nodes, self.num_nodes, device=self.device)
    #     return nn.Parameter(W, requires_grad=True)
    
    def initialize_embeddings(self) -> torch.Tensor:
        """
        This overwrites the method in FunctionSEM. It will initialize the node embeddings with shape [lag+1, num_nodes, embedding_size].
        """
        # Initialize the embeddings.
        aux = (
            torch.randn(self.lag + 1, self.num_nodes, self.embedding_size, device=self.device) * 0.01
        )  # shape (lag+1, num_nodes, embedding_size)
        return nn.Parameter(aux, requires_grad=True)

    def forward(self, X: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        This feed-forwards the input to generate predicted parameters for conditional flow. This will return a parameter
        list of shape [len(self.param_dim), N_batch, num_cts_node*param_dim]. The first is the
        length of the tuple (i.e. the num of parameters required for conditional flow), second is the batch number, and third is
        the concatenated parameters for all continuous nodes.
        Args:
            X: A dict consists with two keys, "W" is the weighted adjacency matrix with shape [lag+1, num_node, num_node]
            and "X" is the history data with shape [N_batch, lag, proc_dim_all].

        Returns:
            A tuple of parameters with shape [N_batch, num_cts_node*param_dim_each].
                The length of tuple is len(self.param_dim),
        """

        assert "W" in X and "X" in X and len(X) == 2, "The key for input can only contain three keys, 'W', 'X'."

        X_hist = X["X"]
        W = X["W"]
        assert W.dim() == 4, "W must have shape [batch, lag+1, num_node, num_node]"

        # assert lag
        assert X_hist.shape[1] == W.shape[1], "The input observation should be the history observation."
        X_hist = X_hist.unsqueeze(-2)  # [batch, lag, 1, proc_dim]
        X_hist_masked = X_hist * self.group_mask  # [batch, lag, node, proc_dim]
        E = self.embeddings.expand(
            X_hist_masked.shape[0], -1, -1, -1
        )  # shape (batch_size, lag+1, num_nodes, embedding_size)
        E_lag = E[..., 1:, :, :]  # shape [batch_size, lag, num_nodes, embedding_size]
        E_inst = E[..., 0, :, :]  # shape [batch_size, num_nodes, embedding_size]
        X_in_g = torch.cat(
            [X_hist_masked, E_lag], dim=-1
        )  # shape (batch_size, lag, num_nodes, embedding_size+proc_dim)
        X_emb = self.g(X_in_g)  # shape (batch_size, lag, num_nodes, out_dim_g)

        
        # # # change  add weight
        # W_lag_exp = W_lag_exp*self.W.unsqueeze(0)

        X_aggr_sum = torch.einsum(
            "klij,klio->kjo", W.flip([1]), X_emb
        )  # shape (batch_size, num_nodes, out_dim_g)

        X_in_f = torch.cat([X_aggr_sum, E_inst], dim=-1)  # shape (batch_size, num_nodes, embedding_size+out_dim_g)
        X_rec = self.f(X_in_f)
        # X_rec = torch.stack([self.f[i](X_in_f[:,i]) for i in range(self.num_nodes)],dim=1) # shape (batch_size, num_nodes, total_params)
        X_selected = X_rec[..., self.cts_node, :] * self.init_scale  # shape [batch_size, cts_node, total_params]
        param_list = torch.split(
            X_selected, self.param_dim, dim=-1
        )  # a list of tensor with shape [batch_size, cts_node, each_params]

        output = tuple(
            param.reshape([-1, len(self.cts_node) * param.shape[-1]]) for param in param_list
        )  # Tuple with shape [batch, cts_node*each_param]
        return output
