# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import DataLoader

from ...datasets.dataset import Dataset, TemporalDataset
from ...datasets.temporal_tensor_dataset import TemporalTensorDataset, MyTemporalTensorDataset
from ...datasets.variables import Variables
from ...utils.causality_utils import (
    get_ate_from_samples,
    get_mask_and_value_from_temporal_idxs,
    intervention_to_tensor,
    process_adjacency_mats,
)
from ...utils.helper_functions import to_tensors
from ...utils.nri_utils import convert_temporal_to_static_adjacency_matrix, edge_prediction_metrics_multisample
from ..imodel import IModelForTimeseries
from .base_distributions import TemporalConditionalSplineFlow
from .deci import DECI
from .generation_functions import TemporalContractiveInvertibleGNN
from .variational_distributions import AdjMatrix, TemporalThreeWayGrahpDist


class Rhino(DECI, IModelForTimeseries):
    """
    This class implements the AR-DECI model for end-to-end time series causal inference. It is inherited from the DECI class.
    One of the principle is to re-use as many code from the DECI as possible to avoid code repetition. For an overview of the design and
    formulation, please refer to the design proposal doc/proposals/AR-DECI.md.
    """

    _saved_most_likely_adjacency_file = "saved_most_likely_adjacency.npy"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        lag: int,
        allow_instantaneous: bool,
        imputation: bool = False,
        lambda_dag: float = 1.0,
        lambda_sparse: float = 1.0,
        lambda_prior: float = 1.0,
        tau_gumbel: float = 1.0,
        base_distribution_type: str = "spline",
        spline_bins: int = 8,
        var_dist_A_mode: str = "temporal_three",
        norm_layers: bool = True,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        cate_rff_n_features: int = 3000,
        cate_rff_lengthscale: Union[float, List[float], Tuple[float, float]] = (
            0.1,
            1.0,
        ),
        prior_A_confidence: float = 0.5,
        graph_constraint_matrix: Optional[np.ndarray] = None,
        ICGNN_embedding_size: Optional[int] = None,
        init_logits: Optional[List[float]] = None,
        conditional_embedding_size: Optional[int] = None,
        conditional_encoder_layer_sizes: Optional[List[int]] = None,
        conditional_decoder_layer_sizes: Optional[List[int]] = None,
        conditional_spline_order: str = "quadratic",
        additional_spline_flow: int = 0,
        disable_diagonal_eval: bool = True,
    ):
        """
        Initialize the Rhino. Most of the parameters are from DECI class. For initialization, we first initialize the DECI class, and then
        replace the (1) static variational distribution to temporal VI distribution, (2) the static ICGNN to temporal ICGNN,
        (3) setup the temporal graph constraints matrix. The above steps will be done in super().__init__(...). We also add assertions to make sure
        the model satisfies the V0 design principle (e.g. does not support missing data, etc.).
        Args:
            lag: The lag for the AR-DECI model.
            allow_instantaneous: Whether to allow the instantaneous effects.
            ICGNN_embedding_size: This is the embedding sizes for ICGNN.
            init_logits: The initialized logits used for temporal variational distribution. See TemporalThreeWayGraphDist doc string for details.
            conditional_embedding_size: the embedding size of hypernet of conditional spline flow.
            conditional_encoder_layer_sizes: hypernet encoder layer size
            conditional_decoder_layer_sizes: hypernet decoder layer size
            conditional_spline_order: the transformation order used for conditional spline flow
            additional_spline_flow: the number of additional spline flow on top of conditional spline flow.
            Other parameters: Refer to DECI class docstring.
        """
        # Assertions: (1) lag>0, (2) imputation==False
        assert lag > 0, "The lag must be greater than 0."
        assert imputation is False, "For V0 AR-DECI, the imputation must be False."

        # Initialize the DECI class. Note that some of the methods in DECI class will be overwritten (e.g. self._set_graph_constraint(...)).
        # For soft prior of AR-DECI, this is not set inside __init__, since the prior matrix should be designed as an attribute of dataset.
        # Thus, the prior matrix will be set in run_train(), where datasest is one of the input. Here, we use None for prior_A.
        # For variational distribution and ICGNN, we overwrite the self._create_variational_distribution(...) and self._create_ICGNN(...)
        # to generate the correct type of var_dist and ICGNN.

        # Note that we may want to check if variables are all continuous.
        self.allow_instantaneous = allow_instantaneous
        self.init_logits = init_logits
        self.lag = lag
        self.cts_node = variables.continuous_idxs
        self.cts_dim = len(self.cts_node)
        # conditional spline flow hyper-params.
        self.conditional_embedding_size = conditional_embedding_size
        self.conditional_encoder_layer_sizes = conditional_encoder_layer_sizes
        self.conditional_decoder_layer_sizes = conditional_decoder_layer_sizes
        self.conditional_spline_order = conditional_spline_order
        self.additional_spline_flow = additional_spline_flow
        # For V0 AR-DECI, we only support mode_adjacency="learn", so hardcoded this argument.
        super().__init__(
            model_id=model_id,
            variables=variables,
            save_dir=save_dir,
            device=device,
            imputation=imputation,
            lambda_dag=lambda_dag,
            lambda_sparse=lambda_sparse,
            lambda_prior=lambda_prior,
            tau_gumbel=tau_gumbel,
            base_distribution_type=base_distribution_type,
            spline_bins=spline_bins,
            var_dist_A_mode=var_dist_A_mode,
            imputer_layer_sizes=None,
            mode_adjacency="learn",
            norm_layers=norm_layers,
            res_connection=res_connection,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            cate_rff_n_features=cate_rff_n_features,
            cate_rff_lengthscale=cate_rff_lengthscale,
            prior_A=None,
            prior_A_confidence=prior_A_confidence,
            prior_mask=None,
            graph_constraint_matrix=graph_constraint_matrix,
            embedding_size=ICGNN_embedding_size,
            disable_diagonal_eval=disable_diagonal_eval,
        )

    def _generate_error_likelihoods(self, base_distribution_string: str, variables: Variables) -> Dict[str, nn.Module]:
        """
        This overwrite the parent functions. To avoid code repetition, if the base_distribution type is 'conditional_spline',
        we call the parent method with type 'spline' and then replace the dict['continuous'] with the conditional spline.
        Args:
            base_distribution_string: type of base distributions, can be "fixed_gaussian", "gaussian", "spline" or "conditional_spline"
            variables: variables object

        Returns:
            error_likelihood dict
        """
        error_likelihoods = super()._generate_error_likelihoods(
            base_distribution_string if base_distribution_string != "conditional_spline" else "spline",
            variables,
        )

        if base_distribution_string == "conditional_spline":
            error_likelihoods["continuous"] = TemporalConditionalSplineFlow(
                cts_node=self.cts_node,
                group_mask=torch.tensor(self.variables.group_mask).to(self.device),
                device=self.device,
                lag=self.lag,
                num_bins=self.spline_bins,
                additional_flow=self.additional_spline_flow,
                layers_g=self.conditional_encoder_layer_sizes,
                layers_f=self.conditional_decoder_layer_sizes,
                embedding_size=self.conditional_embedding_size,
                order=self.conditional_spline_order,
            )
        return error_likelihoods

    def _create_var_dist_A_for_deci(self, var_dist_A_mode: str) -> Optional[AdjMatrix]:
        """
        This overwrites the original DECI one to generate a variational distribution supporting the temporal adj matrix.
        Args:
            var_dist_A_mode: the type of the variational distribution

        Returns:
            An instance of variational distribution.
        """
        assert (
            var_dist_A_mode == "temporal_three"
        ), f"Currently, var_dist_A only support type temporal_three, but {var_dist_A_mode} given"
        var_dist_A = TemporalThreeWayGrahpDist(
            device=self.device,
            input_dim=self.num_nodes,
            lag=self.lag,

            norm_layer=self.norm_layer,
            res_connection=self.res_connection,
            encoder_layer_sizes=self.encoder_layer_sizes,

            tau_gumbel=self.tau_gumbel,
            init_logits=self.init_logits,
        )
        return var_dist_A

    def _create_ICGNN_for_deci(self) -> nn.Module:
        """
        This overwrites the original one in DECI to generate an ICGNN that supports the auto-regressive formulation.

        Returns:
            An instance of the temporal ICGNN
        """

        return TemporalContractiveInvertibleGNN(
            group_mask=torch.tensor(self.variables.group_mask),
            lag=self.lag,
            device=self.device,
            norm_layer=self.norm_layer,
            res_connection=self.res_connection,
            encoder_layer_sizes=self.encoder_layer_sizes,
            decoder_layer_sizes=self.decoder_layer_sizes,
            embedding_size=self.embedding_size,
        )

    def networkx_graph(self) -> nx.DiGraph:
        """
        This function converts the most probable graph to networkx graph. Due to the incompatibility of networkx and temporal
        adjacency matrix, we need to convert the temporal adj matrix to its static version before changing it to networkx graph.
        """
        adj_mat = self.get_adj_matrix(
            samples=1, most_likely_graph=True, squeeze=True
        )  # shape [lag+1, num_nodes, num_nodes]
        # Convert to static graph
        static_adj_mat = convert_temporal_to_static_adjacency_matrix(adj_mat, conversion_type="full_time", fill_value=0)
        # Check if non DAG adjacency matrix
        assert np.trace(scipy.linalg.expm(static_adj_mat)) == (self.lag + 1) * self.num_nodes, "Generate non DAG graph"
        return nx.convert_matrix.from_numpy_matrix(static_adj_mat, create_using=nx.DiGraph)

    def sample_graph_posterior(self, do_round: bool = True, samples: int = 100) -> Tuple[List[nx.DiGraph], np.ndarray]:
        """
        This function samples the graph from the variational posterior and convert them into networkx graph without duplicates.
        Due to the incompatibility of temporal adj matrix and networkx graph, they will be converted to its corresponding
        static adj before changing them to networkx graph.
        Args:
            do_round: If we round the probability during sampling.
            samples: The number of sampled graphs.

        Returns:
            A list of networkx digraph object.

        """

        adj_mats = self.get_adj_matrix(
            do_round=do_round, samples=samples, most_likely_graph=False
        )  # shape [samples, lag+1, num_nodes, num_nodes]
        # Convert to static graph
        static_adj_mats = convert_temporal_to_static_adjacency_matrix(
            adj_mats, conversion_type="full_time", fill_value=0
        )
        adj_mats, adj_weights = process_adjacency_mats(static_adj_mats, (self.lag + 1) * self.num_nodes)
        graph_list = [nx.convert_matrix.from_numpy_matrix(adj_mat, create_using=nx.DiGraph) for adj_mat in adj_mats]
        return graph_list, adj_weights

    @classmethod
    def name(cls) -> str:
        return "rhino"

    def set_graph_constraint(self, graph_constraint_matrix: Optional[np.ndarray]):
        """
        This method overwrite the original set_graph_constraint method in DECI class, s.t. it supports the temporal
        graph constraints. For the meaning of value in each constraint matrix, please refer to the DECI class docstring.
        Args:
            graph_constraint_matrix: temporal graph constraints matrix with shape [lag+1, num_nodes, num_nodes] or None.
        """
        # The implementation is very similar to the original one in DECI class. The difference is that for neg_constraint_matrix
        # we need to disable the diagonal elements of constraint[0, ...] rather than the entire constraint matrix in original DECI.
        # Also the original implementation only supports 2D matrix.
        if graph_constraint_matrix is None:
            neg_constraint_matrix = np.ones((self.lag + 1, self.num_nodes, self.num_nodes))
            if self.allow_instantaneous:
                np.fill_diagonal(neg_constraint_matrix[0, ...], 0)
            else:
                neg_constraint_matrix[0, ...] = np.zeros((self.num_nodes, self.num_nodes))
            self.neg_constraint_matrix = torch.as_tensor(neg_constraint_matrix, device=self.device, dtype=torch.float32)
            self.pos_constraint_matrix = torch.zeros((self.lag + 1, self.num_nodes, self.num_nodes), device=self.device)
        else:
            negative_constraint_matrix = np.nan_to_num(graph_constraint_matrix, nan=1.0)
            if not self.allow_instantaneous:
                negative_constraint_matrix[0, ...] = np.zeros((self.num_nodes, self.num_nodes))
            self.neg_constraint_matrix = torch.as_tensor(
                negative_constraint_matrix, device=self.device, dtype=torch.float32
            )
            # Disable diagonal elements in the instant graph constraint.
            torch.diagonal(self.neg_constraint_matrix[0, ...]).zero_()
            positive_constraint_matrix = np.nan_to_num(graph_constraint_matrix, nan=0.0)
            self.pos_constraint_matrix = torch.as_tensor(
                positive_constraint_matrix, device=self.device, dtype=torch.float32
            )

    def _log_prob(
        self,
        x: torch.Tensor,
        predict: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        W: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """
        This method computes the log probability of the observed data given the predicitons from SEM
        Args:
            x: a temporal data tensor with shape [N, lag+1, proc_dims] (proc_dims may not be equal to num_nodes with categorical variables)
            or [lag+1, proc_dims]
            predict: predictions from SEM with shape [N, proc_dims] or [proc_dims]
            intervention_mask: a mask indicating which variables have been intervened upon. For V0, do not support it.
            W: The weighted adjacency matrix used to compute the predict with shape [lag+1, num_nodes, num_nodes].

        Returns: A log probability tensor with shape [N] or a scalar

        """
        # The overall code structure should be similar to original DECI, but the key difference is the data are now in
        # temporal format with shape [N, lag+1, proc_dims], where data[:,-1,:] represents the data in current time step.
        # From the formulation of AR-DECI, we only care about the conditional log prob p(x_t|x_{<t}). So, the
        # predict from SEM should have shape [N, proc_dims] or [proc_dims]. Then, we can compute data[,-1,:] - predict to get the log probability.

        typed_regions = self.variables.processed_cols_by_type
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, lag+1, proc_dim]
        batch_size, _, proc_dim = x.shape

        if predict.dim() == 1:
            predict = predict.unsqueeze(0)

        # Continuous
        cts_bin_log_prob = torch.zeros(batch_size, proc_dim).to(self.device)  # [batch, proc_dim]
        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        if continuous_range:
            # History-dependent noise
            if self.base_distribution_type == "conditional_spline":
                assert W is not None
                cts_bin_log_prob[..., continuous_range] = self.likelihoods["continuous"].log_prob(
                    x[..., -1, continuous_range] - predict[..., continuous_range],
                    X_history=x[..., 0:-1, :],
                    W=W,
                )
            else:
                cts_bin_log_prob[..., continuous_range] = self.likelihoods["continuous"].log_prob(
                    x[..., -1, continuous_range] - predict[..., continuous_range]
                )

        # Binary
        binary_range = [i for region in typed_regions["binary"] for i in region]
        if binary_range:
            cts_bin_log_prob[..., binary_range] = self.likelihoods["binary"].log_prob(
                x[..., -1, binary_range], predict[..., binary_range]
            )

        if intervention_mask is not None:
            cts_bin_log_prob[..., intervention_mask] = 0.0

        log_prob = cts_bin_log_prob.sum(-1)  # [1] or [batch]

        # Categorical
        if "categorical" in typed_regions:
            for region, likelihood, idx in zip(
                typed_regions["categorical"],
                self.likelihoods["categorical"],
                self.variables.var_idxs_by_type["categorical"],
            ):
                # Can skip likelihood computation completely if intervened
                if (intervention_mask is None) or (intervention_mask[idx] is False):
                    log_prob += likelihood.log_prob(x[..., -1, region], predict[..., region])

        return log_prob

    def _icgnn_cts_mse(self, x: torch.Tensor, predict: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean-squared error (MSE) of the ICGNN on the continuous variables of the model.

        Args:
            x: a temporal data tensor with shape [N, lag+1, proc_dims] (proc_dims may not be equal to num_nodes with categorical variables)
            or [lag+1, proc_dims]
            predict: predictions from SEM with shape [N, proc_dims] or [proc_dims]

        Returns:
            MSE of ICGNN predictions on continuous variables. A number if x has shape (lag+1, proc_dims), or an array of
            shape (N) is X has shape (N, lag+1, proc_dim).
        """
        typed_regions = self.variables.processed_cols_by_type
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, lag+1, proc_dim]

        if predict.dim() == 1:
            predict = predict.unsqueeze(0)

        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        return (x[..., -1, continuous_range] - predict[..., continuous_range]).pow(2).sum(-1)

    def _sample_base(self, Nsamples: int, time_span: int = 1) -> torch.Tensor:
        """
        This method draws noise samples from the base distribution with simulation time_span.
        Args:
            Nsamples: The batch size of samples.
            time_span: The simulation time span.

        Returns: A tensor with shape [Nsamples, time_span, proc_dims]
        """

        # This noise is typcally used for simulating the SEM to generate observations. However, since AR-DECI is a temporal model,
        # simulating it requires an additional argument, time_span, to specify the simulation length.

        # The code structure is similar to
        # original DECI in V0, but draw samples with shape [Nsamples, time_span, proc_dims], rather than [Nsamples, proc_dims].

        sample = torch.zeros((Nsamples, time_span, self.processed_dim_all), device=self.device)
        total_size = np.prod(sample.shape[:-1])
        typed_regions = self.variables.processed_cols_by_type
        # Continuous and binary
        for type_region in ["continuous", "binary"]:
            range_ = [i for region in typed_regions[type_region] for i in region]
            if range_:
                if self.base_distribution_type == "conditional_spline" and type_region == "continuous":
                    sample[..., range_] = (
                        self.likelihoods[type_region].base_dist.sample([total_size]).view(*sample.shape[:-1], -1)
                    )  # shape[Nsamples, time_span, cts_dim]
                else:
                    sample[..., range_] = self.likelihoods[type_region].sample(total_size).view(*sample.shape[:-1], -1)

        # Categorical
        if "categorical" in typed_regions:
            for region, likelihood in zip(typed_regions["categorical"], self.likelihoods["categorical"]):
                sample[..., region] = likelihood.sample(total_size).view(*sample.shape[:-1], -1)

        return sample

    #change
    def sample(  # type:ignore
        self,
        Nsamples: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        samples_per_graph_groups: int = 1,
        X_history: Optional[Union[torch.Tensor, np.ndarray]] = None,
        time_span: int = 1,
    ) -> torch.Tensor:
        """
        This method samples the observations for the AR-DECI model. For V0, due to the lack of source node model, one has to provide
        the history (history length > lag) for conditional observation generation.
        Args:
            X_history: The history that model conditioned on. Shape [N_history_batch, hist_length, proc_dims] or [hist_length, proc_dims]
            time_span: The simulation length
            Nsamples: The number of samples for each history.
            most_likely_graph: Whether to use the most likely graph.
            intervention_idxs: ndarray or Tensor with shape [num_interventions, 2], where intervention_idxs[...,0] stores the index of the intervetnon variable
                and intervention_idxs[...,1] stores the ahead time of that intervention. E.g. intervention_idxs[0] = [4,0] means the intervention variable idx is 4
                at current time t.
            intervention_values: ndarray or Tensor with shape [proc_dim] (proc_dim depends on num_interventions) storing the intervention values corresponding to intervention_idxs
            samples_per_graph: The number of samples per sampled graph from posterior. If most_likely_graph is true, the Nsamples
            should be equal to samples_per_graph.
            max_batch_size: maximum batch size to use for AR-DECI when drawing samples. Larger is faster but more memory intensive

        Returns: A tensor with shape [N_history_batch, Nsamples, time_span, proc_dims]

        """
        assert X_history is not None, "For V0 AR-DECI, empty history generation is not supported"
        # Assertions for Nsamples must be multiple of samples_per_graph.
        if most_likely_graph:
            assert Nsamples == samples_per_graph_groups
        else:
            assert (
                Nsamples % samples_per_graph_groups == 0
            ), f"Nsamples ({Nsamples}) must be multiples of samples_per_graph ({samples_per_graph_groups})"
        # convert X_history to torch.Tensor
        if isinstance(X_history, np.ndarray):
            X_history = to_tensors(X_history, device=self.device, dtype=torch.float)[0]

        # Assert history shape
        assert (
            X_history.dim() == 2 or X_history.dim() == 3
        ), "The shape of X_history must be [hist_length, proc_dims] or [N_history_batch, hist_length, proc_dims]"
        if X_history.dim() == 2:
            X_history = X_history[None, ...]  # shape [1, history_length, proc_dims]
        # Assertions for length_hist must be larger than lag.
        N_history_batch, len_history, proc_dim = X_history.shape
        assert (
            len_history >= self.lag
        ), f"Length of history ({len_history}) must be equal or larger than model lag ({self.lag}) "

        # Convert intervention_idxs and intervention_values to torch.Tensor and generate intervention_mask
        if intervention_idxs is not None and intervention_values is not None:
            # These interventions are sorted
            intervention_idxs, intervention_mask, intervention_values = intervention_to_tensor(
                intervention_idxs=intervention_idxs,
                intervention_values=intervention_values,
                group_mask=self.variables.group_mask,
                device=self.device,
                is_temporal=True,
            )

        else:
            intervention_mask, intervention_values = None, None

        # Find out the gumbel and binary regions as original DECI.
        gumbel_max_regions = self.variables.processed_cols_by_type["categorical"]
        gt_zero_region = [j for i in self.variables.processed_cols_by_type["binary"] for j in i]

        with torch.no_grad():
            num_graph_samples = Nsamples // samples_per_graph_groups
            if self.base_distribution_type == "conditional_spline":
                # Noise cannot be sampled outside the simulate_SEM due to history dependence.
                # Need call self.ICGNN.simulate_SEM_conditional(...)
                # Get conditional distribution
                conditional_dist = self.likelihoods["continuous"]
                cts_node = conditional_dist.cts_node
                Z = self._sample_base(
                        samples_per_graph_groups * N_history_batch, time_span=time_span
                    ) # [batch*samples_per_graph, time_span, proc_dim]
                X_all = torch.cat(
                    [X_history, torch.zeros(N_history_batch, time_span, proc_dim).to(self.device)], dim=1
                 ).repeat(samples_per_graph_groups, 1,1) # shape [samples_per_graph* batch_size, time_span+history_length, processed_dim_all]
                X_all_total = []
                X_simulate_total = []
                #X_history  shape [batch, history_length, proc_dims]
                # Iterate over graph samples
                if intervention_mask is not None and intervention_values is not None:
                    assert (
                        time_span >= intervention_mask.shape[0]
                    ), "The future ahead time for observation generation must be >= the ahead time for intervention"
                    # Convert the time_length in intervention mask to be compatible with X_all
                    false_matrix_conditioning = torch.full(X_history.shape[1:], False, dtype=torch.bool, device=self.device)
                    false_matrix_future = torch.full(
                        (Z.shape[1] - intervention_mask.shape[0], Z.shape[2]), False, dtype=torch.bool, device=self.device
                    )
                    intervention_mask = torch.cat(
                        (false_matrix_conditioning, intervention_mask, false_matrix_future), dim=0
                    )  # shape [history_length+ time_span, processed_dim_all]

                for num in range(num_graph_samples):
                    Z_ = Z.clone()
                    X_all_ =X_all.clone()
                    for time in range(time_span):
                        history_start_idx = len_history + time - self.lag
                        inst_end_idx = len_history + time + 1
                        history = X_all_[:, history_start_idx : inst_end_idx - 1] # batch*samples_per_graph, lag, node
                        W_adj = self.get_weighted_adj_matrix(
                            x_history=history[:N_history_batch],
                            do_round=most_likely_graph,
                            samples=1,
                            most_likely_graph=most_likely_graph,
                        ).view(-1, self.lag+1,proc_dim,proc_dim)  #[ batch, lag+1, nodes, nodes]
                        # import pdb
                        # pdb.set_trace()
                        W_adj=W_adj.repeat(samples_per_graph_groups,1,1,1)

                        Z_[:, time, cts_node] = conditional_dist.transform_noise(  # type:ignore
                            Z=Z_[:, time, cts_node], X_history=history, W=W_adj
                        )  # shape [batch_size, cts_dim]
                        # Add intervention logic here.
                        if intervention_mask is not None and intervention_values is not None:
                            # Assign the intervention values to the corresponding indices in X_all, specified by intervention_mask
                            X_all_[..., intervention_mask] = intervention_values
                        # Generate the observations based on the history (given history + previously-generated observations)
                        # and exogenous noise Z.
                        generated_observations = self.ICGNN.f.feed_forward(X_all_[:, history_start_idx : inst_end_idx], W_adj=W_adj) + Z[:, time, :] # batch*samples_per_graph, lag, node

                        # Logic for processing discrete and binary variables. This is similar to static version.
                        if gumbel_max_regions is not None:
                            for region in gumbel_max_regions:
                                maxes = generated_observations[:, region].max(-1, keepdim=True)[0]  # shape [batch_size, 1]
                                generated_observations[:, region] = (generated_observations[:, region] >= maxes).float()
                        if gt_zero_region is not None:
                            generated_observations[:, gt_zero_region] = (generated_observations[:, gt_zero_region] > 0).float()
                        X_all[:, len_history + time, :] = generated_observations
                    if intervention_mask is not None and intervention_values is not None:
                        # assign intervention_values to X_all at corresponding index specified by intervention_mask
                        X_all_[..., intervention_mask] = intervention_values

                    X_simulate = X_all_[:, len_history:, :].clone()  # shape [batch_size, time_span, proc_dim]
                    X_all_total.append(X_all_)
                    X_simulate_total.append(X_simulate)
                

     
                samples = (
                    torch.cat(X_simulate_total, dim=0).view(Nsamples, N_history_batch, time_span, proc_dim)
                )  # shape [Nsamples, N_history_batch, time_span, proc_dim]
            else:


                Z = self._sample_base(
                        samples_per_graph_groups * N_history_batch, time_span=time_span
                    ) # [batch*samples_per_graph, time_span, proc_dim]
                X_all = torch.cat(
                    [X_history, torch.zeros(N_history_batch, time_span, proc_dim).to(self.device)], dim=1
                 ).repeat(samples_per_graph_groups, 1,1) # shape [samples_per_graph* batch_size, time_span+history_length, processed_dim_all]
                X_all_total = []
                X_simulate_total = []
                #X_history  shape [batch, history_length, proc_dims]
                # Iterate over graph samples
                if intervention_mask is not None and intervention_values is not None:
                    assert (
                        time_span >= intervention_mask.shape[0]
                    ), "The future ahead time for observation generation must be >= the ahead time for intervention"
                    # Convert the time_length in intervention mask to be compatible with X_all
                    false_matrix_conditioning = torch.full(X_history.shape[1:], False, dtype=torch.bool, device=self.device)
                    false_matrix_future = torch.full(
                        (Z.shape[1] - intervention_mask.shape[0], Z.shape[2]), False, dtype=torch.bool, device=self.device
                    )
                    intervention_mask = torch.cat(
                        (false_matrix_conditioning, intervention_mask, false_matrix_future), dim=0
                    )  # shape [history_length+ time_span, processed_dim_all]

                for num in range(num_graph_samples):
                    Z_ = Z.clone()
                    X_all_ =X_all.clone()
                    for time in range(time_span):
                        history_start_idx = len_history + time - self.lag
                        inst_end_idx = len_history + time + 1
                        history = X_all_[:, history_start_idx : inst_end_idx - 1] # batch*samples_per_graph, lag, node
                        W_adj = self.get_weighted_adj_matrix(
                            x_history=history[:N_history_batch],
                            do_round=most_likely_graph,
                            samples=1,
                            most_likely_graph=most_likely_graph,
                        ).view(-1, self.lag+1,proc_dim,proc_dim)  #[ batch, lag+1, nodes, nodes]
                        # import pdb
                        # pdb.set_trace()
                        W_adj=W_adj.repeat(samples_per_graph_groups,1,1,1)


                        # Add intervention logic here.
                        if intervention_mask is not None and intervention_values is not None:
                            # Assign the intervention values to the corresponding indices in X_all, specified by intervention_mask
                            X_all_[..., intervention_mask] = intervention_values
                        # Generate the observations based on the history (given history + previously-generated observations)
                        # and exogenous noise Z.
                        generated_observations = self.ICGNN.f.feed_forward(X_all_[:, history_start_idx : inst_end_idx], W_adj=W_adj) + Z[:, time, :] # batch*samples_per_graph, lag, node

                        # Logic for processing discrete and binary variables. This is similar to static version.
                        if gumbel_max_regions is not None:
                            for region in gumbel_max_regions:
                                maxes = generated_observations[:, region].max(-1, keepdim=True)[0]  # shape [batch_size, 1]
                                generated_observations[:, region] = (generated_observations[:, region] >= maxes).float()
                        if gt_zero_region is not None:
                            generated_observations[:, gt_zero_region] = (generated_observations[:, gt_zero_region] > 0).float()
                        X_all[:, len_history + time, :] = generated_observations
                    if intervention_mask is not None and intervention_values is not None:
                        # assign intervention_values to X_all at corresponding index specified by intervention_mask
                        X_all_[..., intervention_mask] = intervention_values

                    X_simulate = X_all_[:, len_history:, :].clone()  # shape [batch_size, time_span, proc_dim]
                    X_all_total.append(X_all_)
                    X_simulate_total.append(X_simulate)
                

     
                samples = (
                    torch.cat(X_simulate_total, dim=0).view(Nsamples, N_history_batch, time_span, proc_dim)
                )  # shape [Nsamples, N_history_batch, time_span, proc_dim]

            # else:

            #     # Sample weighted graph from posterior with shape [num_graph_samples, lag+1, num_nodes, num_nodes]
            #     # Repeat to shape [N_samples*N_history_batch, lag+1, node, node]
            #     if most_likely_graph:
            #         W_adj_samples = W_adj_samples.expand(
            #             Nsamples * N_history_batch, -1, -1, -1
            #         )  # shape [Nsamples*N_history_batch, lag+1, node, node]
            #     else:
            #         W_adj_samples = torch.repeat_interleave(
            #             W_adj_samples, repeats=int(samples_per_graph), dim=0
            #         ).repeat(
            #             N_history_batch, 1, 1, 1
            #         )  # [Nsamples*N_history_batch, lag+1, node, node]

            #     # repeat X_history to shape [N_history_batch*N_sample, hist_length, proc_dim]
            #     X_history_rep = torch.repeat_interleave(
            #         X_history, repeats=Nsamples, dim=0
            #     )  # [N_history_batch*Nsamples, hist_len, proc_dim]
            #     # Sample noise from base distribution with shape [N_history_batch*Nsamples, time_span, proc_dims]
            #     Z = self._sample_base(Nsamples * N_history_batch, time_span=time_span)
            #     X = []
            #     for W_adj_batch, Z_batch, X_history_batch in zip(
            #         torch.split(W_adj_samples, max_batch_size, dim=0),
            #         torch.split(Z, max_batch_size, dim=0),
            #         torch.split(X_history_rep, max_batch_size, dim=0),
            #     ):
            #         # W_adj_batch, Z_batch have shape [max_batch, lag+1, node, node] and [max_batch, time_span, proc_dim]
            #         X.append(
            #             self.ICGNN.simulate_SEM(
            #                 Z_batch,
            #                 W_adj_batch,
            #                 X_history_batch,
            #                 gumbel_max_regions,
            #                 gt_zero_region,
            #                 intervention_mask=intervention_mask,
            #                 intervention_values=intervention_values,
            #             )[
            #                 1
            #             ].detach()  # shape[max_batch, time_span, proc_dim]
            #         )
            #     samples = torch.cat(X, dim=0).view(
            #         N_history_batch, Nsamples, time_span, proc_dim
            #     )  # shape [N_history_batch, Nsamples, time_span, proc_dim]

        return samples

    def log_prob(
        self,
        X: torch.Tensor,
        Nsamples_per_graph: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        This computes the log probability of the observations. For V0, does not support intervention.
        Most part is just a copy of parent method, the only difference is that for "conditional_spline", we need to pass
        W to self._log_prob.
        Args:
            X: The observation with shape [N_batch, lag+1, proc_dims]
            Nsamples_per_graph: The number of graph samples.
            most_likely_graph: whether to use the most likely graph. If true, Nsamples should be 1.
            intervention_idxs: Currently not support
            intervention_values: Currently not support

        Returns: a numpy with shape [N_batch]

        """
        # Assert for X shape
        assert X.dim() == 3, "X should be of shape [N_batch, lag+1, proc_dims]"
        # Assertions: intervention_idxs and intervention_values must be None for V0.
        assert intervention_idxs is None, "intervention_idxs is not supported for V0"
        assert intervention_values is None, "intervention_values is not supported for V0"
        # Assertions: Nsamples_per_graph must be 1 if most_likely_graph is true.
        if most_likely_graph:
            assert Nsamples_per_graph == 1, "Nsamples_per_graph should be 1 if most_likely_graph is true"

        return super().log_prob(
            X=X,
            Nsamples_per_graph=Nsamples_per_graph,
            most_likely_graph=most_likely_graph,
            intervention_idxs=intervention_idxs,
            intervention_values=intervention_values,
        )

    def get_params_variational_distribution(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For V0, we do not support missing values, so there is no imputer. Raise NotImplementedError for now.

        """
        raise NotImplementedError

    def impute_and_compute_entropy(self, x: torch.Tensor, mask: torch.Tensor):
        """
        For V0, we do not support missing values, so there is no imputer. Raise NotImplementedError for now.
        """
        raise NotImplementedError

    def impute(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        impute_config_dict: Optional[Dict[str, int]] = None,
        *,
        vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        average: bool = True,
    ) -> np.ndarray:
        """
        For V0, we do not support missing values, so there is no imputer. Raise NotImplementedError for now.
        """
        raise NotImplementedError

    def process_dataset(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        variables: Optional[Variables] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method process the dataset using dataprocesor. The implementation is identical to the one in DECI, but we add
        additional check for mask to make sure there is no missing value for V0. Will delete this overwrite in the future if
        we have the support for missing values.
        """
        # Call super().process_dataset to generate data and mask
        data, mask = super().process_dataset(dataset, train_config_dict, variables)
        # Assert mask is all 1.
        assert np.all(mask == 1)
        return data, mask

    def _create_dataset_for_deci(
        self, dataset: TemporalDataset, train_config_dict: Dict[str, Any]
    ) -> Tuple[DataLoader, int]:
        """
        This creates a dataloader for AR-DECI to load temporal tensors. It also returns the size of the training data.

        Args:
            dataset: the training dataset.
            train_config_dict: the training config dict.

        Returns:
            dataloader: A dataloader that supports loading temporal data with shape [N_batch, lag+1, proc_dims].
            num_samples: The size of the training data set.
        """
        # The implementation is identical to the one in FT-DECI but with is_autoregressive=True.
        data, mask = self.process_dataset(dataset, train_config_dict)
        tensor_dataset = TemporalTensorDataset(
            *to_tensors(data, mask, device=self.device),
            lag=self.lag,
            is_autoregressive=True,
            index_segmentation=dataset.train_segmentation,
        )
        dataloader = DataLoader(tensor_dataset, batch_size=train_config_dict["batch_size"], shuffle=True)

        return dataloader, len(tensor_dataset)

    def my_data_loader(self, data, batch_size, index_segmentation):
        tensor_dataset = MyTemporalTensorDataset(
            *to_tensors(data, device=self.device),
            lag=self.lag,
            is_autoregressive=True,
            index_segmentation=index_segmentation,
        )
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    #change
    def _create_dataset_for_sample(
        self, dataset: TemporalDataset, infer_config_dict: Dict[str, Any]
    ) -> Tuple[DataLoader, int]:
        """
        This creates a dataloader for AR-DECI to load temporal tensors. It also returns the size of the training data.

        Args:
            dataset: the training dataset.
            train_config_dict: the training config dict.

        Returns:
            dataloader: A dataloader that supports loading temporal data with shape [N_batch, lag+1, proc_dims].
            num_samples: The size of the training data set.
        """
        # The implementation is identical to the one in FT-DECI but with is_autoregressive=True.
        dataloader = {
            'train':{},
            'val':{},
            'test':{}
        }
        batch_size = infer_config_dict['batch_size']
        processed_dataset = self.data_processor.process_dataset(dataset)
        train_data, _ = processed_dataset.train_data_and_mask
        val_data, _ = processed_dataset.val_data_and_mask
        test_data, _ = processed_dataset.test_data_and_mask
        dataloader['train']['data_len'] = len(train_data)
        dataloader['train']['loader'] = self.my_data_loader(train_data, batch_size, dataset.train_segmentation)
        
        if val_data is not None:
            dataloader['val']['data_len'] = len(val_data)
            dataloader['val']['loader'] = self.my_data_loader(val_data, batch_size, dataset._val_segmentation)
        if test_data is not None:
            dataloader['test']['data_len'] = len(test_data)
            dataloader['test']['loader'] = self.my_data_loader(test_data, batch_size, dataset._test_segmentation)
        return dataloader

    def set_prior_A(
        self,
        prior_A: Optional[Union[np.ndarray, torch.Tensor]],
        prior_mask: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> None:
        """
        Overwrite its parent method.
        Set the soft priors for AR-DECI. The prior_A is a soft prior in the temporal format with shape [lag+1, num_nodes, num_nodes].
        prior_mask is a binary mask for soft prior, where mask_{ij}=1 means the value in prior_{ij} set to be the prior,
        and prior_{ij}=0 means we ignore the value in prior_{ij}.
        If prior_A = None, the default choice is zero matrix with shape [lag+1, num_nodes, num_nodes]. The mask is the same.
        Args:
            prior_A: a soft prior with shape [lag+1, num_nodes, num_nodes].
            prior_mask: a binary mask for soft prior with the same shape as prior_A

        Returns: No return

        """
        if prior_A is not None:
            assert prior_mask is not None, "prior_mask cannot be None for nonempty prior_A"
            assert (
                len(prior_A.shape) == 3
            ), "prior_A must be a temporal soft prior with shape [lag+1, num_nodes, num_nodes]"
            assert (
                prior_A.shape == prior_mask.shape
            ), f"prior_A ({prior_A.shape}) must match the shape of prior_mask ({prior_mask.shape})"
            # Assert prior_A[0,...] diagonal is 0 to avoid non-DAG.
            assert all(prior_A[0, ...].diagonal() == 0), "The diagonal element of instant matrix in prior_A must be 0."

            assert (
                self.lag == prior_A.shape[0] - 1
            ), f"The lag of the model ({self.lag}) must match the lag of prior adj matrix ({prior_A.shape[0] - 1})"

            # Set the prior_A and prior_mask
            self.prior_A = nn.Parameter(
                torch.as_tensor(prior_A, device=self.device, dtype=torch.float32),
                requires_grad=False,
            )
            self.prior_mask = nn.Parameter(
                torch.as_tensor(prior_mask, device=self.device, dtype=torch.float32),
                requires_grad=False,
            )
            self.exist_prior = True
        else:
            self.exist_prior = False
            # Default prior_A and mask
            self.prior_A = nn.Parameter(
                torch.zeros((self.lag + 1, self.num_nodes, self.num_nodes), device=self.device),
                requires_grad=False,
            )
            self.prior_mask = nn.Parameter(
                torch.zeros((self.lag + 1, self.num_nodes, self.num_nodes), device=self.device),
                requires_grad=False,
            )

    def _create_val_dataset_for_deci(
        self, dataset: TemporalDataset, train_config_dict: Dict[str, Any]
    ) -> Tuple[Union[DataLoader, TemporalTensorDataset], int]:

        processed_dataset = self.data_processor.process_dataset(dataset)
        val_data, val_mask = processed_dataset.val_data_and_mask
        val_dataset = TemporalTensorDataset(
            *to_tensors(val_data, val_mask, device=self.device),
            lag=self.lag,
            is_autoregressive=True,
            index_segmentation=dataset.get_val_segmentation(),
        )
        val_dataloader = DataLoader(val_dataset, batch_size=train_config_dict["batch_size"], shuffle=True)

        return val_dataloader, len(val_dataset)

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        This method implements the training scripts of AR-DECI. This also setup a soft prior (if exists) for the AR-DECI.
        We should also change the termination condition rather than the using num_below_tol in DECI,
        since if we set allow_instantaneous=False, AR-DECI will always be a DAG. By default,
        only 5 training epochs (num_below_tol>=5, it will break the training loop in DECI).
        The evaluation for loss tracker should also be adapted so that it supports temporal adjacency matrix.
        This can be done by calling convert_to_static in FT-DECI to transform the temporal adj to static adj, and re-using the evaluation for DECI.
        Compared to DECI training, the differences are (1) setup a soft prior for AR-DECI, (2) different stopping creterion, (3) evaluation of temporal adjacency matrix for loss tracker.
        Args:
            dataset: the training temporal dataset.
            train_config_dict: the training config dict.
            report_progress_callback: the callback function to report progress.
            run_context: the run context.

        Returns: No return
        """
        # Load the soft prior from dataset if exists and update the prior for AR-DECI by calling self.set_prior_A(...).
        assert isinstance(dataset, TemporalDataset)
        # Setup the logging machinery (similar to DECI training).
        # Setup the optimizer (similar to DECI training).
        # Outer optimization loop. Note the termination condition should be changed based on the value we set for allow_instantaneous.
        # Inner loop by calling self.optimize_inner_auglag(...). No change is needed.
        # Update rho, alpha, loss tracker (similar to DECI).
        assert train_config_dict["max_p_train_dropout"] == 0.0, "Current AR-DECI does not support missing values."
        super().run_train(
            dataset,
            train_config_dict=train_config_dict,
            report_progress_callback=report_progress_callback,
        )

    def run_inference(self, dataset, infer_config_dict):
        dataloader = self._create_dataset_for_sample(dataset, infer_config_dict)
        total_metric={
            'train':{},
            'val':{},
            'test':{}
        }
        for k, item in dataloader.items():
            total_metric[k]['data_num'] = item['data_len']
            metric={}
            mse_sample = None
            acc_sample = None
            acc_mask = None
            for x_, target_ in item['loader']:
                x=x_[0] #batch lag nodes
                pre_value = x[:,-1].clone()
                #normalize
                x_mean = x.mean(1).unsqueeze(1) #batch 1 nodes
                x_std = x.std(1).unsqueeze(1)
                x = (x-x_mean)/x_std
                target=target_[0] # batch time_span nodes
                predicts = self.sample(X_history=x,
                                    Nsamples=infer_config_dict['Nsamples'],
                                    most_likely_graph=infer_config_dict['most_likely_graph'],
                                    intervention_idxs=infer_config_dict['intervention_idxs'],
                                    intervention_values=infer_config_dict['intervention_values'],
                                    samples_per_graph_groups=infer_config_dict['samples_per_graph_groups'],
                                    time_span = infer_config_dict['time_span'])
                #shape [nsamples, batch, time_span, nodes]

                # renormalize
                predicts = predicts*x_std +x_mean

                mse_sample_ = (target-predicts).pow(2)
                mse_sample = torch.cat([mse_sample,mse_sample_], dim=1) if mse_sample!=None else mse_sample_

                if infer_config_dict['movement'] == True:
                    move = target[:, 0]-pre_value
                    move_target = torch.where(move>0, 1, 0)

                    predict_move = predicts[:,:,0]-pre_value  #sample batch nodes
                    move_predict  = torch.where(predict_move>0, 1, 0) #sample batch nodes
                    acc_sample_ = (move_target == move_predict).float()
                    acc_sample = torch.cat([acc_sample,acc_sample_], dim=1) if acc_sample!=None else acc_sample_
            mse_sample = mse_sample.mean([1,2])
            
            # 每个节点的mse均值和采样方差
            metric['mse_mean_pre_node'] = mse_sample.mean(0).tolist()
            metric['mse_var_pre_node'] = mse_sample.var(0).tolist()
            # 所有节点的mse均值和采样方差
            metric['mse_mean'] = mse_sample.mean(1).mean(0).tolist()
            metric['mse_var'] = mse_sample.mean(1).var(0).tolist()
            if infer_config_dict['movement'] == True:
                acc_sample = acc_sample.mean(1)
                metric['acc_mean_per_node'] = acc_sample.mean(0).tolist()
                metric['acc_var_per_node'] =acc_sample.var(0).tolist()
                metric['acc_mean'] = acc_sample.mean(1).mean(0).tolist()
                metric['acc_var'] = acc_sample.mean(1).var(0).tolist()
                total_metric[k]['metric']=metric

                    


        return total_metric



    def get_adj_matrix_tensor(
        self, x_history, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False
    ) -> torch.Tensor:
        if self.mode_adjacency == "learn":
            if most_likely_graph:
                assert samples == 1, "When passing most_likely_graph, only 1 sample can be returned."
                A_samples = [self.var_dist_A.get_adj_matrix(x_history, do_round=do_round)]
            else:
                A_samples = [self.var_dist_A.sample_A(x_history) for _ in range(samples)]
                if do_round:
                    A_samples = [A.round() for A in A_samples]
            adj = torch.stack(A_samples, dim=0)
        elif self.mode_adjacency == "upper":
            adj = (
                torch.triu(torch.ones(self.num_nodes, self.num_nodes), diagonal=1)
                .to(self.device)
                .expand(samples, -1, -1)
            )
        elif self.mode_adjacency == "lower":
            adj = (
                torch.tril(torch.ones(self.num_nodes, self.num_nodes), diagonal=-1)
                .to(self.device)
                .expand(samples, -1, -1)
            )
        else:
            raise NotImplementedError(f"Adjacency mode {self.mode_adjacency} not implemented")
        return self._apply_constraints(adj)


    def get_adj_matrix(
        self,
        x_history,
        do_round: bool = True,
        samples: int = 100,
        most_likely_graph: bool = False,
        squeeze: bool = False,
    ) -> np.ndarray:
        """
        Returns the adjacency matrix (or several) as a numpy array.
        """
        adj_matrix = self.get_adj_matrix_tensor(x_history, do_round, samples, most_likely_graph)

        if squeeze and samples == 1:
            adj_matrix = adj_matrix.squeeze(0)
        # Here we have the cast to np.float64 because the original type
        # np.float32 has some issues with json, when saving causality results
        # to a file.
        return adj_matrix.detach().cpu().numpy().astype(np.float64)


    def get_weighted_adj_matrix(
        self,
        x_history, 
        do_round: bool = True,
        samples: int = 100,
        most_likely_graph: bool = False,
        squeeze: bool = False,
    ) -> torch.Tensor:
        """
        Returns the weighted adjacency matrix (or several) as a numpy array.
        """
        A_samples = self.get_adj_matrix_tensor(x_history, do_round, samples, most_likely_graph)

        W_adjs = A_samples * self.ICGNN.get_weighted_adjacency().unsqueeze(0)

        if squeeze and samples == 1:
            W_adjs = W_adjs.squeeze(0)

        return W_adjs
    

    def _ELBO_terms(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes all terms involved in the ELBO.

        Args:
            X: Batched samples from the dataset, size (batch_size, lag+1, input_dim).

        Returns:
            Dict[key, torch.Tensor] containing all the terms involved in the ELBO.
        """
        # Get adjacency matrix with weights

        # normalize
        x_history = X[:, :-1] #batch lag nodes
        x_mean = x_history.mean(1).unsqueeze(1) #batch 1 nodes
        x_std = x_history.std(1).unsqueeze(1) #batch 1 nodes
        X = (X-x_mean)/x_std


        A_sample = self.get_adj_matrix_tensor(x_history=X[:, :-1], do_round=False, samples=1, most_likely_graph=False).squeeze(0)
        #[batch, lag+1, nodes, nodes]
        if self.mode_adjacency == "learn":
            factor_q = 1.0
        elif self.mode_adjacency in ["upper", "lower"]:
            factor_q = 0.0
        else:
            raise NotImplementedError(f"Adjacency mode {self.mode_adjacency} not implemented")
 
        W_adj = A_sample * self.ICGNN.get_weighted_adjacency()  #[batch, lag+1, nodes, nodes]
        predict = self.ICGNN.predict(X, W_adj)
        log_p_A = self._log_prior_A(A_sample)  # A number

        
        log_p_base = self._log_prob(
                X,
                predict,
                W=W_adj if self.base_distribution_type == "conditional_spline" else None,
            )  # (B)

            # self.ICGNN.predict(X, W_adj)
        log_q_A = self.var_dist_A.entropy()  # A number

        cts_mse = self._icgnn_cts_mse(X, predict)  # (B)

        return {
            "log_p_A": log_p_A,
            "log_p_base": log_p_base,
            "log_q_A": log_q_A * factor_q,
            "cts_mse": cts_mse,
        }


    def compute_loss(
        self,
        step: int,
        x: torch.Tensor,
        mask_train_batch: torch.Tensor,
        input_mask: torch.Tensor,
        num_samples: int,
        tracker: Dict,
        train_config_dict: Dict[str, Any],
        alpha: float = None,
        rho: float = None,
        adj_true: Optional[np.ndarray] = None,
        compute_cd_fscore: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Computes the loss and updates trackers of different terms.

        Args:
            step: Inner auglag step.
            x: Input data of shape (batch_size, input_dim).
            mask_train_batch: Mask indicating which values are missing in the dataset of shape (batch_size, input_dim).
            input_mask: Mask indicating which additional values are aritificially masked.
            num_samples: Number of samples used to compute a stochastic estimate of the loss.
            tracker: Tracks terms in the loss during the inner auglag optimisation.
            train_config_dict: Contains training configuration.
            alpha: Parameter used to scale the penalty_dag term in the prior. Defaults to None.
            rho: Parameter used to scale the penalty_dag term in the prior. Defaults to None.
            adj_true: ground truth adj matrix for tracking causal discovery performance in inner loops
            compute_cd_fscore: whether to compute `cd_fscore` metric at each step. Warning: this may have negative
                side-effects on speeed and GPU utilization.

        Returns:
            Tuple containing the loss and the tracker.
        """
        _ = kwargs


        #  Compute remaining terms
        elbo_terms = self._ELBO_terms(x)
        log_p_term = elbo_terms["log_p_base"].mean(dim=0)
        log_p_A_term = elbo_terms["log_p_A"] / num_samples
        log_q_A_term = elbo_terms["log_q_A"] / num_samples
        cts_mse = elbo_terms["cts_mse"].mean(dim=0)
        cts_medse, _ = torch.median(elbo_terms["cts_mse"], dim=0)


        if train_config_dict["anneal_entropy"] == "linear":
            ELBO = log_p_term -  log_p_A_term - log_q_A_term / max(step - 5, 1) 
        elif train_config_dict["anneal_entropy"] == "noanneal":
            ELBO = log_p_term - log_p_A_term - log_q_A_term 
        loss = -ELBO


        tracker["loss"].append(loss.item())
        
        loss = loss +  cts_mse

        tracker["log_p_A_sparse"].append(log_p_A_term.item())
        tracker["log_p_x"].append(log_p_term.item())
        tracker["log_q_A"].append(log_q_A_term.item())
        tracker["cts_mse_icgnn"].append(cts_mse.item())
        tracker["cts_medse_icgnn"].append(cts_medse.item())
        return loss, tracker


