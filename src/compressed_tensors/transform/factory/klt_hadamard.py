# SPDX-License-Identifier: Apache-2.0

"""
KLT-Enhanced Rotation Transform Factory

Implements the Karhunen-Loève Transformation (KLT) enhanced rotation from:
    MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods
    (Xu et al., ICLR 2025, arXiv:2501.13484)

The key insight is that a standard Hadamard rotation does not equalize channel
variances (Eq. 9 in the paper). KLT-Enhanced rotation fixes this by:

    H_K = K @ H                                             (Eq. 11)

where K is the eigenvector matrix from the eigenvalue decomposition of the
covariance matrix of calibration data, and H is the Hadamard matrix.

After this combined rotation, every channel's variance becomes identical:

    (C_{X H_K})_{ll} = 1/((n-1)*m) * sum_j(lambda_j)       (Eq. 13)

This makes quantization significantly easier for models like Mamba, where
Parallel Scan amplifies outlier channels and causes heavy-tailed distributions.

Usage with QuIP modifier (llm-compressor):
    The KLT factory is registered as "klt-hadamard" and can be used anywhere
    the standard "hadamard" factory is used. It additionally requires
    calibration data to compute the eigenvector matrix K.

    Example TransformScheme:
        TransformScheme(
            type="klt-hadamard",
            apply=[
                TransformArgs(targets="Linear", location="weight_input", ...),
                TransformArgs(targets="Linear", location="input", ...),
            ],
            precision=torch.float64,
        )

    Before applying to the model, calibration data must be provided:
        factory = TransformFactory.from_scheme(scheme, name="klt")
        factory.set_calibration_data(calibration_data_dict)
        factory.apply_to_model(model)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from compressed_tensors.transform import TransformArgs, TransformScheme
from compressed_tensors.transform.factory.base import TransformBase, TransformFactory
from compressed_tensors.transform.utils.hadamard import (
    deterministic_hadamard_matrix,
    is_pow2,
    random_hadamard_matrix,
)
from compressed_tensors.transform.utils.matrix import (
    apply_transform_weight,
    get_transform_size,
)
from compressed_tensors.utils import get_execution_device, get_offloaded_device
from compressed_tensors.utils.helpers import ParameterizedDefaultDict
from torch import Tensor, device, dtype
from torch.nn import Module, Parameter


__all__ = ["KLTHadamardFactory", "KLTHadamardTransform"]


def compute_klt_matrix(
    calibration_data: Tensor,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Compute the KLT (Karhunen-Loève Transform) eigenvector matrix K from
    calibration data.

    Given centered data X of shape (n, m), the covariance is:
        C_X = (1/(n-1)) * X^T X = K Λ K^T          (Eq. 10)

    where K contains the eigenvectors and Λ the eigenvalues.

    :param calibration_data: tensor of shape (n_samples, feature_dim) or
        (n_samples, ..., feature_dim). The last dimension is treated as the
        channel/feature dimension. All leading dimensions are flattened into
        the sample dimension.
    :param dtype: precision for the eigendecomposition
    :param device: device for computation
    :return: eigenvector matrix K of shape (m, m), where m = feature_dim
    """
    data = calibration_data.to(dtype=dtype, device=device)

    # Flatten all dims except the last into the sample dimension
    if data.ndim > 2:
        data = data.reshape(-1, data.shape[-1])

    # Center the data (zero-mean columns)
    data = data - data.mean(dim=0, keepdim=True)

    n, m = data.shape

    # Covariance matrix: C_X = (1/(n-1)) * X^T X
    # For numerical stability, use the full covariance computation
    cov = (data.T @ data) / (n - 1)

    # Eigenvalue decomposition: C_X = K Λ K^T
    # torch.linalg.eigh returns eigenvalues in ascending order and
    # guarantees K is orthogonal for symmetric (Hermitian) matrices
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # K is the eigenvector matrix (columns are eigenvectors)
    return eigenvectors


def compute_klt_hadamard_matrix(
    calibration_data: Tensor,
    size: int,
    hadamard_dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    gen: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Compute the KLT-Enhanced rotation matrix H_K = K @ H (Eq. 11).

    This combined matrix:
    1. First applies KLT to decorrelate channels (diagonalize covariance)
    2. Then applies Hadamard to equalize max values

    The result equalizes both max values AND variances across channels.

    :param calibration_data: calibration data for computing K, shape
        (..., feature_dim)
    :param size: size of the transform matrix (must match feature_dim of
        calibration data or be a divisor for block-diagonal application)
    :param hadamard_dtype: dtype for Hadamard matrix construction
    :param device: device for computation
    :param gen: optional random generator for random-hadamard variant
    :return: KLT-enhanced rotation matrix H_K of shape (size, size)
    """
    # Compute eigenvector matrix K from calibration data
    K = compute_klt_matrix(calibration_data, dtype=hadamard_dtype, device=device)

    assert K.shape[0] == size and K.shape[1] == size, (
        f"Calibration data feature dimension ({K.shape[0]}) must match "
        f"transform size ({size})"
    )

    # Construct Hadamard matrix H
    if is_pow2(size):
        H = deterministic_hadamard_matrix(size, hadamard_dtype, device)
    else:
        H = random_hadamard_matrix(size, hadamard_dtype, device, gen)

    # KLT-Enhanced rotation: H_K = K @ H  (Eq. 11)
    H_K = K @ H

    return H_K


@TransformFactory.register("klt-hadamard")
class KLTHadamardFactory(TransformFactory):
    """
    Factory for KLT-Enhanced Hadamard transforms (MambaQuant, Xu et al. 2025).

    Combines the Karhunen-Loève Transform with Hadamard rotation to achieve
    both max-value balancing and variance equalization across channels, which
    is critical for quantizing Mamba models where Parallel Scan amplifies
    channel outliers.

    Unlike the standard HadamardFactory, this factory requires calibration
    data to compute the KLT eigenvector matrix. Calibration data should be
    provided via `set_calibration_data()` before calling `apply_to_model()`.

    The calibration data dict maps transform sizes to calibration tensors:
        {
            768: tensor_of_shape_(n_samples, 768),
            3072: tensor_of_shape_(n_samples, 3072),
        }

    Alternatively, `register_calibration_hook()` can be used to collect
    calibration data automatically during a forward pass.

    :param name: name associated with transform scheme
    :param scheme: transform scheme which defines how transforms should be
        created
    :param seed: random seed for Hadamard randomization
    """

    def __init__(self, name: str, scheme: TransformScheme, seed: int | None = None):
        super().__init__(name, scheme, seed)
        self.weights = ParameterizedDefaultDict(self._create_weight)
        self.perms = ParameterizedDefaultDict(self._create_permutation)

        # Calibration data: maps transform_size -> calibration tensor
        self._calibration_data: Dict[int, Tensor] = {}

        # Collection hooks for automatic calibration gathering
        self._hooks: list = []
        self._collected_activations: Dict[int, list[Tensor]] = {}

    def set_calibration_data(self, data: Dict[int, Tensor]) -> None:
        """
        Provide pre-computed calibration data for KLT computation.

        :param data: dictionary mapping transform size -> calibration tensor
            of shape (n_samples, size) or (n_samples, ..., size)
        """
        self._calibration_data = data
        # Invalidate cached weights since calibration data changed
        self.weights = ParameterizedDefaultDict(self._create_weight)

    def set_calibration_data_for_size(self, size: int, data: Tensor) -> None:
        """
        Provide calibration data for a specific transform size.

        :param size: the feature dimension / transform size
        :param data: calibration tensor of shape (n_samples, size) or
            (n_samples, ..., size)
        """
        self._calibration_data[size] = data
        # Remove cached weight for this size so it gets recomputed
        if size in self.weights:
            del self.weights[size]

    def register_calibration_hooks(
        self, model: Module, args_list: list[TransformArgs]
    ) -> list:
        """
        Register forward hooks on target modules to collect activation data
        for KLT calibration. Run calibration data through the model after
        calling this, then call `finalize_calibration()`.

        :param model: the model to hook
        :param args_list: list of TransformArgs defining which modules to hook
        :return: list of hook handles for removal
        """
        from compressed_tensors.utils import match_named_modules

        handles = []
        for args in args_list:
            for name, module in match_named_modules(
                model, args.targets, args.ignore
            ):
                size = get_transform_size(
                    module, args.location, self.scheme.head_dim
                )

                if size not in self._collected_activations:
                    self._collected_activations[size] = []

                def _make_hook(sz):
                    def hook_fn(mod, input, output):
                        # Collect input activations for input-side transforms
                        if args.location in (
                            TransformLocation.INPUT,
                            TransformLocation.WEIGHT_INPUT,
                        ):
                            act = input[0] if isinstance(input, tuple) else input
                        else:
                            act = output
                        # Detach and move to CPU to save GPU memory
                        self._collected_activations[sz].append(
                            act.detach().cpu()
                        )

                    return hook_fn

                handle = module.register_forward_hook(_make_hook(size))
                handles.append(handle)

        self._hooks = handles
        return handles

    def finalize_calibration(self) -> None:
        """
        Finalize calibration by concatenating collected activations and
        computing KLT matrices. Call this after running calibration data
        through hooked model.
        """
        for size, act_list in self._collected_activations.items():
            if act_list:
                # Concatenate along sample dimension and flatten to 2D
                combined = torch.cat(act_list, dim=0)
                if combined.ndim > 2:
                    combined = combined.reshape(-1, combined.shape[-1])
                self._calibration_data[size] = combined

        # Clean up
        self._collected_activations.clear()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        # Invalidate cached weights
        self.weights = ParameterizedDefaultDict(self._create_weight)

    def create_transform(self, module: Module, args: TransformArgs):
        """
        Create a KLTHadamardTransform for applying to a module. Transforms
        with the same size, dtype, and device are cached.

        :param module: parent module that transform will be applied to
        :param args: defines how the transform will be applied to the module
        :return: KLTHadamardTransform instance
        """
        size = get_transform_size(module, args.location, self.scheme.head_dim)
        exec_device = get_execution_device(module)
        off_device = get_offloaded_device(module)
        precision = self.scheme.precision if args.is_online() else torch.float64

        factory_kwargs = {
            "device": off_device,
            "construct_device": exec_device,
            "precision": precision,
        }
        weight = self.weights.get(size, factory_kwargs=factory_kwargs)
        perm = self.perms[weight] if self.scheme.randomize else None
        return KLTHadamardTransform(weight, perm, self.scheme, args, type(module))

    def _create_weight(
        self,
        size: int,
        device: device,
        construct_device: device,
        precision: dtype,
    ) -> Parameter:
        """
        Create the KLT-Enhanced rotation matrix H_K = K @ H.

        If calibration data is available for the given size, compute the full
        KLT-enhanced matrix. Otherwise, fall back to a standard Hadamard
        matrix (with a warning).

        :param size: dimension of the transform matrix
        :param device: target device for the parameter
        :param construct_device: device to use during construction
        :param precision: dtype for computation
        :return: Parameter containing the KLT-enhanced rotation matrix
        """
        if size in self._calibration_data:
            data = compute_klt_hadamard_matrix(
                calibration_data=self._calibration_data[size],
                size=size,
                hadamard_dtype=precision,
                device=construct_device,
                gen=self.generator if self.scheme.randomize else None,
            )
        else:
            import warnings

            warnings.warn(
                f"No calibration data available for transform size {size}. "
                f"Falling back to standard Hadamard matrix. Provide calibration "
                f"data via set_calibration_data() for KLT-enhanced rotation.",
                stacklevel=2,
            )
            if is_pow2(size):
                data = deterministic_hadamard_matrix(
                    size, precision, construct_device
                )
            else:
                data = random_hadamard_matrix(
                    size, precision, construct_device, self.generator
                )

        data = data.to(device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)

    def _create_permutation(self, weight: Parameter) -> Parameter:
        data = torch.randperm(weight.size(0), generator=self.generator)
        return Parameter(data, requires_grad=False)


class KLTHadamardTransform(TransformBase):
    """
    Applies the KLT-Enhanced rotation H_K (or its inverse) to a value.

    The forward pass is identical to HadamardTransform — the KLT enhancement
    is baked into the weight matrix at construction time. The key difference
    is that H_K = K @ H is orthogonal but NOT symmetric (unlike pure
    Hadamard), so the inverse is H_K^T (the transpose), not H_K itself.

    Also note: The Hadamard normalization factor 1/sqrt(m) does NOT apply
    to the KLT-enhanced matrix because K is already orthonormal. The
    combined matrix H_K = K @ H is orthogonal up to the Hadamard scaling,
    so we still divide by sqrt(m) to normalize.
    """

    _dynamic_tied_weights_keys: list[str] = ["weight", "perm"]

    def __init__(
        self,
        weight: Parameter,
        perm: Parameter | None,
        scheme: TransformScheme,
        args: TransformArgs,
        module_type: type[torch.nn.Module],
    ):
        super().__init__()
        self.weight = weight
        self.perm = perm
        self.scheme = scheme
        self.args = args
        self.module_type = module_type
        self._scale = torch.tensor(
            weight.size(0), dtype=torch.float64
        ).sqrt()

    def forward(self, value: Tensor) -> Tensor:
        weight = self.weight

        if self.perm is not None:
            weight = weight[self.perm][:, self.perm]

        if self.args.inverse:
            # H_K is orthogonal (up to scale), so inverse = transpose
            # Unlike pure Hadamard, H_K is NOT symmetric, so H_K^T != H_K
            weight = weight.T

        return (
            apply_transform_weight(
                weight.to(device=value.device),
                value.to(dtype=weight.dtype),
                self.args.location,
                self.module_type,
            )
            / self._scale
        ).to(value.dtype)