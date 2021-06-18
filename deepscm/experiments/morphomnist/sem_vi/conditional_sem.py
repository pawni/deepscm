import torch
import pyro
import numpy as np

from pyro.nn import pyro_method
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import (
    ComposeTransform, ExpTransform, Spline, SigmoidTransform
)
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from deepscm.distributions.transforms.affine import ConditionalAffineTransform
from pyro.nn import DenseNN
import deepscm.pyro_addons.messengers

from .base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalVISEM(BaseVISEM):
    context_dim = 2

    def __init__(self, independence_weighting: bool = False, independence_samples: int = 16,
                 weight_z: bool = False, auxiliary_models: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.independence_weighting = independence_weighting
        self.independence_samples = independence_samples
        self.weight_z = weight_z
        self.auxiliary_models = auxiliary_models

        self.intensity_cache = None
        self.thickness_cache = None
        self.weight_cache = None

        # Flow for modelling t Gamma
        self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = ComposeTransform([self.thickness_flow_components, self.thickness_flow_constraint_transforms])

        # affine flow for s normal
        intensity_net = DenseNN(1, [1], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        self.intensity_flow_components = ConditionalAffineTransform(context_nn=intensity_net, event_dim=0)
        self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        self.intensity_flow_transforms = [self.intensity_flow_components, self.intensity_flow_constraint_transforms]

        if self.auxiliary_models:
            self.aux_thickness = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, 3, 2, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, 2, 1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 1)
            )

            self.aux_intensity = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, 3, 2, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, 2, 1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 1)
            )

    @pyro_method
    def pgm_model(self):
        thickness_base_dist = Normal(self.thickness_base_loc, self.thickness_base_scale).to_event(1)
        thickness_dist = TransformedDistribution(thickness_base_dist, self.thickness_flow_transforms)

        thickness = pyro.sample('thickness', thickness_dist)
        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.thickness_flow_components

        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale).to_event(1)
        intensity_dist = ConditionalTransformedDistribution(intensity_base_dist, self.intensity_flow_transforms).condition(thickness_)

        intensity = pyro.sample('intensity', intensity_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.intensity_flow_components

        return thickness, intensity

    @pyro_method
    def model(self):
        thickness, intensity = self.pgm_model()

        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        intensity_ = self.intensity_flow_norm.inv(intensity)

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        latent = torch.cat([z, thickness_, intensity_], 1)

        x_dist = self._get_transformed_x_dist(latent)

        x = pyro.sample('x', x_dist)

        return x, z, thickness, intensity

    @pyro_method
    def aux_model(self, x):
        aux_thickness = pyro.sample('aux_thickness', Normal(self.aux_thickness(x), 1))
        aux_intensity = pyro.sample('aux_intensity', Normal(self.aux_intensity(x), 1))

        return aux_thickness, aux_intensity

    @pyro_method
    def counterfactual_aux_model(self, x, thickness, intensity):
        counterfactual = self.get_training_counterfactual(x, thickness, intensity)

        conditioning = {'aux_thickness': counterfactual['thickness'], 'aux_intensity': counterfactual['intensity']}
        with pyro.plate('observations', x.shape[0]):
            pyro.condition(self.aux_model, data=conditioning)(counterfactual['x'])

    @pyro_method
    def guide(self, x, thickness, intensity):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
            intensity_ = self.intensity_flow_norm.inv(intensity)

            hidden = torch.cat([hidden, thickness_, intensity_], 1)
            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z

    @pyro.poutine.block()
    @torch.no_grad()
    def calc_scale_dict(self, thickness, intensity):
        if self.weight_cache is not None and torch.equal(self.intensity_cache, intensity) and torch.equal(self.thickness_cache, thickness):
            return self.weight_cache

        data_dict = {'thickness': thickness, 'intensity': intensity}
        orig_trace = pyro.poutine.trace(self.conditioned_pgm_model).get_trace(**data_dict)
        orig_trace.compute_log_prob()
        log_prob = orig_trace.nodes['intensity']['log_prob']

        data_dict['thickness'] = torch.stack(
            [thickness[np.random.permutation(thickness.shape[0])] for _ in range(self.independence_samples)])
        data_dict['intensity'] = torch.stack(
            [intensity for _ in range(self.independence_samples)])

        data_dict['thickness'] = data_dict['thickness'].view([-1, 1])
        data_dict['intensity'] = data_dict['intensity'].view([-1, 1])

        marginal_trace = pyro.poutine.trace(self.conditioned_pgm_model).get_trace(**data_dict)
        marginal_trace.compute_log_prob()
        marginal_intensity = marginal_trace.nodes['intensity']['log_prob'].view([self.independence_samples, -1]).mean(0)

        weights = torch.exp(marginal_intensity - log_prob)

        weight_dict = {'x': weights}
        if self.weight_z:
            weight_dict['z'] = weights

        self.thickness_cache = thickness
        self.intensity_cache = intensity
        self.weight_cache = weight_dict

        return weight_dict

    @pyro_method
    def svi_model(self, x, thickness, intensity):
        model_fn = super().svi_model
        if self.auxiliary_models:
            def model_fn(x, thickness, intensity):
                with pyro.plate('observations', x.shape[0]):
                    pyro.condition(self.model, data={'x': x, 'thickness': thickness, 'intensity': intensity})()

                    pyro.condition(self.aux_model, data={'aux_thickness': thickness, 'aux_intensity': intensity})(x)

        if self.independence_weighting:
            scale_dict = self.calc_scale_dict(thickness, intensity)
            deepscm.pyro_addons.messengers.site_scale_messenger(model_fn(x, thickness, intensity), scale_dict=scale_dict)
        else:
            model_fn(x, thickness, intensity)

    @pyro_method
    def svi_guide(self, x, thickness, intensity):
        if self.independence_weighting and self.weight_z:
            scale_dict = self.calc_scale_dict(thickness, intensity)
            deepscm.pyro_addons.messengers.site_scale_messenger(super().svi_guide(x, thickness, intensity), scale_dict=scale_dict)
        else:
            super().svi_guide(x, thickness, intensity)

    @pyro_method
    def infer_thickness_base(self, thickness):
        return self.thickness_flow_transforms.inv(thickness)

    @pyro_method
    def infer_intensity_base(self, thickness, intensity):
        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale)
        cond_intensity_transforms = ComposeTransform(
            ConditionalTransformedDistribution(intensity_base_dist, self.intensity_flow_transforms).condition(thickness).transforms)
        return cond_intensity_transforms.inv(intensity)

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--independence_weighting', default=False, action='store_true', help="Reweights loss to encourage independence.")
        parser.add_argument('--independence_samples', default=16, type=int, help="Number of samples to calculate marginal distribution.")
        parser.add_argument('--weight_z', default=False, action='store_true', help="Reweights loss on z to encourage independence.")
        parser.add_argument('--auxiliary_models', default=False, action='store_true', help="Use auxiliary models to enforce parent values.")
        return parser


MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM
