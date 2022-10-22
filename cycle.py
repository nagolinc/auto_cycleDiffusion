# scripts imports
#from ldm.models.diffusion.ddim import DDIMSampler
from contextlib import contextmanager, nullcontext
from torch import autocast
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from omegaconf import OmegaConf
import glob
import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state,sd_model


#his ddimSampler is different

# Created by Chen Henry Wu
"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        self.register_buffer('ddim_sqrt_one_minus_alphas_prev', np.sqrt(1. - ddim_alphas_prev))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates


    def refine(self,
               S,
               refine_steps,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self._refine(refine_steps, conditioning, size,
                                              callback=callback,
                                              img_callback=img_callback,
                                              quantize_denoised=quantize_x0,
                                              mask=mask, x0=x0,
                                              ddim_use_original_steps=False,
                                              noise_dropout=noise_dropout,
                                              temperature=temperature,
                                              score_corrector=score_corrector,
                                              corrector_kwargs=corrector_kwargs,
                                              log_every_t=log_every_t,
                                              unconditional_guidance_scale=unconditional_guidance_scale,
                                              unconditional_conditioning=unconditional_conditioning,
                                              )
        return samples, intermediates

    def sample_with_eps(self,
                        S,
                        eps_list,
                        batch_size,
                        shape,
                        conditioning=None,
                        callback=None,
                        normals_sequence=None,
                        img_callback=None,
                        quantize_x0=False,
                        eta=0.,
                        mask=None,
                        x0=None,
                        temperature=1.,
                        noise_dropout=0.,
                        score_corrector=None,
                        corrector_kwargs=None,
                        verbose=True,
                        x_T=None,
                        skip_steps=0,
                        log_every_t=100,
                        unconditional_guidance_scale=1.,
                        unconditional_conditioning=None,
                        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
                        **kwargs
                        ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_with_eps(conditioning, size,
                                                             eps_list,
                                                             callback=callback,
                                                             img_callback=img_callback,
                                                             quantize_denoised=quantize_x0,
                                                             mask=mask, x0=x0,
                                                             ddim_use_original_steps=False,
                                                             noise_dropout=noise_dropout,
                                                             temperature=temperature,
                                                             score_corrector=score_corrector,
                                                             corrector_kwargs=corrector_kwargs,
                                                             x_T=x_T,
                                                             skip_steps=skip_steps,
                                                             log_every_t=log_every_t,
                                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                                             unconditional_conditioning=unconditional_conditioning,
                                                             )
        return samples, intermediates

    def ddpm_ddim_encoding(self,
                           S,
                           batch_size,
                           shape,
                           conditioning=None,
                           callback=None,
                           normals_sequence=None,
                           img_callback=None,
                           quantize_x0=False,
                           eta=0.,
                           white_box_steps=None,
                           skip_steps=0,
                           x0=None,
                           temperature=1.,
                           noise_dropout=0.,
                           score_corrector=None,
                           corrector_kwargs=None,
                           verbose=True,
                           log_every_t=100,
                           unconditional_guidance_scale=1.,
                           unconditional_conditioning=None,
                           # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
                           **kwargs
                           ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        assert eta > 0

        z_list = self._ddpm_ddim_encoding(conditioning, size,
                                          callback=callback,
                                          img_callback=img_callback,
                                          quantize_denoised=quantize_x0,
                                          eta=eta, white_box_steps=white_box_steps, skip_steps=skip_steps,
                                          x0=x0,
                                          ddim_use_original_steps=False,
                                          noise_dropout=noise_dropout,
                                          temperature=temperature,
                                          score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          log_every_t=log_every_t,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          )

        return z_list

    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=True)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def _refine(self, refine_steps, cond, shape,
                ddim_use_original_steps=False,
                callback=None, timesteps=None, quantize_denoised=False,
                mask=None, x0=None, img_callback=None, log_every_t=100,
                temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                unconditional_guidance_scale=1., unconditional_conditioning=None):
        device = self.model.betas.device
        b = shape[0]

        # Sample xt
        alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
        at = alphas[refine_steps - 1]
        xt = at.sqrt() * x0 + (1 - at).sqrt() * torch.randn_like(x0)

        img = xt

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps with {refine_steps} refinement steps")

        assert refine_steps < total_steps
        refine_time_range = time_range[-refine_steps:]
        iterator = tqdm(refine_time_range, desc='DDIM Sampler', total=refine_steps, disable=True)

        for i, step in enumerate(iterator):
            index = refine_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == refine_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def ddim_sampling_with_eps(self, cond, shape, eps_list,
                               x_T=None, ddim_use_original_steps=False,
                               callback=None, timesteps=None, skip_steps=0, quantize_denoised=False,
                               mask=None, x0=None, img_callback=None, log_every_t=100,
                               temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                               unconditional_guidance_scale=1., unconditional_conditioning=None, ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:  # x_T is x_t if using skip_steps.
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        refine_steps = total_steps - skip_steps
        print(f"Running DDIM Sampling with {total_steps} timesteps and {refine_steps} refinement steps")

        refine_time_range = time_range[-refine_steps:]
        iterator = tqdm(refine_time_range, desc='DDIM Sampler', total=refine_steps, disable=True)

        for i, step in enumerate(iterator):
            index = refine_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim_with_eps(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                               quantize_denoised=quantize_denoised, temperature=temperature,
                                               noise_dropout=noise_dropout, score_corrector=score_corrector,
                                               corrector_kwargs=corrector_kwargs,
                                               unconditional_guidance_scale=unconditional_guidance_scale,
                                               unconditional_conditioning=unconditional_conditioning,
                                               eps=eps_list[:, i] if i < eps_list.shape[1] else None,
                                               )
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def _ddpm_ddim_encoding(self, cond, shape,
                            ddim_use_original_steps=False,
                            callback=None, timesteps=None, quantize_denoised=False,
                            eta=None, white_box_steps=None, skip_steps=0,
                            x0=None, img_callback=None, log_every_t=100,
                            temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                            unconditional_guidance_scale=1., unconditional_conditioning=None):

        assert eta > 0
        device = self.model.betas.device
        b = shape[0]

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        refine_steps = total_steps - skip_steps
        print(f"Running DDIM Sampling with {total_steps} timesteps and {refine_steps} refinement steps")

        refine_time_range = time_range[-refine_steps:]
        iterator = tqdm(refine_time_range, desc='DDIM Sampler', total=refine_steps, disable=True)

        # Sample xt
        alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
        at = alphas[refine_steps - 1]
        xt = at.sqrt() * x0 + (1 - at).sqrt() * torch.randn_like(x0)
        z_list = [xt, ]

        for i, step in enumerate(iterator):
            index = refine_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if i < white_box_steps - skip_steps - 1:
                xt_next = self.sample_xt_next(x0=x0, xt=xt, index=index, use_original_steps=ddim_use_original_steps)

                eps = self.compute_eps(
                    xt=xt, xt_next=xt_next, c=cond, t=ts, index=index, use_original_steps=ddim_use_original_steps,
                    quantize_denoised=quantize_denoised, temperature=temperature,
                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                    corrector_kwargs=corrector_kwargs,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning)
                xt = xt_next
                z_list.append(eps)
            else:
                break

        return z_list

    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        elif unconditional_guidance_scale == 0:
            e_t = self.model.apply_model(x, t, unconditional_conditioning)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    def compute_eps(self, xt, xt_next, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *xt.shape, xt.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(xt, t, c)
        elif unconditional_guidance_scale == 0:
            e_t = self.model.apply_model(xt, t, unconditional_conditioning)
        else:
            x_in = torch.cat([xt] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, xt, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (xt - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        eps = (xt_next - a_prev.sqrt() * pred_x0 - dir_xt) / sigma_t / temperature
        return eps

    def sample_xt_next(self, x0, xt, index, use_original_steps=False):
        if index == 0:
            return x0

        b, *_, device = *x0.shape, x0.device

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)

        # direction pointing to x_t
        e_t = (xt - a_t.sqrt() * x0) / (1 - a_t).sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn(x0.shape, device=device)
        xt_next = a_prev.sqrt() * x0 + dir_xt + noise
        return xt_next

    def p_sample_ddim_with_eps(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                               temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                               unconditional_guidance_scale=1., unconditional_conditioning=None, eps=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        elif unconditional_guidance_scale == 0:
            e_t = self.model.apply_model(x, t, unconditional_conditioning)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        if eps is None:
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        else:
            noise = sigma_t * eps * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec


# Created by Chen Henry Wu
import os
import argparse
import sys
sys.path.append(os.path.abspath('model/lib/stable_diffusion'))


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def prepare_stable_diffusion_text(source_model_type):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')

    config = OmegaConf.load(os.path.join(
        'model/lib/stable_diffusion/configs/stable-diffusion/v1-inference.yaml'))
    ckpt = os.path.join('ckpts', 'stable_diffusion', source_model_type)

    return config, ckpt


def get_condition(model, text, bs):
    assert isinstance(text, list)
    assert isinstance(text[0], str)

    #print('about to suffer',model,bs)

    uc = model.get_learned_conditioning(bs * [""])
    print("model.cond_stage_key: ", model.cond_stage_key)
    c = model.get_learned_conditioning(text)
    print("c.shape: ", c.shape)
    print('-' * 50)
    return c, uc


def convsample_ddim_conditional(model, steps, shape, x_T, skip_steps, eta, eps_list, scale, text):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    c, uc = get_condition(model, text, bs)
    samples, intermediates = ddim.sample_with_eps(steps,
                                                  eps_list,
                                                  conditioning=c,
                                                  batch_size=bs,
                                                  shape=shape,
                                                  eta=eta,
                                                  verbose=False,
                                                  x_T=x_T,
                                                  skip_steps=skip_steps,
                                                  unconditional_guidance_scale=scale,
                                                  unconditional_conditioning=uc
                                                  )
    return samples, intermediates


def make_convolutional_sample_with_eps_conditional(model, custom_steps, eta, x_T, skip_steps, eps_list,
                                                   scale, text):
    with model.ema_scope("Plotting"):
        sample, intermediates = convsample_ddim_conditional(model,
                                                            steps=custom_steps,
                                                            shape=x_T.shape,
                                                            x_T=x_T,
                                                            skip_steps=skip_steps,
                                                            eta=eta,
                                                            eps_list=eps_list,
                                                            scale=scale,
                                                            text=text)

    x_sample = model.decode_first_stage(sample)

    return x_sample


def ddpm_ddim_encoding_conditional(model, steps, shape, eta, white_box_steps, skip_steps, x0, scale, text):
    with model.ema_scope("Plotting"):
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        c, uc = get_condition(model, text, bs)

        z_list = ddim.ddpm_ddim_encoding(steps,
                                         conditioning=c,
                                         batch_size=bs,
                                         shape=shape,
                                         eta=eta,
                                         white_box_steps=white_box_steps,
                                         skip_steps=skip_steps,
                                         verbose=False,
                                         x0=x0,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         )

    return z_list


class SDStochasticTextWrapper(torch.nn.Module):

    def __init__(self, source_model, custom_steps, eta, white_box_steps, skip_steps,
                 encoder_unconditional_guidance_scales=1, decoder_unconditional_guidance_scales=5,
                 n_trials=1):
        super(SDStochasticTextWrapper, self).__init__()

        self.encoder_unconditional_guidance_scales = encoder_unconditional_guidance_scales
        self.decoder_unconditional_guidance_scales = decoder_unconditional_guidance_scales
        self.n_trials = n_trials

        # Set up generator
        #self.config, self.ckpt = prepare_stable_diffusion_text(source_model_type)

        #print(self.config)

        #self.generator = load_model_from_config(self.config, self.ckpt, verbose=True)

        self.generator = source_model

        self.precision = "full"

        print(75 * "-")

        self.eta = eta
        self.custom_steps = custom_steps
        self.white_box_steps = white_box_steps
        self.skip_steps = skip_steps

        self.resolution = 512
        print(f"resolution: {self.resolution}")

        print(
            f'Using DDIM sampling with {self.custom_steps} sampling steps and eta={self.eta}')

        # Freeze.
        # requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0],
                                  std=[2.0, 2.0, 2.0])]
        )

    def generate(self, z_ensemble, decode_text):
        precision_scope = autocast if self.precision == "autocast" else nullcontext
        with precision_scope("cuda"):
            img_ensemble = []
            for i, z in enumerate(z_ensemble):
                skip_steps = self.skip_steps[i % len(self.skip_steps)]
                bsz = z.shape[0]
                if self.white_box_steps != -1:
                    eps_list = z.view(bsz, (self.white_box_steps - skip_steps),
                                      self.generator.channels, self.generator.image_size, self.generator.image_size)
                else:
                    eps_list = z.view(bsz, 1, self.generator.channels,
                                      self.generator.image_size, self.generator.image_size)
                x_T = eps_list[:, 0]
                eps_list = eps_list[:, 1:]

                for decoder_unconditional_guidance_scale in self.decoder_unconditional_guidance_scales:
                    img = make_convolutional_sample_with_eps_conditional(self.generator,
                                                                         custom_steps=self.custom_steps,
                                                                         eta=self.eta,
                                                                         x_T=x_T,
                                                                         skip_steps=skip_steps,
                                                                         eps_list=eps_list,
                                                                         scale=decoder_unconditional_guidance_scale,
                                                                         text=decode_text)
                    img_ensemble.append(img)

        return img_ensemble

    def encode(self, image, encode_text):
        # Eval mode for the generator.
        self.generator.eval()

        precision_scope = autocast if self.precision == "autocast" else nullcontext

        # Normalize.
        image = (image - 0.5) * 2.0
        # Resize.
        assert image.shape[2] == image.shape[3] == self.resolution
        with precision_scope("cuda"):
            with torch.no_grad():
                # Encode.
                encoder_posterior = self.generator.encode_first_stage(image)
                z = self.generator.get_first_stage_encoding(encoder_posterior)
                x0 = z

        with precision_scope("cuda"):
            bsz = image.shape[0]
            z_ensemble = []
            for trial in range(self.n_trials):
                for encoder_unconditional_guidance_scale in self.encoder_unconditional_guidance_scales:
                    for skip_steps in self.skip_steps:
                        with torch.no_grad():
                            # DDIM forward.
                            z_list = ddpm_ddim_encoding_conditional(self.generator,
                                                                    steps=self.custom_steps,
                                                                    shape=x0.shape,
                                                                    eta=self.eta,
                                                                    white_box_steps=self.white_box_steps,
                                                                    skip_steps=skip_steps,
                                                                    x0=x0,
                                                                    scale=encoder_unconditional_guidance_scale,
                                                                    text=encode_text)
                            
                            z = torch.stack(z_list, dim=1).view(bsz, -1)
                            z_ensemble.append(z)

        return z_ensemble

    def forward(self, z_ensemble, original_img, encode_text, decode_text):
        # Eval mode for the generator.
        self.generator.eval()

        print("DOING GENERATE")
        img_ensemble = self.generate(z_ensemble, decode_text)

        print("DONE GENERATING")

        assert len(img_ensemble) == len(self.decoder_unconditional_guidance_scales) * len(
            self.encoder_unconditional_guidance_scales) * len(self.skip_steps) * self.n_trials

        # Post process.
        img_ensemble = [self.post_process(img) for img in img_ensemble]

        return img_ensemble

    @property
    def device(self):
        return next(self.parameters()).device


class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "cycleDiffusion"


# Determines when the script should be shown in the dropdown menu via the
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.


    def show(self, is_img2img):

        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        encode_text = gr.Textbox(label="encode_text", lines=1)
        decode_text = gr.Textbox(label="encode_text", lines=1)
        overwrite = gr.Checkbox(False, label="Overwrite existing files")
        skip_steps = gr.Slider(label="skip steps", minimum=0,
                               maximum=150, step=1, value=20)

        encoder_unconditional_guidance_scales = gr.Slider(label="encoder_unconditional_guidance_scales", minimum=1,
                               maximum=7, step=0.1, value=1)                 
        decoder_unconditional_guidance_scales = gr.Slider(label="decoder_unconditional_guidance_scales", minimum=1,
                               maximum=7, step=0.1, value=3)                    

        return [encode_text, decode_text, overwrite, skip_steps,encoder_unconditional_guidance_scales,decoder_unconditional_guidance_scales]


# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, encode_text, decode_text, overwrite, skip_steps,encoder_unconditional_guidance_scales, decoder_unconditional_guidance_scales):

        # If overwrite is false, append the rotation information to the filename
        # using the "basename" parameter and save it in the same directory.
        # If overwrite is true, stop the model from saving its outputs and
        # save the rotated and flipped images instead.
        basename = "cycle"
        if (not overwrite):
            basename = "cycle"
        else:
            p.do_not_save_samples = True

        #proc = process_images(p) #this will run the SD pipeline first, so don't do that
        #images=proc.images
        images=p.init_images

        #I don't really know if changing these is a good idea (probably once we muck around with other samplers. then try it)
        white_box_steps=100
        #custom_steps = p.steps #don't do this
        custom_steps=99

        #maybe change this in the future to allow other sizes?
        width,height=512,512
        
        eta = p.eta
        #eta can't be none
        if eta is None:
            eta=1.0

        source_model=p.sd_model
        sd = SDStochasticTextWrapper(source_model, custom_steps, eta, white_box_steps, [skip_steps],
                                    [encoder_unconditional_guidance_scales], [decoder_unconditional_guidance_scales],
                                    n_trials=1)

        for i in range(len(images)):

            original_image = images[i]
            original_image=original_image.resize((width,height))

            convert_tensor = transforms.ToTensor()

            img=convert_tensor(original_image)
            img=torch.unsqueeze(img, 0).half().cuda()

            with autocast('cuda'):
                print("ENCODING")
                z = sd.encode(img, [encode_text])
                print("GENERATING")
                img = sd(z, img, [encode_text], [decode_text])

            toPIL=transforms.ToPILImage()

            img=torch.squeeze(img[0])

            #images[i] = toPIL((img+1)/2)
            img=torch.clamp(img, min=0, max=1)
            images[i] = toPIL(img)

            #images.save_image(images[i], p.outpath_samples, basename,
            #                  p.seed + i, p.prompt, opts.samples_format, info=p.info, p=p)

        #return images
        processed = Processed(p, images)
        return processed
