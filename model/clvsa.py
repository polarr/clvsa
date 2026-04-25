from typing import List, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.attention import DotProductSelfAttention, DotProductInterAttention
from model.clsa import *

class CLVSAModel(nn.Module):
    """
    Prior:
      p_theta(z_t | decoder_state_t, z_{t-1})

    Posterior:
      q_phi(z_t | backward_state_t)

    Training:
      z_t is sampled from posterior q_phi.

    Eval/test:
      z_t comes from prior p_theta.
      By default we use prior mean for deterministic evaluation.
    """
    is_variational = True

    def __init__(
        self,
        config: ModelConfig,
        use_encoder_self_attention: bool = True,
        use_decoder_self_attention: bool = True,
        use_inter_attention: bool = True,
        sample_prior_eval: bool = False,
    ):
        super().__init__()

        self.config = config

        self.rows = config.block_rows
        self.cols = config.block_cols
        self.channels = config.conv_channels
        self.num_layers = config.num_layers
        self.state_dim = self.rows * self.cols * self.channels
        self.latent_dim = getattr(config, "latent_dim", 32)
        self.output_dim = config.output_dim

        self.use_decoder_self_attention = use_decoder_self_attention
        self.use_inter_attention = use_inter_attention
        self.sample_prior_eval = sample_prior_eval

        self.encoder = CLSAEncoder(
            config,
            use_self_attention=use_encoder_self_attention,
        )

        # Forward decoder core.
        self.decoder_cells = nn.ModuleList()
        self.decoder_inter_attn = nn.ModuleList()
        self.decoder_self_attn = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            in_channels = 1 if layer_idx == 0 else self.channels

            self.decoder_cells.append(
                build_convlstm_cell(config, input_channels=in_channels)
            )

            if self.use_inter_attention:
                self.decoder_inter_attn.append(
                    DotProductInterAttention(self.state_dim)
                )

            if self.use_decoder_self_attention:
                self.decoder_self_attn.append(
                    DotProductSelfAttention(self.state_dim)
                )

        # Backward decoder / approximate posterior core.
        # First layer receives x frame + label frame, so input_channels=2.
        self.label_embedding = nn.Embedding(
            num_embeddings=config.output_dim,
            embedding_dim=self.rows * self.cols,
        )

        self.backward_cells = nn.ModuleList()
        self.backward_self_attn = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            in_channels = 2 if layer_idx == 0 else self.channels

            self.backward_cells.append(
                build_convlstm_cell(config, input_channels=in_channels)
            )

            self.backward_self_attn.append(
                DotProductSelfAttention(self.state_dim)
            )

        self.prior_fc_dim = getattr(config, "prior_fc_dim", 512)
        self.posterior_fc_dim = getattr(config, "posterior_fc_dim", 256)

        # Prior p_theta(z_t | x_1:t, z_1:t-1)
        self.prior_net = nn.Sequential(
            nn.Linear(self.state_dim + self.latent_dim, self.prior_fc_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.prior_mu = nn.Linear(self.prior_fc_dim, self.latent_dim)
        self.prior_logvar = nn.Linear(self.prior_fc_dim, self.latent_dim)

        # Posterior q_phi(z_t | x_T:t, y'_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(self.state_dim, self.posterior_fc_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.post_mu = nn.Linear(self.posterior_fc_dim, self.latent_dim)
        self.post_logvar = nn.Linear(self.posterior_fc_dim, self.latent_dim)

        # Main decoder prediction p_theta(y_t | h_t, z_t)
        self.head = PredictionHead(
            input_dim=self.state_dim + self.latent_dim,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

        # Backward decoder auxiliary prediction p_xi(y'_t | b_t)
        self.backward_head = PredictionHead(
            input_dim=self.state_dim,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

    def _prepare_frames(self, x_flat: torch.Tensor) -> torch.Tensor:
        b, t, _ = x_flat.shape
        x = x_flat.contiguous().view(b, t, self.rows, self.cols)
        return x.unsqueeze(3)  # (B, T, rows, 1, cols)

    def _vec_to_frame(self, x_vec: torch.Tensor) -> torch.Tensor:
        return x_vec.view(x_vec.size(0), self.rows, self.channels, self.cols)

    def _frame_to_vec(self, x_frame: torch.Tensor) -> torch.Tensor:
        return x_frame.reshape(x_frame.size(0), -1)

    def _label_frames(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, T)
        returns: (B, T, rows, 1, cols)
        """
        b, t = y.shape
        emb = self.label_embedding(y)  # (B, T, rows * cols)
        emb = emb.view(b, t, self.rows, self.cols)
        return emb.unsqueeze(3)

    def _reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_diag_gaussian(
        self,
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mu: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(q || p), diagonal Gaussian.

        Inputs:
          q_mu, q_logvar, p_mu, p_logvar: (B, T, latent_dim)

        Returns:
          scalar mean over batch/time.
        """
        kl = 0.5 * (
            p_logvar
            - q_logvar
            + (torch.exp(q_logvar) + (q_mu - p_mu).pow(2)) / torch.exp(p_logvar)
            - 1.0
        )

        return kl.sum(dim=-1).mean()

    def _build_posterior(
        self,
        x_dec: torch.Tensor,
        y_dec: torch.Tensor,
    ):
        """
        Build approximate posterior q_phi(z_t | x_{T:t}, y'_t)
        with a backward ConvLSTM decoder.

        Returns tensors in original time order:
          q_mu:            (B, T, latent_dim)
          q_logvar:        (B, T, latent_dim)
          backward_logits: (B, T, output_dim)
        """
        x = self._prepare_frames(x_dec)
        y_frames = self._label_frames(y_dec)

        x_rev = torch.flip(x, dims=[1])
        y_frames_rev = torch.flip(y_frames, dims=[1])

        b, t, _, _, _ = x_rev.shape

        hidden = [
            cell.init_state(b, x_dec.device, x_dec.dtype)
            for cell in self.backward_cells
        ]

        raw_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        refined_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        q_mu_rev = []
        q_logvar_rev = []
        backward_logits_rev = []

        for step in range(t):
            # First layer receives both input frame and label frame.
            layer_input = torch.cat(
                [x_rev[:, step], y_frames_rev[:, step]],
                dim=2,
            )  # (B, rows, 2, cols)

            new_hidden = []

            for layer_idx, cell in enumerate(self.backward_cells):
                h_prev, c_prev = hidden[layer_idx]
                h_raw, c_new = cell(layer_input, (h_prev, c_prev))

                h_raw_vec = self._frame_to_vec(h_raw)

                h_vec, _ = self.backward_self_attn[layer_idx].forward_step(
                    h_raw_vec,
                    raw_histories[layer_idx],
                    refined_histories[layer_idx],
                )

                raw_histories[layer_idx].append(h_raw_vec)
                refined_histories[layer_idx].append(h_vec)

                h_out = self._vec_to_frame(h_vec)
                new_hidden.append((h_raw, c_new))
                layer_input = h_out

            hidden = new_hidden

            b_vec = refined_histories[-1][-1]

            posterior_h = self.posterior_net(b_vec)

            q_mu_rev.append(self.post_mu(posterior_h).unsqueeze(1))
            q_logvar_rev.append(self.post_logvar(posterior_h).unsqueeze(1))
            backward_logits_rev.append(self.backward_head(b_vec).unsqueeze(1))

        q_mu = torch.flip(torch.cat(q_mu_rev, dim=1), dims=[1])
        q_logvar = torch.flip(torch.cat(q_logvar_rev, dim=1), dims=[1])
        backward_logits = torch.flip(torch.cat(backward_logits_rev, dim=1), dims=[1])

        return q_mu, q_logvar, backward_logits

    def _decode_with_latents(
        self,
        x_dec: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_hidden: List[Tuple[torch.Tensor, torch.Tensor]],
        q_mu: torch.Tensor = None,
        q_logvar: torch.Tensor = None,
    ):
        x = self._prepare_frames(x_dec)
        b, t, _, _, _ = x.shape

        hidden = [(h.clone(), c.clone()) for (h, c) in encoder_hidden]

        pre_self_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        refined_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        logits_steps = []
        p_mu_steps = []
        p_logvar_steps = []
        z_steps = []

        prev_z = x_dec.new_zeros(b, self.latent_dim)

        for step in range(t):
            layer_input = x[:, step]
            new_hidden = []

            for layer_idx, cell in enumerate(self.decoder_cells):
                h_prev, c_prev = hidden[layer_idx]
                h_raw, c_new = cell(layer_input, (h_prev, c_prev))

                h_vec = self._frame_to_vec(h_raw)

                if self.use_inter_attention:
                    h_vec, _ = self.decoder_inter_attn[layer_idx](
                        h_vec,
                        encoder_outputs,
                    )

                if self.use_decoder_self_attention:
                    h_pre_self_vec = h_vec

                    h_vec, _ = self.decoder_self_attn[layer_idx].forward_step(
                        h_pre_self_vec,
                        pre_self_histories[layer_idx],
                        refined_histories[layer_idx],
                    )

                    pre_self_histories[layer_idx].append(h_pre_self_vec)

                refined_histories[layer_idx].append(h_vec)

                h_out = self._vec_to_frame(h_vec)
                new_hidden.append((h_raw, c_new))
                layer_input = h_out

            hidden = new_hidden

            h_top = refined_histories[-1][-1]

            prior_input = torch.cat([h_top, prev_z], dim=-1)
            prior_h = self.prior_net(prior_input)

            p_mu_t = self.prior_mu(prior_h)
            p_logvar_t = self.prior_logvar(prior_h)

            if self.training and q_mu is not None and q_logvar is not None:
                z_t = self._reparameterize(
                    q_mu[:, step, :],
                    q_logvar[:, step, :],
                )
            else:
                if self.sample_prior_eval:
                    z_t = self._reparameterize(p_mu_t, p_logvar_t)
                else:
                    z_t = p_mu_t

            logits_t = self.head(torch.cat([h_top, z_t], dim=-1))

            logits_steps.append(logits_t.unsqueeze(1))
            p_mu_steps.append(p_mu_t.unsqueeze(1))
            p_logvar_steps.append(p_logvar_t.unsqueeze(1))
            z_steps.append(z_t.unsqueeze(1))

            prev_z = z_t

        logits = torch.cat(logits_steps, dim=1)
        p_mu = torch.cat(p_mu_steps, dim=1)
        p_logvar = torch.cat(p_logvar_steps, dim=1)
        z_seq = torch.cat(z_steps, dim=1)

        return logits, p_mu, p_logvar, z_seq

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        y_dec: torch.Tensor = None,
        return_attention: bool = False,
    ):
        if return_attention:
            raise NotImplementedError("CLVSA attention return is not implemented yet.")

        encoder_outputs, encoder_hidden, _ = self.encoder(
            x_enc,
            return_attention=False,
        )

        q_mu = None
        q_logvar = None
        backward_logits = None

        if self.training:
            if y_dec is None:
                raise ValueError("CLVSA requires y_dec during training.")

            q_mu, q_logvar, backward_logits = self._build_posterior(
                x_dec=x_dec,
                y_dec=y_dec,
            )

        logits, p_mu, p_logvar, z_seq = self._decode_with_latents(
            x_dec=x_dec,
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden,
            q_mu=q_mu,
            q_logvar=q_logvar,
        )

        if not self.training:
            return logits

        kl = self._kl_diag_gaussian(
            q_mu=q_mu,
            q_logvar=q_logvar,
            p_mu=p_mu,
            p_logvar=p_logvar,
        )

        return {
            "logits": logits,
            "backward_logits": backward_logits,
            "kl": kl,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "z": z_seq,
        }