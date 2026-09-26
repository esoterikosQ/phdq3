"""Experimental batch-1 global patch prefix reuse for no-cache BLT forward.

Entropy and local encoder still run on the full input. Global transformer and
optionally local decoder can reuse state. This is a diagnostic backend, not an eval backend:
BF16 hidden states change with input length and full-split token parity is unproven.
"""
import torch

from .frontier import patch_starts, shared_closed_patch_count


class _LayerKV:
    def __init__(self, previous=None, count=0):
        self.layers = {
            index: (key[:, :, :count], value[:, :, :count])
            for index, (key, value) in (previous or {}).items()
        }

    def update(self, key, value, layer_idx):
        if layer_idx in self.layers:
            old_key, old_value = self.layers[layer_idx]
            key = torch.cat((old_key, key), dim=2)
            value = torch.cat((old_value, value), dim=2)
        self.layers[layer_idx] = (key.detach(), value.detach())
        return key, value


class GlobalPrefixReuse:
    """Keep structurally closed global KV and optionally decoder byte KV."""

    def __init__(self, model, *, reuse_decoder=False):
        self.model = model
        self.reuse_decoder = reuse_decoder
        self.reset()

    def reset(self):
        self.previous_ids = None
        self.starts = None
        self.global_hidden = None
        self.kv = None
        self.decoder_kv = None

    def run(self, input_ids):
        if self.model.training or torch.is_grad_enabled():
            raise ValueError('Global prefix reuse requires eval and inference_mode')
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
            raise ValueError('Only unpadded batch-1 prefixes of at least two bytes are supported')
        current_ids = tuple(int(value) for value in input_ids[0].tolist())
        if self.previous_ids is not None and current_ids[:-1] != self.previous_ids:
            self.reset()
        captured = {}

        def save_patch(_module, _inputs, result):
            captured['starts'] = patch_starts(result[1][0].tolist(), input_length=len(current_ids))

        global_module = self.model.model.global_transformer
        original_forward = global_module.forward
        decoder_module = self.model.model.local_decoder
        original_decoder_forward = decoder_module.forward

        def forward_global(*args, **kwargs):
            if args or 'past_key_values' in kwargs:
                raise ValueError('Unexpected global transformer call signature')
            starts = captured['starts']
            count = (shared_closed_patch_count(self.starts, starts)
                     if self.starts is not None else 0)
            if current_ids[-1] == 2:  # EOS can change the segment mask.
                count = 0
            # The decoder uses the preceding patch's global output. If no new
            # boundary appeared, the current open patch is not queried for the
            # last byte; retaining its old value avoids the entire global pass.
            if current_ids[-1] != 2 and starts == self.starts and len(starts) > 1:
                captured['global_hidden'] = self.global_hidden
                captured['kv'] = self.kv
                captured['reused_closed_patches'] = count
                captured['skipped_global'] = True
                return self.global_hidden
            cache = _LayerKV(self.kv, count)
            if count:
                tail = original_forward(
                    inputs_embeds=kwargs['inputs_embeds'][:, count:],
                    attention_mask=kwargs['attention_mask'][:, :, count:, :],
                    position_ids=kwargs['position_ids'][:, count:],
                    past_key_values=cache,
                )
                result = torch.cat((self.global_hidden[:, :count], tail), dim=1)
            else:
                result = original_forward(**kwargs, past_key_values=cache)
            captured['global_hidden'] = result.detach()
            captured['kv'] = cache.layers
            captured['reused_closed_patches'] = count
            captured['skipped_global'] = False
            return result

        def forward_decoder(*args, **kwargs):
            if args or kwargs.get('past_key_values') is not None:
                raise ValueError('Unexpected local decoder call signature')
            kwargs = dict(kwargs)
            kwargs.pop('past_key_values', None)
            same_boundaries = current_ids[-1] != 2 and captured['starts'] == self.starts
            can_reuse = bool(self.reuse_decoder and same_boundaries and self.decoder_kv)
            cache = _LayerKV(self.decoder_kv, len(self.previous_ids)) if can_reuse else _LayerKV()
            if can_reuse:
                # Cross attention still sees the current global patch table;
                # only the last byte query and its self-attention KV are new.
                kwargs = dict(kwargs)
                kwargs['input_ids'] = kwargs['input_ids'][:, -1:]
                kwargs['inputs_embeds'] = kwargs['inputs_embeds'][:, -1:]
                kwargs['attention_mask'] = kwargs['attention_mask'][:, :, -1:, :]
                kwargs['position_ids'] = kwargs['position_ids'][:, -1:]
                kwargs['encoder_attention_mask'] = kwargs['encoder_attention_mask'][:, :, -1:, :]
            result = original_decoder_forward(**kwargs, past_key_values=cache)
            captured['decoder_kv'] = cache.layers
            captured['reused_decoder'] = can_reuse
            return result

        handle = self.model.model.patcher.register_forward_hook(save_patch)
        global_module.forward = forward_global
        if self.reuse_decoder:
            decoder_module.forward = forward_decoder
        try:
            output = self.model(input_ids=input_ids, use_cache=False)
        except Exception:
            self.reset()
            raise
        finally:
            global_module.forward = original_forward
            if self.reuse_decoder:
                decoder_module.forward = original_decoder_forward
            handle.remove()
        self.previous_ids = current_ids
        self.starts = captured['starts']
        self.global_hidden = captured['global_hidden']
        self.kv = captured['kv']
        if self.reuse_decoder:
            self.decoder_kv = captured['decoder_kv']
        return (output, captured['reused_closed_patches'], captured['skipped_global'],
                captured.get('reused_decoder', False))
