import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time
import numpy as np
from weaver.utils.logger import _logger
from weaver.utils.nn.tools import (
    train_classification,
    evaluate_regression,
)

from weaver.utils.import_tools import import_module

_mod = import_module(os.path.join(os.path.dirname(__file__), '../ParticleTransformer2023.py'), 'ParT')
SequenceTrimmer = _mod.SequenceTrimmer
Embed = _mod.Embed
PairEmbed = _mod.PairEmbed
Block = _mod.Block
trunc_normal_ = _mod.trunc_normal_
build_sparse_tensor = _mod.build_sparse_tensor

class ParticleTransformerMultiClsTokens(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 num_cls_tokens=1,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[(512, 0.1)],
                 activation='gelu',
                 enable_mem_efficient=False,
                 # misc
                 trim=True,
                 for_inference=False,
                 num_classes_cls=None, # for onnx export
                 use_amp=False,
                 return_embed=False,
                 export_embed=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.num_classes_cls = num_classes_cls
        self.use_amp = use_amp
        self.return_embed = return_embed
        self.export_embed = export_embed

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True,
                           enable_mem_efficient=enable_mem_efficient)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.num_cls_tokens = num_cls_tokens
        self.cls_token = nn.Parameter(torch.zeros(num_cls_tokens, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(self.num_cls_tokens, x.size(1), -1)  # (N_cls_token, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            cls_tokens = self.norm(cls_tokens) # (N_cls_token, N, C)
            x_cls = cls_tokens[0] # (N, C), only use the first cls token

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                # for onnx export
                assert self.num_classes_cls is not None
                output_cls, output_reg = output.split([self.num_classes_cls, output.size(1) - self.num_classes_cls], dim=1)
                output_cls = torch.softmax(output_cls, dim=1)
                output = torch.cat([output_cls, output_reg], dim=-1)

            # print('output:\n', output)
            if self.export_embed:
                return torch.cat([output, x_cls], dim=-1)

            if self.return_embed == False:
                return output
            else:
                return output, cls_tokens


class CLIPWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        kwargs_main = {k: v for k, v in kwargs.items() if not k.startswith('gen_mod')}
        kwargs_main.update(num_cls_tokens=2, return_embed=True)

        kwargs_gen = {k[len('gen_mod_'):]: v for k, v in kwargs.items() if k.startswith('gen_mod')}
        kwargs_gen.update(num_classes=None, fc_params=None, return_embed=False)

        self.mod_main = ParticleTransformerMultiClsTokens(**kwargs_main)
        self.mod_gen = ParticleTransformerMultiClsTokens(**kwargs_gen)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod_main.cls_token', 'mod_gen.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask, gen_points, gen_features, gen_lorentz_vectors, gen_mask, labels):

        logits_cls, tokens = self.mod_main(features, v=lorentz_vectors, mask=mask) # the second token is the latent space for CLIP loss
        logits_cont = tokens[1] # (N, C)

        logits_cont_gen = self.mod_gen(gen_features, v=gen_lorentz_vectors, mask=gen_mask)

        labels_cls = labels.long()

        # logits_cls, labels_cls for classification, same with Sophon
        # logits_cont, logits_cont_gen for contrastive learning

        return logits_cls, labels_cls, logits_cont, logits_cont_gen


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        kwargs.update(return_embed=False)
        self.mod_main = ParticleTransformerMultiClsTokens(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod_main.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod_main(features, v=lorentz_vectors, mask=mask)


class CLIPLoss(torch.nn.Module):
    '''
        Computes the CLIP loss and classification loss
    '''

    def __init__(self, beta=1.):
        super(CLIPLoss, self).__init__()
        self.beta = beta
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cls_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, logits_cls, labels_cls, logits_cont, logits_cont_gen):
        '''
            logits_cls: (batch, out_dim), the output logits_cls of main ParT for standard classification.
            labels_cls: (batch,), labels for classification
            logits_cont: (batch, dim_latent), the latent features of main ParT to compute contrastive loss
            logits_cont_gen: (batch, dim_latent), the latent features of GEN-level ParT to compute contrastive loss
        '''
        # compute classification loss
        loss_cls = F.cross_entropy(logits_cls, labels_cls)

        # CLIP constrastive learning
        # normalize the features
        logits_cont = logits_cont / logits_cont.norm(dim=-1, keepdim=True)
        logits_cont_gen = logits_cont_gen / logits_cont_gen.norm(dim=-1, keepdim=True)

        # compute cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_cont_2d = logit_scale * logits_cont @ logits_cont_gen.t() # (batch, batch)
        logits_cont_2d_t = logits_cont_2d.t()
        indices = torch.arange(logits_cont.size(0)).to(logits_cont.device) # (batch,)

        loss_cont = F.cross_entropy(logits_cont_2d, indices) + F.cross_entropy(logits_cont_2d_t, indices)

        loss = loss_cls + self.beta * loss_cont
        return {'loss': loss, 'loss_cls': loss_cls.detach(), 'loss_cont': loss_cont.detach()}


def get_model(data_config, **kwargs):

    cfg = dict(
        ## main ParT configs
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=data_config.label_value_cls_num,
        # network configurations
        pair_input_dim=4,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        num_cls_tokens=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
        num_classes_cls=data_config.label_value_cls_num, # for onnx export

        ## GEN ParT configs
        gen_mod_input_dim=len(data_config.input_dicts.get('gen_features', [])),
        gen_mod_num_classes=None,
        # network configurations
        gen_mod_pair_input_dim=4,
        gen_mod_embed_dims=[128, 512, 128],
        gen_mod_pair_embed_dims=[64, 64, 64],
        gen_mod_num_heads=8,
        gen_mod_num_layers=4,
        gen_mod_num_cls_layers=2,
        gen_mod_block_params=None,
        gen_mod_cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        gen_mod_fc_params=None,
        gen_mod_activation='gelu',
        # misc
        gen_mod_trim=True,
        gen_mod_for_inference=False,
    )

    kwargs.pop('beta', None)
    for_clip_finetune = kwargs.pop('for_clip_finetune', False)
    cfg.update(**kwargs)

    if not for_clip_finetune:
        _logger.info('Model config: %s' % str(cfg))
        model = CLIPWrapper(**cfg)
    else:
        cfg = {k: v for k, v in cfg.items() if not k.startswith('gen_mod_')}
        _logger.info('Model config: %s' % str(cfg))
        model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    beta = kwargs.get('beta', 1.)
    for_clip_finetune = kwargs.pop('for_clip_finetune', False)
    if not for_clip_finetune:
        return CLIPLoss(beta=beta)
    else:
        return torch.nn.CrossEntropyLoss()


def get_train_fn(data_config, **kwargs):
    for_clip_finetune = kwargs.get('for_clip_finetune', False)
    if not for_clip_finetune:
        return train_custom_clip
    else:
        return train_classification


def get_evaluate_fn(data_config, **kwargs):
    for_clip_finetune = kwargs.get('for_clip_finetune', False)
    if not for_clip_finetune:
        return evaluate_custom_clip
    else:
        return evaluate_regression


def train_custom_clip(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    num_batches = 0
    count = 0
    total_correct = 0
    total_losses = None
    start_time = time.time()
    flag = False
    with tqdm.tqdm(train_loader) as tq:
        for X, _, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            num_examples = inputs[0].shape[0]

            opt.zero_grad()
            model_output = model(*inputs)
            if not isinstance(model_output, tuple):
                model_output = (model_output,)
            losses = loss_func(*model_output)
            if not isinstance(losses, dict):
                losses = {'loss': losses}
            # print(losses)
            if grad_scaler is None:
                losses['loss'].backward()
                opt.step()
            else:
                grad_scaler.scale(losses['loss']).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            # also evaluate classification performance here
            logits, label = model_output[0], model_output[1]
            _, preds = logits.max(1)
            correct = (preds == label).sum().item()
            total_correct += correct

            num_batches += 1
            count += num_examples
            if total_losses is None:
                total_losses = {k: 0. for k in losses}
            for k in losses:
                losses[k] = losses[k].item()
                total_losses[k] += losses[k]
            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Acc': '%.5f' % (correct / num_examples),
                **{k: '%.5f' % losses[k] for k in list(losses.keys())[:3]}
            })

            if tb_helper:
                tb_helper.write_scalars(
                    [
                        ("lr/train", scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'], tb_helper.batch_train_count + num_batches),
                        ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches)
                    ] + 
                    [(k + '/train', losses[k], tb_helper.batch_train_count + num_batches) for k in losses])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train ' + ', '.join(['Avg_%s: %.5f' % (k, total_losses[k] / num_batches) for k in losses]))

    if tb_helper:
        tb_helper.write_scalars(
            [("Acc/train (epoch)", total_correct / count, epoch)] +
            [(k + '/train (epoch)', total_losses[k] / num_batches, epoch) for k in losses]
            )
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')

        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_custom_clip(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                    eval_metrics=[], tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    num_batches = 0
    count = 0
    total_correct = 0
    total_losses = None
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                num_examples = inputs[0].shape[0]
                model_output = model(*inputs)
                if for_training:
                    if not isinstance(model_output, tuple):
                        model_output = (model_output,)
                    losses = loss_func(*model_output)
                else:
                    losses = torch.Tensor([0.])
                if not isinstance(losses, dict):
                    losses = {'loss': losses}

                # also evaluate classification performance here
                logits, label = model_output[0], model_output[1]
                _, preds = logits.max(1)
                correct = (preds == label).sum().item()
                total_correct += correct

                num_batches += 1
                count += num_examples
                if total_losses is None:
                    total_losses = {k: 0. for k in losses}
                for k in losses:
                    losses[k] = losses[k].item()
                    total_losses[k] += losses[k]
                tq.set_postfix({
                    'Acc': '%.5f' % (correct / num_examples),
                    **{k: '%.5f' % losses[k] for k in list(losses.keys())[:3]}
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    # scores = np.concatenate(scores)
    # labels = {k: _concat(v) for k, v in labels.items()}
    # metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    # _logger.info('Evaluation metrics: \n%s', '\n'.join(
    #     ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars(
            [("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch)] + 
            [(k + '/%s (epoch)' % tb_mode, total_losses[k] / num_batches, epoch) for k in losses]
            )
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    if for_training:
        return total_losses['loss'] / count
    else:
        # convert 2D labels/scores
        # observers = {k: _concat(v) for k, v in observers.items()}
        zeros = np.zeros_like(total_losses['loss'])
        return total_losses['loss'] / count, zeros, zeros, {'k': zeros}
