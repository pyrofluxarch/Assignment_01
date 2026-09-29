"""Train from scratch or continue a checkpoint; select checkpoints on validation only."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/student_config.json')
    p.add_argument('--resume-checkpoint', type=Path,
                   help='Continue model weights; optimizer is restarted and ancestry is recorded.')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/submission-v1')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Validation interval; 0 evaluates only after training.')
    p.add_argument('--save-best', action='store_true',
                   help='Save the lowest-validation-BPB checkpoint among evaluated steps.')
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=.1)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 0:
        p.error('Steps and batch size must be positive; eval-every must be nonnegative.')
    if args.lr <= 0 or not 0 <= args.min_lr_ratio <= 1 or args.warmup_steps < 0 or args.weight_decay < 0:
        p.error('Invalid optimizer or learning-rate settings.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    parent = None
    ancestry = None
    if args.resume_checkpoint is not None:
        parent = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=True)
        if parent['protocol'] != PROTOCOL or parent['implementation'] != args.implementation:
            p.error('Parent protocol or implementation does not match this run.')
        ancestry = sha(args.resume_checkpoint)
        config = parent['config']
        if args.config is not None and json.loads(args.config.read_text()) != config:
            p.error('Explicit config must match the resumed checkpoint.')
    else:
        config = json.loads((args.config or ROOT/'configs/baseline.json').read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    if parent is not None:
        model.load_state_dict(parent['model'])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    history, validation_history = [], []
    intermediate_validation_seconds = 0.
    best = None

    def validate(step):
        nonlocal best, intermediate_validation_seconds
        row = score(model, *data['validation'], device, 'fp32')
        row.pop('window_nll_nats')
        intermediate_validation_seconds += row['seconds']
        result = {'step': step, **row}
        validation_history.append(result)
        print(json.dumps({'validation': result}), flush=True)
        if best is None or row['bpb'] < best['validation']['bpb']:
            best = {
                'step': step,
                'validation': row,
                'model': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
        return row

    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        warmup = min(1., (step + 1) / args.warmup_steps) if args.warmup_steps else 1.
        learning_rate = args.lr * warmup * (
            args.min_lr_ratio + (1 - args.min_lr_ratio) * .5 * (1 + math.cos(math.pi * step / args.steps))
        )
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(),
                   'seconds': time.perf_counter() - started - intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            validate(step + 1)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds
    if not validation_history or validation_history[-1]['step'] != args.steps:
        validate(args.steps)
    final_validation = validation_history[-1]
    selected = best if args.save_best else {
        'step': args.steps, 'validation': final_validation,
        'model': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }
    prior_targets = parent.get('train_tokens', 0) if parent is not None else 0
    target_count = prior_targets + selected['step'] * args.batch_size * 256
    checkpoint = args.run_dir/'checkpoint.pt'
    payload = {'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
               'model': selected['model'], 'seed': args.seed, 'train_tokens': target_count}
    if ancestry is not None:
        payload['checkpoint_ancestry_sha256'] = ancestry
    torch.save(payload, checkpoint)
    result = {'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
              'seed': args.seed, 'parameters': sum(p.numel() for p in model.parameters()),
              'precision': precision, 'train_tokens': target_count,
              'run_train_tokens': args.steps * args.batch_size * 256,
              'selected_step': selected['step'], 'validation': selected['validation'],
              'final_validation': final_validation, 'history': history,
              'validation_history': validation_history,
              'preparation_seconds': preparation_seconds, 'train_seconds': train_seconds,
              'intermediate_validation_seconds': intermediate_validation_seconds,
              'process_seconds': time.perf_counter() - total_started,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'learning_rate': args.lr, 'min_lr_ratio': args.min_lr_ratio,
              'warmup_steps': args.warmup_steps, 'weight_decay': args.weight_decay,
              'checkpoint_ancestry_sha256': ancestry,
              'checkpoint_sha256': sha(checkpoint),
              'implementation_sha256': implementation_sha, **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result | {'history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()
