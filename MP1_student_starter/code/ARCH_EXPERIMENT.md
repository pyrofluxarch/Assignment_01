# GPT architecture experiment

## Files and reproduction

Place `student.py`, `student_arch.py`, `student_stats.npz`, `arch_config.json`, and `arch_checkpoint.pt` in the supplied `code/` directory. The model imports the supplied `model.py`, and the evaluator needs the unchanged `common.py`, `evaluate.py`, and `data/` files.

```bash
python evaluate.py --checkpoint arch_checkpoint.pt --device cpu --precision fp32 --split validation
python evaluate.py --checkpoint arch_checkpoint.pt --device cpu --precision fp32 --split test
```

To reproduce training from random initialization, first generate the statistics with `python build_student_stats.py`, then run:

```bash
python train.py --implementation student_arch --config arch_config.json --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --run-dir runs/arch-reproduction
```

The training run and the selected checkpoint share the same model tensors. The training checkpoint SHA-256 is `90e5ecca642a5879a8ba5f7fb317f500873917da7a785a331fae0b1af5fbbc00`; only the inference mixture configuration was changed after validation selection. The selected checkpoint SHA-256 is `cddab1e93e3f693eb01ec251b07f1f7a23e59a2ba0e9e3637135b870e7e88391`. The training run processed 9,830,400 targets in 1,183.09 seconds on this CPU environment, with seed 17. The 20-step pilot processed another 163,840 training targets. Validation-only search reused cached model probabilities to check 421 mixture combinations; no additional gradient updates were performed. The test split was scored once after the final parameters were frozen.

## Architecture and results

The neural backbone uses six blocks of width 160 and five attention heads (2,188,320 parameters). Relative to the classroom baseline, it uses rotary position encoding, RMSNorm and a SwiGLU feedforward layer. `student.py` supplies the same causal window copy and train-only n-gram backoff as the earlier experiment. The selected window copy weight is 0.25, and the train-only n-gram smoothing values are 1000 and 80.

| Predictor | Complete validation BPB | Complete test BPB |
|---|---:|---:|
| Earlier baseline GPT plus both memories | 1.778330 | 1.791600 |
| New architecture, initial mixture | 1.685524 | — |
| New architecture, validation-selected mixture | **1.665352** | **1.677835** |

CPU FP32 test scoring took 27.73 seconds; process peak RSS was 2,301,316 KiB. The checkpoint is 8,767,541 bytes; its train-only statistics contain 38,020,210 bytes uncompressed, for 44.62 MiB combined. This time is close to the README's 5× reference time of 29.6 seconds; hardware-matched timing should be repeated before asserting formal compliance. The causal, normalization, finiteness, and example-independence checks passed with the new architecture. The requested 1.5 BPB threshold was not reached.

`guide/GUIDE.md` was not supplied, so its additional requirements could not be checked.
