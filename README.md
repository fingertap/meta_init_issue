# Mini-repro for precision issue of `transformers` models initialized with meta device

Here we explore the issue of FSDP having accuracy problems when the passed model is a Transformers model.

Observations of the phenomenon:

1. When the model passed is on CUDA, the loss is normal.
2. When the model passed is on meta, and after specifying `param_init_fn` for initialization and then loading `state_dict`, the loss is abnormal.
3. When the model passed is on meta, and after explicitly calling `to_empty` and then loading `state_dict`, the loss is abnormal.

Therefore, the problem lies in the difference between the Transformers model after `to_empty` and `load_state_dict`, and the model directly read into CUDA.

I can reproduce the issue with a minimal example that runs on a single A100 GPU.

```bash
python single_gpu.py -m cuda
python single_gpu.py -m meta_accelerate
python single_gpu.py -m meta_torch
```

The outputs are as follows:

```
cuda mean loss 0.151278076171875
meta_accelerate mean loss 4.5384375
meta_torch mean loss 3.5678125
```

I expect that the loss of `meta_accelerate` and `meta_torch` should be the similar to `cuda`, since I have already loaded the state_dict of the model.
