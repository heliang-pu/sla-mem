"""MEM's short-horizon video memory for SLA (Spatial-Language-Action, a JAX pi0.5).

- mem.py: the space-time SigLIP, the patches that carry each camera's frames through SLA's
  pipeline (install()), the input packing (as_image()) and the inference history
  (FrameHistory). Its packing and history need NumPy alone; the rest runs in SLA's env.
- train.py: SLA's own training (scripts/train_align.py) with MEM installed.
"""
