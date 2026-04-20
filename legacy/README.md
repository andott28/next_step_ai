Legacy Code
===========

This directory contains older experimental model, SCA, sparse-attention, and one-off diagnostic scripts that are not part of the maintained README quick-start path.

The maintained package surface is the streaming runtime under `llama3_neuroplastic/experiments/streaming_llama_runtime.py` plus the fitting, benchmark, eval, and verification CLIs in `llama3_neuroplastic/experiments/`.

Do not import from this directory in production code. Promote a file back into `llama3_neuroplastic/` only after fixing its imports, adding tests, and documenting its CLI or API in the README.
