# Linux-Docs-LLM

A small‐scale, single-GPU BERT-LM pretrained on Linux documentation. This repo contains all custom code for:

- Streaming & masking data  
- Transformer model definition  
- Training loop with gradient accumulation, metrics & plotting  
- Interactive sampling  

It leverages HuggingFace tokenizers and schedulers, PyTorch core, Matplotlib and TQDM for progress bars.

All the code was written by myself while referencing related papers and pytorch documention.
The code architecture is heavily inspired by the DDPM model's code of assignment 2.
ChatGPT was only used to help with researching relevant paperes and filtering the data.

## Video presentation

<iframe width="1113" height="626" src="https://www.youtube.com/embed/C-KXEVPbiFA" title="Linux lm project presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

If the embeded player isn't working, here is a direct link: https://youtu.be/C-KXEVPbiFA

**Correction:** In the presentation I mistakenly say that the model has 6 heads, but in reality it has 4 heads.

---

## Repository Structure

```text
.
├── README.md                     <- this file
├── requirements.txt              <- pip install these
├── dataset.py                    <- StreamingMLMDataset: reads, chunks & masks text
├── modules.py                    <- TokenEmbedding, SinusoidalPositionalEmbedding, TransformerBlock
├── network.py                    <- TransformerLM: decoder-only Transformer + generate()
├── model.py                      <- LMModule: loss, save/load, sample wrapper
├── train.py                      <- end-to-end training script + metrics + plotting
├── interactive.py                <- command-line demo
├── sampler.txt                   <- sampling utilities
└── steps_per_epoch_calculator.py <- calculates steps per epoch for progress bar to work
```

## Running The Model

Download and unzip the models

- 13M parameters: https://tucgr-my.sharepoint.com/:u:/g/personal/nsopiadis_tuc_gr/EVGJ77i6fKNDtRtpN9ZPDEIBIa4p0nvWMFPsX9E4DgU9lg?e=dyQ2Ky
- 25M parameters: https://tucgr-my.sharepoint.com/:u:/g/personal/nsopiadis_tuc_gr/EemwADOI_hlBl1NxBNur7ZEBs_pOKkxQ2405NKv0UAMgTw?e=yUzTbK

Install pip requirements with ``pip3 install requirements.txt``

Run ``python3 interactive.py [path/to/checkpoint.pt]``
