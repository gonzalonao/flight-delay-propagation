# Documentation

Navigable index for the `flight-delay-propagation` project. Start with the
[top-level README](../README.md) for the overview; the documents below go
deeper on specific areas.

## Architecture & modeling

- **[Ablation log](ablation-log.md)** — the experiment log: every atomic
  change to features, loss, architecture and training, with which ones
  landed in the deployed champion. Doubles as the rationale for the final
  feature set.

## Deployment

- **[Deployment plan](deployment-plan.md)** — full end-to-end architecture
  across local training, Azure Functions and Microsoft Fabric, including the
  hourly ingestion/inference loop and the OneLake Delta tables.
- **[Power BI report spec](powerbi-report-spec.md)** — the data model, DAX
  measures and page-by-page visual specification for the dashboard.

## Operations

- **[GPU setup](gpu-setup.md)** — getting CUDA working (RTX 50-series /
  Blackwell needs cu128 wheels), forcing the device from config, and the
  Colab / CPU fallbacks.

## Roadmap

- **[Roadmap / future work](roadmap.md)** — planned improvements, led by
  serving real weather to the live model.

## Diagrams

The figures used across the docs and README live in [`images/`](images/):
the spatio-temporal Transformer architecture, the end-to-end pipeline, the
deployment architecture, and dashboard screenshots.
