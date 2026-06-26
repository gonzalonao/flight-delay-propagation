# GPU Setup — RTX 5070 (Blackwell) y otras GPUs NVIDIA

Esta guía resuelve el problema más común al lanzar entrenamientos:
**`torch.cuda.is_available()` devuelve `False` (o `True` pero el modelo
sigue corriendo en CPU)**.

## TL;DR

```bash
# Hardware soportado: cualquier GPU NVIDIA con driver >= 525.
# RTX 5070 / 5080 / 5090 (Blackwell, sm_120) requieren wheels CUDA 12.8+.
# RTX 30xx / 40xx funcionan con cu121 o cu124, pero cu128 también vale.

pip uninstall -y torch torchvision torch-geometric torch-scatter torch-sparse torch-cluster
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install torch_geometric
```

Verifica:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Salida esperada en la 5070:

```
2.6.0+cu128  12.8  True  NVIDIA GeForce RTX 5070
```

## Forzar GPU desde el config

Por defecto los scripts usan `device: auto` (CUDA si está, si no CPU con
warning). Para que un entrenamiento **falle si no hay GPU** en lugar de
caer silenciosamente a CPU (que era lo que enmascaraba runs lentos antes
del fix), añade en el YAML del modelo:

```yaml
training:
  device: cuda    # auto | cuda | cpu
```

Con `device: cuda`, si `torch.cuda.is_available()` es `False` el script
lanza un `RuntimeError` con instrucciones concretas (build CPU-only,
wheels antiguos, driver desactualizado…) en lugar de seguir adelante
en CPU.

## Diagnóstico

Si tras instalar cu128 sigues sin ver la GPU, consulta el log del primer
arranque. `select_device()` ahora imprime:

```
Dispositivo: cuda:0 (NVIDIA GeForce RTX 5070, sm_120, 12.0 GB, torch CUDA build=12.8)
```

Si sale algo como esto:

```
Dispositivo: cpu (CUDA no disponible).
  torch version: 2.5.0+cpu | CUDA build: None
```

→ El wheel instalado es **CPU-only**. Reinstala con el index-url cu128
de arriba.

Si sale:

```
GPU Blackwell (sm_120) detectada con torch CUDA build 12.1. Es probable
que las operaciones fallen en runtime: instala wheels cu128.
```

→ Tienes wheels antiguos. PyTorch detecta la GPU pero no tiene kernels
para sm_120 — el primer `.cuda()` real reventará. Igualmente, reinstala
con cu128.

## Fallback: Colab

Si la 5070 no se puede activar a tiempo (driver, permisos, etc.),
sube los snapshots cacheados a Drive y entrena en Colab:

```python
# En Colab
!pip install torch_geometric
# torch ya viene con cu121 en Colab; suficiente para T4/L4/A100.
```

El caché de snapshots (`data/processed/snapshots/snapshots_<hash>.pt`)
es portable: si subes el `.pt` a Drive y montas Drive en Colab,
`build_graph_dataset` hace **HIT en caché** y la fase de preprocesado
desaparece — el bottleneck pasa a ser data-loader, no construcción de
features.

## Fallback: CPU (Ryzen 7 9700X)

Funciona, pero es ~10–20× más lento. Útil para depurar el pipeline,
no para la corrida final del TFM. Asegúrate de cachear snapshots
primero (un solo build vale para todas las re-ejecuciones siguientes).
