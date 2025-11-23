# Birdrecorder

## Install dependencies

UV has a special implementation for selecting pytorch backends. 
[pytorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/#the-uv-pip-interface)

To install an older CUDA variant (for e.g. old 1060) install with `cu118` e.g.

```shell
uv pip install torch torchvision --torch-backend=cu118
``` 

Or since I have the extras defined:

``` shell
uv sync --dev --extra cu118
``` 

