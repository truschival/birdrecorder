# Birdrecorder

A motion / yolo object detector that starts and stops recording based on acitity in the stream.

## Command Line Options

`uv run birdrecorder` executes with more or less sensible default arguments.

|   Option              |        Default      | Description                                       |
|:--------------------- | :-----------------: | :------------------------------------------------ | 
| `--detection`         | `yolo`              | Detection of motion or objects ('yolo', 'opencv)  | 
| `--source`            | `/dev/video0`       | Stream source (movie or capture device)           |
| `--auto-focus`        | false               | enable/disable auto focus of camera               |
| `--auto-white-balance`| false               | disable auto-white balance of camera              |
| `--color-temp`        | 5500                | Color temperatuer if auto-white-balance disabled  |  
| `--hysteresis`        | 15                  | Number of frames required to trigger recording    |
| `--min-area`          | 500                 | for opencv motion detector, minimum pixel area    |
| `--mark`              | false               | Show bounding boxes in recording                  |

## Install dependencies

Ultralytics Yolo depends on pytorch which in turn is uses Nvidia CUDA in its default installation.
If you have a recent NVidia Card you can get away without any extras. For older NVidia Cards or CPU-Only installations choose one of the extras `cu118` (CUDA 11.x series) or `cpu` (CPU Only)

1. Install ['uv'](https://docs.astral.sh/uv/).
2. Install the required dependencies and cuda backends (for yolo). 

Example for CUDA 11.x
  
   ```shell
   uv sync --dev --extra cu118
   ```

### additional information for CUDA / Pytorch

UV has a special implementation for selecting pytorch backends. 
[pytorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/#the-uv-pip-interface)
To install an older CUDA variant (for e.g. old 1060) install with `cu118` e.g.

```shell
uv pip install torch torchvision --torch-backend=cu118
``` 
