README:

Installing tensorflow on mac-os is a bit tricky. The following steps are outlined as they helped me navigate this issue:

1. Create conda environment with python=3.8.16
2. Install jupyter and ipykernel using the following command
```
pip install notebook
```
4. Install tensorflow dependencies using the following code:

```
conda install -c apple tensorflow-deps
```
Important Note: Before running the command above, make sure to add the following lines in in ~/.condarc

```
subdirs:
  - osx-arm64
```

5. Install the tensorflow-macos and tensorflow-metal using the following command:

```
python -m pip install tensorflow-macos tensorflow-metal

```
6. Install other packages such as pandas, matplolib through pip. Installing through conda runs into compatibility issues with tensorflow-deps library