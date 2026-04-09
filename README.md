## Environment setup

This project was tested with Python 3.10.19.

Create a virtual environment:

```bash
conda create -n obgnn python=3.10
conda activate obgnn
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Note: If running on MIT ORCD Engaging, you might need to add this line before using `conda`
```bash
module load miniforge
```

Note: To get an interactive session with CPU
```bash
salloc -N 1 -c 4 --mem=32G -p mit_normal_cpu --time=1:00:00
```
with GPU:
```bash
salloc -N 1 -G 1 -c 4 -p mit_normal_gpu --time=1:00:00
```