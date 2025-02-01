# Speech Recognition Model Benchmark Tool

This tool automates the process of benchmarking speech recognition models using datasets from Hugging Face Hub.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/simonlobgromov/RunSTT_benchmark
cd RunSTT_benchmark
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Authentication Setup

Before running the benchmark, setup Hugging Face authentication:

1. Configure git credentials:
```bash
git config --global credential.helper store
```

2. Login to Hugging Face:
```bash
huggingface-cli login
```
When prompted, enter your Hugging Face token (get it from https://huggingface.co/settings/tokens)

## Configuration

Create a `config.yaml` file with the following structure:

```yaml
dataset:
  reponame: 'username/dataset-name'  # Input dataset on HF Hub
  split: 'train'                     # Dataset split (optional)

result_dataset:
  reponame: 'username/result-dataset' # Where to save results
  private: false                      # Repository visibility

model_id: 'model-name'               # Model to benchmark
batch_size: 16                       # Processing batch size
accelerator: 'gpu'
devices: [0, 1, 2, 3]
```

## Usage

Run the benchmark:
```bash
python main.py
```

## Output

The tool creates a dataset with the following columns:
- id: Original sample ID
- raw_transcription: Original raw transcription
- transcription: Processed transcription
- dataset_name: Source dataset name
- augmentation_type: Type of augmentation (if any)
- snr: Signal-to-noise ratio (if applicable)
- text_predict: Model predictions

## Troubleshooting

Common issues and solutions:

1. Authentication errors:
   - Check if you're logged into Hugging Face
   - Verify token permissions
   - Try logging in again

2. Memory issues:
   - Reduce batch_size in config
   - Use a smaller subset of data for testing
   - Check available GPU memory

3. Dataset access issues:
   - Verify dataset permissions
   - Check dataset name spelling
   - Ensure dataset exists on Hub

## Contributing

Feel free to open issues or submit pull requests.


## Citation

```
@misc{stt-benchmark-tool,
  author = {Denis Pavloff},
  title = {Speech Recognition Benchmark Tool for Kyrgyz Language},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/simonlobgromov/RunSTT_benchmark}
}
```
