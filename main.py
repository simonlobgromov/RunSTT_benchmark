from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import torch
from tqdm import tqdm
import yaml


class AutoBenchmark:

    """
    AutoBenchmark - A class for automated speech recognition model benchmarking.

    This class automates the process of:
    - Loading and preparing a speech dataset
    - Running inference using a specified model
    - Creating a benchmark dataset with predictions
    - Pushing results to Hugging Face Hub

    The class handles:
    - Model and processor initialization
    - Batch processing of audio data
    - Dataset transformations and concatenations
    - Result dataset creation and upload

    Args:
      config (str, optional): Path to YAML configuration file. Defaults to 'config.yaml'.
          Config file should contain:
              - dataset: Configuration for input dataset
                  - reponame: Hugging Face dataset repository name
                  - split: Dataset split to use (optional)
              - result_dataset: Configuration for output dataset
                  - reponame: Repository name for results
                  - private: Boolean for repository visibility
              - model_id: Hugging Face model identifier
              - batch_size: Batch size for processing

    Attributes:
      config (dict): Loaded configuration
      batch_size (int): Batch size for processing
      model_id (str): Model identifier
      dataset_config (dict): Input dataset configuration
      result_dataset_config (dict): Output dataset configuration
      device (str): Processing device (CUDA/CPU)
      torch_dtype: PyTorch data type
      model: Loaded speech recognition model
      processor: Model processor
      pipe: Inference pipeline
      dataset: Loaded input dataset
      text_predict_: Dataset with model predictions
      result_dataset: Final combined dataset

    Methods:
      load_config(config_path: str) -> dict:
          Loads configuration from YAML file.

      transcribe() -> None:
          Processes audio data and generates transcriptions.

      make_result_dataset() -> None:
          Creates final dataset combining original data and predictions.

      push_HF() -> None:
          Pushes result dataset to Hugging Face Hub.

    Example:
      >>> benchmark = AutoBenchmark('config.yaml')
      # This will:
      # 1. Load the dataset and model
      # 2. Generate transcriptions
      # 3. Create result dataset
      # 4. Push to Hugging Face Hub

    Notes:
      - Requires GPU for optimal performance
      - Supports batch processing for efficiency
      - Automatically handles device placement
      - Currently uses float32 precision
    """


    def __init__(self, config:str = 'config.yaml'):
        self.config = self.load_config(config)
        self.batch_size = self.config['batch_size']
        self.model_id = self.config['model_id']
        self.dataset_config = self.config['dataset']
        self.result_datasset_config = self.config['result_dataset']

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32 #torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True, use_safetensors=True)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

        self.dataset = load_dataset(self.dataset_config['reponame'])
        if self.dataset_config['split']:
            self.dataset = self.dataset[self.dataset_config['split']]

        self.transcribe()
        self.make_result_dataset()
        self.push_HF()

  
    def load_config(self, config_path:str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

  
    def transcribe(self):
        text_predict = []
        for batch in tqdm(self.dataset.iter(batch_size=self.batch_size), desc='Batch'):
            batch_audio = []
            for sample in batch['audio']:
                batch_audio.append(np.array(sample['array']))
            text_predict.extend(self.pipe(batch_audio, batch_size = self.batch_size))
        self.text_predict_ = Dataset.from_list(text_predict)

    def make_result_dataset(self):
        columns_data = dataset.select_columns(['id',
                                               'raw_transcription',
                                               'transcription',
                                               'dataset_name',
                                               'augmentation_type',
                                               'snr'])
        self.text_predict_ = self.text_predict_.rename_column('text', 'text_predict')
        self.result_dataset = concatenate_datasets([columns_data,
                                                      self.text_predict_], axis=1)
        
    def push_HF(self):
        dataset_repo_name = self.result_datasset_config['reponame']
        private = self.result_datasset_config['private']
        self.result_dataset.push_to_hub(dataset_repo_name, private=private)


benchmark = AutoBenchmark()
