from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import torch
from tqdm import tqdm
import yaml
from multiprocessing import Process

class ChunkBenchmark:

    def __init__(self, config:dict, data_chunk, device:int = None):
        print(f'_______ CREATING MODEL FOR {device} GPU ________')
        self.process_name = f'Process_{device}'
        self.config = config
        self.batch_size = self.config['batch_size']
        self.model_id = self.config['model_id']

        if config['accelerator'] == 'gpu' and torch.cuda.is_available():
            self.device = f"cuda:{device}"
        else:
            self.device = "cpu"
            print(f"GPU requested but not available, using CPU instead")
        
        self.torch_dtype = torch.float32
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

        self.dataset = data_chunk


    def transcribe(self):
        print(f'===START: {self.process_name}')
        text_predict = []
        for batch in tqdm(self.dataset.iter(batch_size=self.batch_size), desc='Batch'):
            batch_audio = []
            for sample in batch['audio']:
                batch_audio.append(np.array(sample['array']))
            text_predict.extend(self.pipe(batch_audio, batch_size = self.batch_size))
        self.text_predict_ = Dataset.from_list(text_predict)

    def __call__(self):

        self.transcribe()
        columns_data = self.dataset.select_columns(['id',
                                                    'raw_transcription',
                                                    'transcription',
                                                    'dataset_name',
                                                    'augmentation_type',
                                                    'snr'])
        self.text_predict_ = self.text_predict_.rename_column('text', 'text_predict')
        print(f'=======END: {self.process_name}')
        return concatenate_datasets([columns_data, self.text_predict_], axis=1)


class ProcessBenchmark:

    def __init__(self, config:str = 'config.yaml', chunk_benchmark = ChunkBenchmark):

        self.config = self.load_config(config)
        self.print_config(self.config)
        self.batch_size = self.config['batch_size']
        self.dataset_config = self.config['dataset']
        self.result_datasset_config = self.config['result_dataset']

        self.dataset = load_dataset(self.dataset_config['reponame'])
        if self.dataset_config['split']:
            self.dataset = self.dataset[self.dataset_config['split']]
        
        self.chunk_benchmark = chunk_benchmark
        # self.res_dataset_list = []

        self.create_process()



    def load_config(self, config_path:str)->dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
        
    def print_config(self, config: dict, indent: int = 4) -> None:
        """
        Recursively prints configuration with proper indentation.
        
        Args:
            config: Configuration dictionary
            indent: Current indentation level
        """
        print("\n" + "="*50)
        print("Configuration:")
        print("="*50)
    
        def _print_recursive(d: dict, indent: int = 4):
            for key, value in d.items():
                indent_str = "  " * indent
                if isinstance(value, dict):
                    print(f"{indent_str}{key}:")
                    _print_recursive(value, indent + 1)
                else:
                    print(f"{indent_str}{key}: {value}")
    
        _print_recursive(config)
        print("\n" + "="*50 + "\n")

    def create_process(self):
        
        if self.config['accelerator'] == 'gpu':
            if isinstance(self.config['devices'], list):
                self.num_gpu = len(self.config['devices'])
                processes = []

                for dev in self.config['devices']:
                    dev = int(dev)
                    chunk = self.dataset.shard(num_shards=self.num_gpu, index=dev)

                    process = Process(
                    target = self.run_one_process,
                    args=(
                        self.config,
                        chunk,
                        dev
                        )
                    )

                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

            else:
                dev = 0
                self.run_one_process(self.config, self.dataset, dev)
        else:
            print('No GPU')

    def run_one_process(self, config:dict, data_chunk, device:int)->None:

        processor = self.chunk_benchmark(config, data_chunk, device)
        # self.res_dataset_list.append(processor())
        result = processor()
        result.save_to_disk(f"temp_result_{device}")
        print(f'======FILE temp_result_{device} SAVED!======')


    def collect_results(self):
        
        if isinstance(self.config['devices'], list):
            results = []
            for dev in self.config['devices']:
                dataset = Dataset.load_from_disk(f"temp_result_{dev}")
                results.append(dataset)
            return concatenate_datasets(results)
        else:
            try:
                return Dataset.load_from_disk("temp_result_0")
            except Exception:
                print('NOT DATASET (((')

    def push_HF(self):

        # if len(self.res_dataset_list) > 1:
        #     result_dataset = concatenate_datasets(self.res_dataset_list)
        # elif len(self.res_dataset_list) == 1:
        #     result_dataset = self.res_dataset_list[0]
        # else:
        #     print('NO DATASET TO PUSH HUB')
        result_dataset = self.collect_results()
        dataset_repo_name = self.result_datasset_config['reponame']
        private = self.result_datasset_config['private']
        result_dataset.push_to_hub(dataset_repo_name, private=private)

if __name__ == "__main__":
    process = ProcessBenchmark()
    print('\n==== SAVING TO HF ====')
    process.push_HF()

