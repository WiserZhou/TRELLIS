"""
Base Trainer Module for TRELLIS Framework

This file contains the abstract base class for training machine learning models in the TRELLIS framework.
The Trainer class provides a comprehensive foundation for training workflows with support for:
- Single and multi-GPU training via PyTorch Distributed Data Parallel (DDP)
- Checkpointing for saving and resuming training
- TensorBoard logging and visualization
- Mixed precision training
- Exponential Moving Average (EMA) of model parameters
- Efficient data loading with prefetching
- Customizable sampling and visualization of model outputs

Subclasses must implement several abstract methods to define model-specific training logic.
"""

from abc import abstractmethod
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler


class Trainer:
    """
    Base class for training.
    
    This abstract class defines the core training loop and utilities
    that are common across different training tasks. Specific training
    implementations should inherit from this class and implement the
    abstract methods.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        **kwargs
    ):
        """
        Initialize the trainer with models, dataset, and training configuration.
        
        Args:
            models: Dictionary of models to train
            dataset: Dataset to train on
            output_dir: Directory to save outputs
            load_dir: Directory to load checkpoints from
            step: Current training step (for resuming)
            max_steps: Maximum number of training steps
            batch_size: Global batch size across all GPUs
            batch_size_per_gpu: Batch size per GPU (alternative to batch_size)
            batch_split: Number of microbatches to split each batch into
            optimizer: Dictionary of optimizer configurations
            lr_scheduler: Learning rate scheduler configuration
            elastic: Configuration for elastic training
            grad_clip: Value for gradient clipping
            ema_rate: Rate for exponential moving average
            fp16_mode: Mode for mixed precision training ('inflat_all' or 'amp')
            fp16_scale_growth: Growth rate for mixed precision scaling
            finetune_ckpt: Checkpoint path for fine-tuning
            log_param_stats: Whether to log parameter statistics
            prefetch_data: Whether to prefetch data to GPU
            i_print: Interval for printing progress
            i_log: Interval for logging metrics
            i_sample: Interval for sampling/visualization
            i_save: Interval for saving checkpoints
            i_ddpcheck: Interval for checking DDP consistency
        """
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck        

        # Set up distributed training configuration
        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        # Calculate batch size parameters
        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        # Initialize models, optimizers, etc.
        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Load checkpoint or initialize from scratch
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        
        # Set up output directories and tensorboard writer on master process
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

        # Check DDP setup for multi-GPU training
        if self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)
            
    @property
    def device(self):
        """
        Get the device that the models are on.
        
        Returns:
            torch.device: The device of the first model parameter found.
        """
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and other components like optimizers, schedulers, etc.
        
        This abstract method must be implemented by subclasses to set up the
        specific models and related components needed for training.
        """
        pass
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader for training.
        
        Sets up the data sampler and dataloader with appropriate batch size,
        workers, and other configurations for efficient data loading.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        
        This method should be implemented to restore the training state
        from a saved checkpoint.
        
        Args:
            load_dir: Directory containing checkpoints
            step: Specific step to load, or 0 for latest
            
        Note: Should be called by all processes in distributed training.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        
        This method should be implemented to save the current training state
        to a checkpoint file.
        
        Note: Should be called only by the rank 0 process in distributed training.
        """
        pass
    
    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        
        This method should be implemented to load pre-trained weights
        for fine-tuning.
        
        Args:
            finetune_ckpt: Path to checkpoint for fine-tuning
            
        Note: Should be called by all processes in distributed training.
        """
        pass
    
    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        
        This method should be implemented to generate samples from the model
        for visualization.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            verbose: Whether to print verbose information
            **kwargs: Additional arguments
            
        Returns:
            dict: Dictionary of generated samples
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image for visualization.
        
        Args:
            sample: Data sample to visualize
            
        Returns:
            torch.Tensor or dict: Processed sample ready for visualization
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset for visualization.
        
        Creates a visualization of dataset samples and saves them to disk.
        
        Args:
            num_samples: Number of samples to visualize
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """
        Sample images from the model and save to disk.
        
        This function coordinates the generation of samples across all processes
        and gathers them on the master process for saving.
        
        Args:
            suffix: Suffix for the output directory name
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            verbose: Whether to print verbose information
            
        Note: This function should be called by all processes in distributed training.
        """
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks to processes for parallel generation
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)

        # Preprocess images for visualization
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}

        # Gather results from all processes
        if self.world_size > 1:
            for key in samples.keys():
                samples[key]['value'] = samples[key]['value'].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images to disk (master process only)
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    utils.save_image(
                        samples[key]['value'],
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min}, max: {max}',
                    )

        if self.is_master:
            print(' Done.')

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average of model parameters.
        
        This method should be implemented to maintain EMA versions of models
        for more stable inference.
        
        Note: Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if Distributed Data Parallel (DDP) is working properly.
        
        This method should verify that parameters are synchronized
        across processes in distributed training.
        
        Note: Should be called by all processes.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses from a minibatch of data.
        
        This method should be implemented to compute all loss components
        needed for training.
        
        Args:
            **mb_data: Minibatch data
            
        Returns:
            dict: Dictionary of loss components
        """
        pass
    
    def load_data(self):
        """
        Load a batch of data from the dataloader.
        
        If prefetching is enabled, alternates between using a pre-fetched
        batch and loading the next batch in the background.
        
        Returns:
            list: List of data dictionaries, split according to batch_split
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # Split data into multiple microbatches if needed
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a single training step.
        
        This method should be implemented to process a batch of data,
        compute losses, and update model parameters.
        
        Args:
            data_list: List of data batches
            
        Returns:
            dict: Dictionary of metrics/losses for logging
        """
        pass

    def run(self):
        """
        Run the full training loop.
        
        This method handles the main training loop, including:
        - Data loading
        - Running training steps
        - Logging metrics
        - Creating snapshots
        - Saving checkpoints
        - Monitoring progress
        """
        if self.is_master:
            print('\nStarting training...')
            self.snapshot_dataset()
        if self.step == 0:
            self.snapshot(suffix='init')
        else: # resume
            self.snapshot(suffix=f'resume_step{self.step:07d}')

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()

            # Load data and run training step
            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress at regular intervals
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check DDP synchronization at regular intervals
            if self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Generate and save sample images at regular intervals
            if self.step % self.i_sample == 0:
                self.snapshot()

            # Handle logging on master process
            if self.is_master:
                log.append((self.step, {}))

                # Log timing information
                log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log training metrics
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scaling factor for mixed precision training
                if self.fp16_mode == 'amp':
                    log[-1][1]['scale'] = self.scaler.get_scale()
                elif self.fp16_mode == 'inflat_all':
                    log[-1][1]['log_scale'] = self.log_scale

                # Write logs to file and tensorboard at regular intervals
                if self.step % self.i_log == 0:
                    ## Save to log file
                    log_str = '\n'.join([
                        f'{step}: {json.dumps(log)}' for step, log in log
                    ])
                    with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
                        log_file.write(log_str + '\n')

                    # Write to tensorboard
                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    log_show = dict_reduce(log_show, lambda x: np.mean(x))
                    log_show = dict_flatten(log_show, sep='/')
                    for key, value in log_show.items():
                        self.writer.add_scalar(key, value, self.step)
                    log = []

                # Save checkpoint at regular intervals
                if self.step % self.i_save == 0:
                    self.save()

        # Final steps after training is complete
        if self.is_master:
            self.snapshot(suffix='final')
            self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop for performance analysis.
        
        Uses PyTorch's profiling tools to collect performance metrics.
        
        Args:
            wait: Number of steps to wait before profiling
            warmup: Number of warmup steps for profiling
            active: Number of active profiling steps
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()