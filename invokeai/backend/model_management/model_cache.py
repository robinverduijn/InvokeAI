"""
Manage a RAM cache of diffusion/transformer models for fast switching.
They are moved between GPU VRAM and CPU RAM as necessary. If the cache
grows larger than a preset maximum, then the least recently used
model will be cleared and (re)loaded from disk when next needed.

The cache returns context manager generators designed to load the
model into the GPU within the context, and unload outside the
context. Use like this:

   cache = ModelCache(max_models_cached=6)
   with cache.get_model('runwayml/stable-diffusion-1-5') as SD1,
          cache.get_model('stabilityai/stable-diffusion-2') as SD2:
       do_something_in_GPU(SD1,SD2)


"""

import gc
import os
import sys
import hashlib
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Dict, Union, types, Optional, Type, Any

import torch

from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging
import invokeai.backend.util.logging as logger
from .models import ModelType, SubModelType, ModelBase

# Maximum size of the cache, in gigs
# Default is roughly enough to hold three fp16 diffusers models in RAM simultaneously
DEFAULT_MAX_CACHE_SIZE = 6.0

# actual size of a gig
GIG = 1073741824

class ModelLocker(object):
    "Forward declaration"
    pass

class ModelCache(object):
    "Forward declaration"
    pass

class SDModelInfo(object):
    """Forward declaration"""
    pass

class _CacheRecord:
    size: int
    model: Any
    cache: ModelCache
    _locks: int

    def __init__(self, cache, model: Any, size: int):
        self.size = size
        self.model = model
        self.cache = cache
        self._locks = 0

    def lock(self):
        self._locks += 1

    def unlock(self):
        self._locks -= 1
        assert self._locks >= 0

    @property
    def locked(self):
        return self._locks > 0

    @property
    def loaded(self):
        if self.model is not None and hasattr(self.model, "device"):
            return self.model.device != self.cache.storage_device
        else:
            return False
    

class ModelCache(object):
    def __init__(
        self,
        max_cache_size: float=DEFAULT_MAX_CACHE_SIZE,
        execution_device: torch.device=torch.device('cuda'),
        storage_device: torch.device=torch.device('cpu'),
        precision: torch.dtype=torch.float16,
        sequential_offload: bool=False,
        lazy_offloading: bool=True,
        sha_chunksize: int = 16777216,
        logger: types.ModuleType = logger
    ):
        '''
        :param max_models: Maximum number of models to cache in CPU RAM [4]
        :param execution_device: Torch device to load active model into [torch.device('cuda')]
        :param storage_device: Torch device to save inactive model in [torch.device('cpu')]
        :param precision: Precision for loaded models [torch.float16]
        :param lazy_offloading: Keep model in VRAM until another model needs to be loaded
        :param sequential_offload: Conserve VRAM by loading and unloading each stage of the pipeline sequentially
        :param sha_chunksize: Chunksize to use when calculating sha256 model hash
        '''
        #max_cache_size = 9999
        execution_device = torch.device('cuda')

        self.model_infos: Dict[str, SDModelInfo] = dict()
        self.lazy_offloading = lazy_offloading
        #self.sequential_offload: bool=sequential_offload
        self.precision: torch.dtype=precision
        self.max_cache_size: int=max_cache_size
        self.execution_device: torch.device=execution_device
        self.storage_device: torch.device=storage_device
        self.sha_chunksize=sha_chunksize
        self.logger = logger

        self._cached_models = dict()
        self._cache_stack = list()

    def get_key(
        self,
        model_path: str,
        model_type: ModelType,
        submodel_type: Optional[ModelType] = None,
    ):

        key = f"{model_path}:{model_type}"
        if submodel_type:
            key += f":{submodel_type}"
        return key

    #def get_model(
    #    self,
    #    repo_id_or_path: Union[str, Path],
    #    model_type: ModelType = ModelType.Diffusers,
    #    subfolder: Path = None,
    #    submodel: ModelType = None,
    #    revision: str = None,
    #    attach_model_part: Tuple[ModelType, str] = (None, None),
    #    gpu_load: bool = True,
    #) -> ModelLocker:  # ?? what does it return
    def _get_model_info(
        self,
        model_path: str,
        model_class: Type[ModelBase],
    ):
        model_info_key = self.get_key(
            model_path=model_path,
            model_type=model_class,
            submodel_type=None,
        )

        if model_info_key not in self.model_infos:
            self.model_infos[model_info_key] = model_class(
                model_path,
                model_class,
            )

        return self.model_infos[model_info_key]

    # TODO: args
    def get_model(
        self,
        model_path: Union[str, Path],
        model_class: Type[ModelBase],
        submodel: Optional[SubModelType] = None,
        gpu_load: bool = True,
    ) -> Any:

        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not os.path.exists(model_path):
            raise Exception(f"Model not found: {model_path}")

        model_info = self._get_model_info(
            model_path=model_path,
            model_class=model_class,
        )
        key = self.get_key(
            model_path=model_path,
            model_type=model_class, # TODO:
            submodel_type=submodel,
        )

        # TODO: lock for no copies on simultaneous calls?
        cache_entry = self._cached_models.get(key, None)
        if cache_entry is None:
            self.logger.info(f'Loading model {model_path}, type {model_class}:{submodel}')

            # this will remove older cached models until
            # there is sufficient room to load the requested model
            self._make_cache_room(model_info.get_size(submodel))

            # clean memory to make MemoryUsage() more accurate
            gc.collect()
            model = model_info.get_model(submodel, torch_dtype=self.precision)
            if mem_used := model_info.get_size(submodel):
                self.logger.debug(f'CPU RAM used for load: {(mem_used/GIG):.2f} GB')

            cache_entry = _CacheRecord(self, model, mem_used)
            self._cached_models[key] = cache_entry

        with suppress(Exception):
            self._cache_stack.remove(key)
        self._cache_stack.append(key)

        return self.ModelLocker(self, key, cache_entry.model, gpu_load)

    class ModelLocker(object):
        def __init__(self, cache, key, model, gpu_load):
            self.gpu_load = gpu_load
            self.cache = cache
            self.key = key
            self.model = model

        def __enter__(self) -> Any:
            if not hasattr(self.model, 'to'):
                return self.model

            cache_entry = self.cache._cached_models[self.key]

            # NOTE that the model has to have the to() method in order for this
            # code to move it into GPU!
            if self.gpu_load:
                cache_entry.lock()

                try:
                    if self.cache.lazy_offloading:
                       self.cache._offload_unlocked_models()
                       
                    if self.model.device != self.cache.execution_device:
                        self.cache.logger.debug(f'Moving {self.key} into {self.cache.execution_device}')
                        with VRAMUsage() as mem:
                            self.model.to(self.cache.execution_device)  # move into GPU
                        self.cache.logger.debug(f'GPU VRAM used for load: {(mem.vram_used/GIG):.2f} GB')
                        
                    self.cache.logger.debug(f'Locking {self.key} in {self.cache.execution_device}')                
                    self.cache._print_cuda_stats()

                except:
                    cache_entry.unlock()
                    raise

            
            # TODO: not fully understand
            # in the event that the caller wants the model in RAM, we
            # move it into CPU if it is in GPU and not locked
            elif cache_entry.loaded and not cache_entry.locked:
                self.model.to(self.cache.storage_device)

            return self.model

        def __exit__(self, type, value, traceback):
            if not hasattr(self.model, 'to'):
                return

            cache_entry = self.cache._cached_models[self.key]
            cache_entry.unlock()
            if not self.cache.lazy_offloading:
                self.cache._offload_unlocked_models()
                self.cache._print_cuda_stats()


    def model_hash(
        self,
        model_path: Union[str, Path],
    ) -> str:
        '''
        Given the HF repo id or path to a model on disk, returns a unique
        hash. Works for legacy checkpoint files, HF models on disk, and HF repo IDs
        :param model_path: Path to model file/directory on disk.
        '''
        return self._local_model_hash(model_path)

    def cache_size(self) -> float:
        "Return the current size of the cache, in GB"
        current_cache_size = sum([m.size for m in self._cached_models.values()])
        return current_cache_size / GIG

    def _has_cuda(self) -> bool:
        return self.execution_device.type == 'cuda'

    def _print_cuda_stats(self):
        vram = "%4.2fG" % (torch.cuda.memory_allocated() / GIG)
        ram = "%4.2fG" % self.cache_size()

        cached_models = 0
        loaded_models = 0
        locked_models = 0
        for model_info in self._cached_models.values():
            cached_models += 1
            if model_info.loaded:
                loaded_models += 1
            if model_info.locked:
                locked_models += 1

        self.logger.debug(f"Current VRAM/RAM usage: {vram}/{ram}; cached_models/loaded_models/locked_models/ = {cached_models}/{loaded_models}/{locked_models}")


    def _make_cache_room(self, model_size):
        # calculate how much memory this model will require
        #multiplier = 2 if self.precision==torch.float32 else 1
        bytes_needed = model_size
        maximum_size = self.max_cache_size * GIG  # stored in GB, convert to bytes
        current_size = sum([m.size for m in self._cached_models.values()])

        if current_size + bytes_needed > maximum_size:
            self.logger.debug(f'Max cache size exceeded: {(current_size/GIG):.2f}/{self.max_cache_size:.2f} GB, need an additional {(bytes_needed/GIG):.2f} GB')

        self.logger.debug(f"Before unloading: cached_models={len(self._cached_models)}")

        pos = 0
        while current_size + bytes_needed > maximum_size and pos < len(self._cache_stack):
            model_key = self._cache_stack[pos]
            cache_entry = self._cached_models[model_key]

            refs = sys.getrefcount(cache_entry.model)

            device = cache_entry.model.device if hasattr(cache_entry.model, "device") else None
            self.logger.debug(f"Model: {model_key}, locks: {cache_entry._locks}, device: {device}, loaded: {cache_entry.loaded}, refs: {refs}")

            # 2 refs:
            # 1 from cache_entry
            # 1 from getrefcount function
            if not cache_entry.locked and refs <= 2:
                self.logger.debug(f'Unloading model {model_key} to free {(model_size/GIG):.2f} GB (-{(cache_entry.size/GIG):.2f} GB)')
                current_size -= cache_entry.size
                del self._cache_stack[pos]
                del self._cached_models[model_key]
                del cache_entry

            else:
                pos += 1

        gc.collect()
        torch.cuda.empty_cache()

        self.logger.debug(f"After unloading: cached_models={len(self._cached_models)}")


    def _offload_unlocked_models(self):
        for model_key, cache_entry in self._cached_models.items():
            if not cache_entry.locked and cache_entry.loaded:
                self.logger.debug(f'Offloading {model_key} from {self.execution_device} into {self.storage_device}')
                cache_entry.model.to(self.storage_device)
        
    def _local_model_hash(self, model_path: Union[str, Path]) -> str:
        sha = hashlib.sha256()
        path = Path(model_path)
        
        hashpath = path / "checksum.sha256"
        if hashpath.exists() and path.stat().st_mtime <= hashpath.stat().st_mtime:
            with open(hashpath) as f:
                hash = f.read()
            return hash
        
        self.logger.debug(f'computing hash of model {path.name}')
        for file in list(path.rglob("*.ckpt")) \
            + list(path.rglob("*.safetensors")) \
            + list(path.rglob("*.pth")):
            with open(file, "rb") as f:
                while chunk := f.read(self.sha_chunksize):
                    sha.update(chunk)
        hash = sha.hexdigest()
        with open(hashpath, "w") as f:
            f.write(hash)
        return hash

class SilenceWarnings(object):
    def __init__(self):
        self.transformers_verbosity = transformers_logging.get_verbosity()
        self.diffusers_verbosity = diffusers_logging.get_verbosity()
        
    def __enter__(self):
        transformers_logging.set_verbosity_error()
        diffusers_logging.set_verbosity_error()
        warnings.simplefilter('ignore')

    def __exit__(self,type,value,traceback):
        transformers_logging.set_verbosity(self.transformers_verbosity)
        diffusers_logging.set_verbosity(self.diffusers_verbosity)
        warnings.simplefilter('default')

class VRAMUsage(object):
    def __init__(self):
        self.vram = None
        self.vram_used = 0
        
    def __enter__(self):
        self.vram = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        self.vram_used = torch.cuda.memory_allocated() - self.vram
