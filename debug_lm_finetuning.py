# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import regex as re
import shutil
import time

import numpy
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
from dataclasses import dataclass
from fastai.basics import *

from run_generation import sample_sequence

from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule, WarmupConstantSchedule, WarmupCosineWithHardRestartsSchedule,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

from sp_encoder import SPEncoder
from yt_encoder import YTEncoder

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from filelock import FileLock
import contextlib

logger = logging.getLogger(__name__)

def log_info(*args, **kwargs):
    if xm.is_master_ordinal():
        logger.info(*args, **kwargs)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

@dataclass
class MovingLoss():
    steps:int=1000
    avg_loss = (0.0, 0.0)
    def add(self, batch_loss:float):
        k_s = 1 - 1/self.steps
        avg_loss = self.avg_loss
        self.avg_loss = (self.avg_loss[0] * k_s + batch_loss * (1-k_s),
                         self.avg_loss[1] * k_s + 1.0 * (1-k_s))
    @property
    def loss(self):
        if self.avg_loss[1]:
            return self.avg_loss[0] / self.avg_loss[1]

def print_sample(model, tokenizer, device, args):
    model.eval()
    raw_text = """ На словах ты Лев Толстой,\n А на деле -"""
    context_tokens = tokenizer.encode(raw_text)
    out = sample_sequence(
        model=model,
        context=context_tokens,
        length=500,
        temperature=1,
        top_k=0,
        top_p=0.9,
        device=device,
        max_input=0
        #is_xlnet=bool(args.model_type == "xlnet"),
    )
    out = out[0, len(context_tokens):].tolist()
    text = raw_text + tokenizer.decode(out)
    log_info(text)
    
    if xm.is_master_ordinal():
        with open(os.path.join(args.output_dir, 'sample.txt'), 'w') as f: 
            f.write(text)
    
    model.train()

class TextDataset(Dataset):
    @staticmethod
    def process_file(file_path, tokenizer, block_size, shuffle):
        directory, filename = os.path.split(file_path)
        directory = os.path.join(directory, 'cached')
        os.makedirs(directory, exist_ok=True)
        cached_features_file = os.path.join(directory, f'cached_lm_{block_size}_{tokenizer.hash}_{filename}')

        if os.path.exists(cached_features_file):
            with open(cached_features_file, 'rb') as handle:
                tokenized_text = pickle.load(handle)
        else:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            if hasattr(tokenizer, 'encode'):
                tokenized_text = tokenizer.encode(text)
            else: 
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(tokenized_text, handle, protocol=pickle.HIGHEST_PROTOCOL)

        examples = []
        # add random shift 
        max_shift = max(min(block_size, len(tokenized_text) - block_size), 0)
        rnd_shift = random.randrange(max_shift) if max_shift and shuffle else 0

        for i in range(rnd_shift, len(tokenized_text)-block_size+1, block_size):
            examples.append(tokenizer.add_special_tokens_single_sentence(tokenized_text[i:i+block_size]))
        # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should loook for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.
        return examples

    def __init__(self, tokenizer, file_path='train', args=None, shuffle=True):
        if not hasattr(tokenizer, 'hash'): tokenizer.hash = ''

        log_info(f"Loading features from {file_path}")
        if os.path.isfile(file_path):
            files = [file_path]
        else:
            assert os.path.isdir(file_path)
            files =  glob.glob(os.path.join(file_path, '*.txt'))
        
        files = sorted(files)
        if shuffle: random.shuffle(files)
        # The dataset can be big, like 230G big. Also, if you train on TPU you need a copy for each of 8 cores. 
        # That is why we take a sample and then do resampling each args.reload_data_file epochs.
        # Even if dataset isn't that big it's still good because there is a random shift during dataloading. You can 
        # consider it as a data augmentation technique. 
        # In case of TPU, you need to make sure initial random seed is the same for each process or TPU will freeze because of 
        # different datasets.
        files = files[:100]

        self.examples = []
        
        for fn in tqdm(files, disable=not xm.is_master_ordinal()):
            self.examples.extend(self.process_file(fn, tokenizer, args.block_size, shuffle))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, file_path=args.eval_data_file if evaluate else args.train_data_file, args=args, shuffle=not evaluate)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        log_info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

# from transformers/modeling_utils.py, adapted to tpu
def save_pretrained(model, save_directory):
    """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
    """
    assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

    # Only save the model it-self if we are using distributed training
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save configuration file
    model_to_save.config.save_pretrained(save_directory)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

    xm.save(model_to_save.state_dict(), output_model_file)
    '''
    def movecpu(obj):
        if hasattr(obj,'cpu'):
            return obj.cpu()
        elif hasattr(obj, '__dict__'):
            return {k:movecpu(v) for k,v in obj.__dict__.items()}
        else: 
            return obj

    torch.save(movecpu(model_to_save), os.path.join(save_directory, 'debug2.bin'))
    '''
    log_info(f"Model weights saved in {output_model_file}")

def save_state(args, model, tokenizer, global_step):
    def save_dir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log_info(f"Saving model checkpoint to {output_dir}")
        save_pretrained(model, output_dir)
        tokenizer.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        with open(os.path.join(output_dir, 'step.txt'), 'w') as c: c.write(str(global_step))
    
    if xm.is_master_ordinal():
        save_dir(args.output_dir)
        checkpoint_prefix = 'checkpoint'
        output_dir = os.path.join(args.output_dir, f'{checkpoint_prefix}-{global_step}')
        save_dir(output_dir)
        _rotate_checkpoints(args, checkpoint_prefix)

class SummaryWriterP(SummaryWriter):
    def __init__(self, prefix=None, logdir=None, comment='', *args, **kwargs):
        if prefix:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(prefix, 
                'runs', current_time + '_' + socket.gethostname() + comment)
        super().__init__(logdir, comment, *args, **kwargs) 

def build_dataloader(args, tokenizer):
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    train_sampler = RandomSampler(train_dataset) 
    if xm.xrt_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset,
                            num_replicas=xm.xrt_world_size(),
                            rank=xm.get_ordinal(),
                            shuffle=True)
    return DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

def train(args, model, tokenizer):
    """ Train the model """
    if xm.is_master_ordinal():
        tb_writer = SummaryWriterP(args.output_dir)

    def summary_write(*args, **kwargs):
        if xm.is_master_ordinal():
            tb_writer.add_scalar(*args, **kwargs)

    args.train_batch_size = args.per_gpu_train_batch_size #* max(1, args.n_gpu)

    train_dataloader = build_dataloader(args, tokenizer)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) ) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # Scale learning rate to num cores
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate * xm.xrt_world_size(), eps=args.adam_epsilon)
    warmup_steps = args.warmup_samples // (args.train_batch_size * xm.xrt_world_size())
    if args.lr_decay:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    elif args.lr_cosine:
        scheduler = WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total, cycles=args.num_train_epochs)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)

    # Train!
    tracker = xm.RateTracker()
    log_info("***** Running training *****")
    log_info("  Num Epochs = %d", args.num_train_epochs)
    log_info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    log_info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size *  (xm.xrt_world_size() if args.local_rank != -1 else 1))
    log_info("  Total optimization steps = %d", t_total)

    try:
        with open(os.path.join(args.model_name_or_path, 'step.txt'), 'r') as c: 
            global_step = int(c.readline())
    except OSError as e:
        global_step = 0

    moving_loss = MovingLoss(1000//args.logging_steps)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=not xm.is_master_ordinal())
    try:    
        for epoch in train_iterator:
            p_train_dataloader = pl.ParallelLoader(train_dataloader, [args.device])
            epoch_iterator = tqdm(p_train_dataloader.per_device_loader(args.device), total=len(train_dataloader), desc="Iteration", disable=not xm.is_master_ordinal())

            model.train()
            for step, batch in enumerate(epoch_iterator):
                optimizer.zero_grad()
                outputs = model(batch, labels=batch)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                loss.backward()
                xm.optimizer_step(optimizer)
                scheduler.step()  
                
                if step > 100:
                    epoch_iterator.close()
                    break
                

            # evaluate once in an epoch    
            if args.evaluate_during_training: 
                log_info(f"Eval {evaluate(args, model, tokenizer)}")
                        
    except (KeyboardInterrupt, SystemExit):
        save_state(args, model, tokenizer, global_step)
        raise

    save_state(args, model, tokenizer, global_step)

    return global_step, moving_loss.loss


def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size 
    eval_dataloader = pl.ParallelLoader(DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False), [args.device])

    # Eval!
    log_info("***** Running evaluation {} *****".format(prefix))
    log_info("  Num examples = %d", len(eval_dataset))
    log_info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    eval_loss =  torch.tensor([0.0])
   
    nb_eval_steps = 0
    model.eval()
    outputs = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader.per_device_loader(args.device), desc="Evaluating", disable=not xm.is_master_ordinal()):
            output = model(batch, labels=batch)
            outputs.append(output[0])

    eval_loss = torch.stack(outputs).mean()
    perplexity = torch.exp(eval_loss.cpu())

    result = {
        "perplexity": perplexity
    }
    return result

lock = None

import tempfile
from xla.test.test_operations import XlaTestCase

class TC(XlaTestCase):
  def assertEqual(self, x, y, prec=None, message='', allow_inf=False):
    if isinstance(prec, str) and message == '':
      message = prec
      prec = None
    if prec is None:
      prec = self.PRECISION
    if isinstance(x, torch.Tensor) and isinstance(y, Number):
      self.assertEqual(
          x.item(), y, prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(y, torch.Tensor) and isinstance(x, Number):
      self.assertEqual(
          x, y.item(), prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(x, torch.Tensor) and isinstance(y, numpy.bool_):
      self.assertEqual(
          x.item(), y, prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(y, torch.Tensor) and isinstance(x, numpy.bool_):
      self.assertEqual(
          x, y.item(), prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):

      def assertTensorsEqual(a, b):
        a = a.cpu()
        b = b.cpu()
        super(XlaTestCase, self).assertEqual(a.size(), b.size(), message)
        if a.numel() > 0:
          if (a.device.type == 'cpu' and
              (a.dtype == torch.float16 or a.dtype == torch.bfloat16)):
            # CPU half and bfloat16 tensors don't have the methods we need below
            a = a.to(torch.float32)
          b = b.to(a)

          if (a.dtype == torch.bool) != (b.dtype == torch.bool):
            raise TypeError('Was expecting both tensors to be bool type.')
          else:
            if a.dtype == torch.bool and b.dtype == torch.bool:
              # we want to respect precision but as bool doesn't support substraction,
              # boolean tensor has to be converted to int
              a = a.to(torch.int)
              b = b.to(torch.int)

            diff = a - b
            if a.is_floating_point():
              # check that NaNs are in the same locations
              nan_mask = torch.isnan(a)
              self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
              diff[nan_mask] = 0
              # inf check if allow_inf=True
              if allow_inf:
                inf_mask = torch.isinf(a)
                inf_sign = inf_mask.sign()
                self.assertTrue(
                    torch.equal(inf_sign,
                                torch.isinf(b).sign()), message)
                diff[inf_mask] = 0
            # TODO: implement abs on CharTensor (int8)
            if diff.is_signed() and diff.dtype != torch.int8:
              diff = diff.abs()
            max_err = diff.max()
            self.assertLessEqual(max_err, prec, message)

      super(XlaTestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
      super(XlaTestCase, self).assertEqual(x.is_quantized, y.is_quantized,
                                           message)
      if x.is_sparse:
        x = self.safeCoalesce(x)
        y = self.safeCoalesce(y)
        assertTensorsEqual(x._indices(), y._indices())
        assertTensorsEqual(x._values(), y._values())
      elif x.is_quantized and y.is_quantized:
        self.assertEqual(
            x.qscheme(),
            y.qscheme(),
            prec=prec,
            message=message,
            allow_inf=allow_inf)
        if x.qscheme() == torch.per_tensor_affine:
          self.assertEqual(
              x.q_scale(),
              y.q_scale(),
              prec=prec,
              message=message,
              allow_inf=allow_inf)
          self.assertEqual(
              x.q_zero_point(),
              y.q_zero_point(),
              prec=prec,
              message=message,
              allow_inf=allow_inf)
        elif x.qscheme() == torch.per_channel_affine:
          self.assertEqual(
              x.q_per_channel_scales(),
              y.q_per_channel_scales(),
              prec=prec,
              message=message,
              allow_inf=allow_inf)
          self.assertEqual(
              x.q_per_channel_zero_points(),
              y.q_per_channel_zero_points(),
              prec=prec,
              message=message,
              allow_inf=allow_inf)
          self.assertEqual(
              x.q_per_channel_axis(),
              y.q_per_channel_axis(),
              prec=prec,
              message=message)
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(
            x.int_repr().to(torch.int32),
            y.int_repr().to(torch.int32),
            prec=prec,
            message=message,
            allow_inf=allow_inf)
      else:
        assertTensorsEqual(x, y)
    elif isinstance(x, self.STRING_CLASSES) and isinstance(
        y, self.STRING_CLASSES):
      super(XlaTestCase, self).assertEqual(x, y, message)
    elif type(x) == set and type(y) == set:
      super(XlaTestCase, self).assertEqual(x, y, message)
    elif isinstance(x, dict) and isinstance(y, dict):
        self.assertEqual(
            set(x.keys()),
            set(y.keys()),
            prec=prec,
            message=message,
            allow_inf=allow_inf)
        key_list = list(x.keys())
        for k in key_list:
            xm.master_print(k)
            try:
                self.assertEqual(x[k], y[k],
                                    prec=prec,
                                    message=message,
                                    allow_inf=allow_inf)
                xm.master_print('ok')
            except Exception as e:
                xm.master_print(e)
                xm.master_print(x[k])
                xm.master_print(y[k])
    elif _is_iterable(x) and _is_iterable(y):
      super(XlaTestCase, self).assertEqual(len(x), len(y), message)
      for x_, y_ in zip(x, y):
        self.assertEqual(
            x_, y_, prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(x, bool) and isinstance(y, bool):
      super(XlaTestCase, self).assertEqual(x, y, message)
    elif isinstance(x, Number) and isinstance(y, Number):
      if abs(x) == math.inf or abs(y) == math.inf:
        if allow_inf:
          super(XlaTestCase, self).assertEqual(x, y, message)
        else:
          self.fail('Expected finite numeric values - x={}, y={}'.format(x, y))
        return
      super(XlaTestCase, self).assertLessEqual(abs(x - y), prec, message)
    else:
      super(XlaTestCase, self).assertEqual(x, y, message)    

def main(index):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--reload_data_file", default=None, type=int,
                        help="Reload dataset every X epoch")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_class", default="", type=str,
                        help="Optional pretrained tokenizer clas")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument('--eval_steps', type=int, default=100,
                        help="Evaluate every X updates steps.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_samples", default=0, type=int,
                        help="Linear warmup over warmup_samples.")
    parser.add_argument("--lr_decay", action='store_true',
                        help="Decay LR using WarmupLinearSchedule.")
    parser.add_argument("--lr_cosine", action='store_true',
                        help="LR using WarmupCosineWithHardRestartsSchedule.")

    parser.add_argument("--unfreeze_level", default=-1, type=int,
                        help="If > 0: freeze all layers except few first and last.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--first_run', action='store_true',
                        help="Cache init")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()
    args.local_rank = index

    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    args.n_gpu = xm.xrt_world_size()
    args.device = xm.xla_device()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if xm.is_master_ordinal() else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    # That is actually very important in case of distributed environment (like TPU). You need same dataset on every node/process. 
    # If you have randomness in dataset creation (like I do) you need to set the same seed in every process.
    set_seed(args)
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # load model from web in single thread or file will be corrupted. 
    lock = FileLock("the.lock") if args.first_run else contextlib.suppress()

    with lock:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        if args.tokenizer_class: tokenizer_class = globals()[args.tokenizer_class]
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        # from_pretrained loads something in a weird way, so that is the fix
        #if os.path.exists(args.model_name_or_path):
        #    model.load_state_dict(torch.load('output/classic_s/pytorch_model.bin'))
    model = model.to(args.device)

    def req_len(model):
        return len([param for item in flatten_model(model) 
                     for param in item.parameters()
                        if param.requires_grad])
    '''
    # freeze all layers but few first and last
    if args.unfreeze_level >= 0:
        b_req_len = req_len(model)
        flat = flatten_model(model)
        flat = [item for item in flat if list(item.parameters())]
        i_start = 3
        i_end = 1
        need_grads = set(flat[:i_start+args.unfreeze_level*3]) | set(flat[-(i_end+args.unfreeze_level*3):])
        for item in flat:
            requires_grad(item, item in need_grads)
        log_info(f"Num of layers before {b_req_len}, after freeze {req_len(model)}")
    '''    
    log_info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train(args, model, tokenizer)

    
    xla_device = xm.xla_device()
    results = evaluate(args, model, tokenizer, "checkpoint-1")
    log_info(f"Eval1 {results}")

    xm.save(model.state_dict(), 'tf4.bin')
    #torch.save(model.state_dict(), 'tf4.bin')
    #time.sleep(60) # for multiprocessing
    state_dict = torch.load('tf4.bin').cpu()

    print(state_dict['transformer.wte.weight'])
    print(state_dict['transformer.wte.weight'].shape)
    
    cpu_model = model_class(config=config)
    cpu_model.load_state_dict(state_dict)
    print(cpu_model._load_state_dict_pre_hooks)
    print(cpu_model._state_dict_hooks)
    print(cpu_model.state_dict()['transformer.wte.weight'])
    print(cpu_model.state_dict()['transformer.wte.weight'].shape)
    #TC().assertEqual(model.state_dict(), cpu_model.state_dict())

    #loaded_model = cpu_model.to(xla_device)

    #results = evaluate(args, loaded_model, tokenizer, "checkpoint-2")
    #log_info(f"Eval2 {results}")

    #TC().assertEqual(model.state_dict(), loaded_model.state_dict())


if __name__ == '__main__':
    xmp.spawn(main, nprocs=1)
#    xmp.spawn(main)
    