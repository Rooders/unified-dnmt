"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import sre_constants
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.constants as Constants
from utils.misc import use_gpu
from utils.statistics import Statistics


def build_loss_compute(model, tgt_vocab, src_vocab, opt, train=True):
  """
  Returns a LossCompute subclass which wraps around an nn.Module subclass
  (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
  object allows this loss to be computed in shards and passes the relevant
  data to a Statistics object which handles training/validation logging.
  Currently, the NMTLossCompute class handles all loss computation except
  for when using a copy mechanism. Despite their name, LossCompute objects
  do not merely compute the loss but also perform the backward pass inside
  their sharded_compute_loss method.
  """
  device = torch.device("cuda" if use_gpu(opt) else "cpu")

  padding_idx = tgt_vocab.stoi[Constants.PAD_WORD]
  src_padding_idx = src_vocab.stoi[Constants.PAD_WORD]
  if opt.label_smoothing > 0 and train:
    criterion = LabelSmoothingLoss(
      opt.label_smoothing, len(tgt_vocab), ignore_index=padding_idx
    )
    # if opt.src_mlm:
    #   src_criterion = LabelSmoothingLoss(
    #   opt.label_smoothing, len(src_vocab), ignore_index=src_padding_idx
    # )
    # else:
    #   src_criterion = None
  
  else:
    criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
    # if opt.src_mlm:
    #   src_criterion = nn.NLLLoss(ignore_index=src_padding_idx, reduction='sum')
    # else:
    #   src_criterion = None
  

  # if the loss function operates on vectors of raw logits instead of
  # probabilities, only the first part of the generator needs to be
  # passed to the NMTLossCompute. At the moment, the only supported
  # loss function of this kind is the sparsemax loss.
  loss_gen = model.generator
  mlm_loss_gen = model.mlm_generator
  compute = NMTLossCompute(criterion, loss_gen, mlm_loss_gen, mlm_weight=opt.mlm_weight, distill_prob=opt.distill_prob, distill_threshold=opt.distill_threshold)
  compute.to(device)
  return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns, mlm_out=None, mlm_labels=None):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)
        
        # compute MLM loss for source side
        if mlm_labels is not None:
          _, mlm_stats = self._compute_loss(batch, mlm_out, mlm_labels, gen_type="mlm_gen")
        else:
          mlm_stats = None 


        return batch_stats, mlm_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, scaler=None,  mlm_out=None, mlm_labels=None, accum_norm=1.0):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = Statistics()
        mlm_batch_stats = Statistics() if mlm_labels is not None else None
        # all_loss = 0.0
        # range_ = (cur_trunc, cur_trunc + trunc_size)
        # shard_state = self._make_shard_state(batch, output, range_, attns)
        # for shard in shards(shard_state, shard_size, creat_graph=True):
        #     loss, stats = self._compute_loss(batch, **shard)
        #     all_loss = loss.div(float(normalization)) + all_loss
        #     batch_stats.update(stats)
        non_pad_mask = batch.tgt[1:] != self.criterion.ignore_index # [seq_len, batch]
        non_pad_idx = non_pad_mask.float().view(-1).nonzero().squeeze(1)
        loss, stats = self._compute_loss(batch, output, batch.tgt[1:], select_mask=non_pad_idx)
        loss = loss.div(float(normalization)) / accum_norm
        # compute MLM loss
        if mlm_labels is not None:
          non_pad_mask = mlm_labels != self.criterion.ignore_index # [seq_len, batch]
          non_pad_idx = non_pad_mask.float().view(-1).nonzero().squeeze(1)
          mlm_normalization = (mlm_labels != self.criterion.ignore_index).float().sum()
          mlm_loss, mlm_stats = self._compute_loss(batch, mlm_out, mlm_labels, gen_type="mlm_gen", select_mask=non_pad_idx)
          mlm_loss = mlm_loss.div(float(mlm_normalization) * accum_norm)
          mlm_batch_stats.update(mlm_stats)
        else:
          mlm_loss = 0.0
          mlm_batch_stats = None
        
        loss = self.mlm_weight * mlm_loss + (1 - self.mlm_weight) * loss
        # use auto-mixed precision
        if scaler is not None:
          scaler.scale(loss).backward()
        else:
          loss.backward()
        
        batch_stats.update(stats)
        return batch_stats, mlm_batch_stats
    
    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
  """
  With label smoothing,
  KL-divergence between q_{smoothed ground truth prob.}(w)
  and p_{prob. computed by model}(w) is minimized.
  """
  def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
    assert 0.0 < label_smoothing <= 1.0
    self.ignore_index = ignore_index
    super(LabelSmoothingLoss, self).__init__()

    smoothing_value = label_smoothing / (tgt_vocab_size - 2)
    one_hot = torch.full((tgt_vocab_size,), smoothing_value)
    one_hot[self.ignore_index] = 0
    self.register_buffer('one_hot', one_hot.unsqueeze(0))

    self.confidence = 1.0 - label_smoothing

  def forward(self, output, target):
    """
    output (FloatTensor): batch_size x n_classes
    target (LongTensor): batch_size
    """
    model_prob = self.one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
    model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

    return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
  """
  Standard NMT Loss Computation.
  """

  def __init__(self, criterion, generator, mlm_generator, mlm_weight=1.0, distill_prob=0.15, distill_threshold=0.2, normalization="sents"):
    super(NMTLossCompute, self).__init__(criterion, generator)
    
    self.mlm_generator = mlm_generator
    self.mlm_weight = mlm_weight
    self.distill_prob = distill_prob
    self.distill_threshold = distill_threshold
  def _make_shard_state(self, batch, output, range_, attns=None):
    return {
        "output": output,
        "target": batch.tgt[range_[0] + 1: range_[1]],
    }
  
  def _make_sample_state(self, target, output, range_):
    return {
        "output": output,
        "target": target[range_[0] + 1: range_[1]],
    }
  
  def compute_distillation_loss(self, tgt, nmt_prob, mlm_prob, select_prob_mask, normalization, scaler=None, annealing_coef=1.0):
    # select_prob_mask: [seq_len, batch_size] 1: use distillation 0: not use
    # nmt_prob, mlm_prob: [seq_len * batch_size, vocab_size]
    batch_stats = Statistics()
    truth = tgt[1:].contiguous().view(-1) # [seq_len-1 * batch_size]
    label_s_prob = self.criterion.one_hot.repeat(truth.size(0), 1) # [seq_len-1 * batch_size, vocab_size]
    label_s_prob.scatter_(1, truth.unsqueeze(1), self.criterion.confidence)
    label_s_prob.masked_fill_((truth == self.criterion.ignore_index).unsqueeze(1), 0) # [token_num, vocab_size]
    mlm_prob = torch.exp(mlm_prob)
    # final_truth_prob = label_s_prob
    # truth label loss of selected words
    select_prob_mask = select_prob_mask.contiguous().view(-1)
    select_idx = select_prob_mask.nonzero().squeeze(1) # [non_zero_num]
    
    # non_select_idx = (1 - select_prob_mask).nonzero().squeeze(1) # [zero_num]
    non_pad_mask = truth != self.criterion.ignore_index # [seq_len, batch]
    non_select_mask = (1-select_prob_mask).bool() & non_pad_mask
    non_select_idx = non_select_mask.float().nonzero().squeeze(1)
    distill_label_loss = F.kl_div(nmt_prob[select_idx], mlm_prob[select_idx], reduction='sum')
    masked_truth_label_loss = F.kl_div(nmt_prob[select_idx], label_s_prob[select_idx], reduction='sum')
    unmasked_truth_label_loss = F.kl_div(nmt_prob[non_select_idx], label_s_prob[non_select_idx], reduction='sum') # [token_num, vocab_size]
    
    # masked_truth_label_loss = truth_label_loss * select_prob_mask # [token_num, vocab_size]
    # unmask_truth_label_loss = truth_label_loss * (1 - select_prob_mask) # [token_num, vocab_size]
    
    
    # distill_label_loss = F.kl_div(nmt_prob, mlm_prob, reduction='none') * select_prob_mask # [token_num, vocab_size] 
    loss = unmasked_truth_label_loss + annealing_coef * masked_truth_label_loss + (1-annealing_coef) * distill_label_loss
    
    # final_truth_prob = mlm_prob * (select_prob_mask.contiguous().view(-1).unsqueeze(1)) \
    #        + label_s_prob * ((1 - select_prob_mask).contiguous().view(-1).unsqueeze(1))
    # loss = F.kl_div(nmt_prob, final_truth_prob, reduction='sum')
    stats = self._stats(loss.clone(), nmt_prob, truth)
    batch_stats.update(stats)
    loss = loss.div(float(normalization))
    if scaler is not None:
      scaler.scale(loss).backward()
    else:
      loss.backward()
    
    return batch_stats

  def only_compute_prob(self, output, target=None, select_prob=False, gen_type='tgt_gen', gen=None):
    # output: seq_len, batch_size, hidden
    # target: seq_len, batche_size
    bottled_output = self._bottle(output) # [token_num, hidden]
    if gen is None:
      gen_t = self.generator if gen_type == 'tgt_gen' else self.mlm_generator
    else:
      gen_t = gen
    
    scores = gen_t(bottled_output) # [token_num, vocab_size]
    
    if select_prob:
      # select the prob in truth
      truth_idx = target[1:].contiguous().view(-1).unsqueeze(1) # [seq_len-1 * batch_size, 1]
      padding_mask = truth_idx == self.criterion.ignore_index # [seq_len-1 * batch_size, 1]
      truth_prob = scores.gather(index=truth_idx, dim=-1) # [token_num, 1]
      truth_prob.masked_fill_(padding_mask, 0.0) # [seq_len-1 * batch_size, 1]
      distill_prob_mask = torch.exp(truth_prob) < self.distill_threshold
      distill_prob_mask = distill_prob_mask.squeeze(1).view(target[1:].size(0), -1).contiguous() #[seq_len, batch_size]
      # prob = distill_prob_mask.float().sum() / (1 - padding_mask.float()).sum() 
      # print(prob)
      return scores, distill_prob_mask.float()
    
    return scores, None


  
  
  def _compute_loss(self, batch, output, target, gen_type="tgt_gen", select_mask=None):
    bottled_output = self._bottle(output) # [token_num, hidden]
    gtruth = target.contiguous().view(-1)
    if select_mask is not None: # [non-pading token_num]
      bottled_output = bottled_output[select_mask]
      gtruth = gtruth[select_mask]
      assert(gtruth.size(0) == bottled_output.size(0))
    gen_t = self.generator if gen_type == 'tgt_gen' else self.mlm_generator    
    scores = gen_t(bottled_output)
    loss = self.criterion(scores, gtruth)
    stats = self._stats(loss.clone(), scores, gtruth)

    return loss, stats


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False, creat_graph=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads, creat_graph=creat_graph)
