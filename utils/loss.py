"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
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
  
  if opt.label_smoothing > 0:
    criterion = LabelSmoothingLoss(
      opt.label_smoothing, len(tgt_vocab), ignore_index=padding_idx
    )
  else:
    criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
  


  # if the loss function operates on vectors of raw logits instead of
  # probabilities, only the first part of the generator needs to be
  # passed to the NMTLossCompute. At the moment, the only supported
  # loss function of this kind is the sparsemax loss.
  loss_gen = model.generator
  compute = NMTLossCompute(criterion, loss_gen, opt.base_b, opt.add_s, opt.cross_task_reg)
  
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

    def __init__(self, criterion, generator, base_b, add_s, cross_task_reg=False):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator
        self.base_b = base_b
        self.add_s = add_s
        self.cross_task_reg = cross_task_reg

    @property
    def padding_idx(self):
        return self.criterion.ignore_index
    
    def src_padding_idx(self):
        return self.src_criterion.ignore_index

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

    def monolithic_compute_loss(self, batch, output, accum_norm=1.0):
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
        repair_batch_stats, trans_batch_stats = None, None
        
        if output["repair_out"] is not None:
          range_ = (0, batch.tgt.size(0))
          shard_state = self._make_shard_state(batch, output["repair_out"], range_, output["attns"])
          _, repair_batch_stats, _ = self._compute_loss(batch, **shard_state)
        
        if output["trans_out"] is not None:
          range_ = (0, batch.tgt.size(0))
          shard_state = self._make_shard_state(batch, output["trans_out"], range_, output["attns"])
          _, trans_batch_stats, _ = self._compute_loss(batch, **shard_state)
        
        return repair_batch_stats, trans_batch_stats
    
    def sharded_compute_loss(self, batch, output, 
                             cur_trunc, trunc_size, shard_size,
                             normalization, scaler=None, accum_norm=1.0, start_adaptive_training=False):
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
        trans = False
        repair = False
        
        if output["repair_out"] is not None:
          repair = True
        
        if output["trans_out"] is not None:
          trans = True

        repair_batch_stats = Statistics()
        trans_batch_stats = Statistics()
        repair_loss = 0.0
        trans_loss = 0.0
        w = 1.0
        repair_scores = [1.0]
        trans_scores = [1.0]
       
        non_pad_mask = batch.tgt[1:] != self.criterion.ignore_index # [seq_len, batch]
        non_pad_idx = non_pad_mask.float().contiguous().view(-1).nonzero().squeeze(1) # [bs, 1]
        
        if start_adaptive_training:
          # [bs, tgt_len]
          repair_o = self._bottle(output["repair_out"])[non_pad_idx]
          trans_o =  self._bottle(output["trans_out"])[non_pad_idx]
          w = F.cosine_similarity(repair_o, trans_o) #[bs * len]
          w = self.base_b + self.add_s * w
        
        loss_reduce = not start_adaptive_training
        
        if repair:
          repair_loss, repair_stats, repair_scores = self._compute_loss(batch, output["repair_out"], batch.tgt[1:], select_mask=non_pad_idx, loss_reduce=loss_reduce)
          repair_loss = repair_loss if loss_reduce else repair_loss.sum(dim=1)
          repair_loss = (repair_loss * w).sum()
          repair_loss = repair_loss.div(float(normalization))
          final_loss = final_loss + repair_loss
          
        
        if trans:
          trans_loss, trans_stats, trans_scores = self._compute_loss(batch, output["trans_out"], batch.tgt[1:], select_mask=non_pad_idx, loss_reduce=loss_reduce)
          
          trans_loss = trans_loss if loss_reduce else trans_loss.sum(dim=1)
          trans_loss = (trans_loss * w).sum()
          trans_loss = trans_loss.div(float(normalization))
          final_loss = final_loss + trans_loss
          

        if self.cross_task_reg:
          t2r_kl = F.kl_div(trans_scores, repair_scores, reduction='sum', log_target=True)
          r2t_kl = F.kl_div(repair_scores, trans_scores, reduction='sum', log_target=True)
          t2r_kl = t2r_kl.div(float(normalization))
          trans_stats.add_enc_loss(t2r_kl.clone().item())
          r2t_kl = r2t_kl.div(float(normalization))
          repair_stats.add_enc_loss(r2t_kl.clone().item())
          final_loss = final_loss + (t2r_kl + r2t_kl) / 2
        
        # use auto-mixed precision
        if scaler is not None:
          scaler.scale(final_loss).backward()
        else:
          final_loss.backward()
        
        if trans:
          trans_batch_stats.update(trans_stats)
        if repair:
          repair_batch_stats.update(repair_stats)
        
        return repair_batch_stats, trans_batch_stats

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

  def forward(self, output, target, loss_reduce=True):
    """
    output (FloatTensor): batch_size x n_classes
    target (LongTensor): batch_size or (sent_num, seq_len)
    """
    # if cls_target:
    #   model_prob = self.one_hot.repeat(target.size(1), 1) # [sent_num, n_cla]
    #   model_prob.scatter_(1, target.transpose(0, 1), self.confidence) # [sent_num, n_cla]
    #   model_prob[:, self.ignore_index] = 0
    #   mask = ((target.transpose(0, 1) != self.ignore_index).sum(1, keepdim=True)) == 0 # [sent_num, 1]
    #   model_prob.masked_fill_(mask, 0)
    # else:
    reduction = "none"
    model_prob = self.one_hot.repeat(target.size(0), 1) # [bs, n_cla]
    model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
    model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
   
    if loss_reduce:
      reduction = "sum"
    
    return F.kl_div(output, model_prob, reduction=reduction)


class NMTLossCompute(LossComputeBase):
  """
  Standard NMT Loss Computation.
  """

  def __init__(self, criterion, generator, src_criterion, src_generator, enc_loss_w, use_z_contronl, normalization="sents"):
    super(NMTLossCompute, self).__init__(criterion, generator, src_criterion, src_generator, enc_loss_w, use_z_contronl)

  def _make_shard_state(self, batch, output, range_, attns=None):
    return {
        "output": output,
        "target": batch.tgt[range_[0] + 1: range_[1]],
    }

  def _compute_loss(self, batch, output, target, select_mask=None, loss_reduce=True):
    bottled_output = self._bottle(output)
    gtruth = target.contiguous().view(-1)
    if select_mask is not None: # [non-pading token_num]
      bottled_output = bottled_output[select_mask]
      gtruth = gtruth[select_mask]
      assert(gtruth.size(0) == bottled_output.size(0))
    scores = self.generator(bottled_output)
    loss = self.criterion(scores, gtruth, loss_reduce=loss_reduce)
    report_loss = loss.clone() if loss_reduce else loss.clone().sum()
    stats = self._stats(report_loss, scores, gtruth)
    return loss, stats, scores

  def _compute_cls_loss(self, batch, cls_hid, label, gen_type="tgt", select_mask=None):
    
    (cls_hid, label) = (cls_hid[select_mask], label[:, select_mask]) if select_mask is not None else (cls_hid, label)
    
    if gen_type == 'tgt':
      scores = self.generator(cls_hid)
      loss = self.criterion(scores, label, cls_target=True)
    if gen_type == 'src':
      scores = self.src_generator(cls_hid)
      loss = self.src_criterion(scores, label, cls_target=True)
    
    return loss
    
    

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


def shards(state, shard_size, eval_only=False):
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
        torch.autograd.backward(inputs, grads)
