"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

import onmt
from onmt.sublayer import PositionwiseFeedForward

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout, model_opt=None, layer_idx=None):
    super(TransformerDecoderLayer, self).__init__()
    self.use_auto_trans = model_opt.use_auto_trans
    self.decoder_cross_before = model_opt.decoder_cross_before
    self.only_fixed = model_opt.only_fixed
    self.gated_auto_src = model_opt.gated_auto_src
    self.share_dec_cross_attn = model_opt.share_dec_cross_attn
    self.doc_ctx_start = (model_opt.dec_layers - model_opt.doc_context_layers) <= layer_idx
    
    self.self_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
    
    self.context_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
    self.enc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
  
    if not self.share_dec_cross_attn and self.doc_ctx_start:
      self.auto_context_attn = onmt.sublayer.MultiHeadedAttention(
        heads, d_model, dropout=dropout)
      self.auto_enc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    self.self_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    self.dropout = dropout
    self.drop = nn.Dropout(dropout)

    if self.gated_auto_src and self.doc_ctx_start:
      self.gate_module = onmt.sublayer.GateController(d_model)
    
    mask = self._get_attn_subsequent_mask(MAX_SIZE)
    # Register self.mask as a buffer in TransformerDecoderLayer, so
    # it gets TransformerDecoderLayer's cuda behavior automatically.
    self.register_buffer('mask', mask)
    
      
    
  def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
              layer_cache=None, step=None, beam_size=None, 
              auto_trans_bank=None, auto_trans_mask=None,
              src_cls_hidden=None, auto_cls_hidden=None,
              src_cls_mask=None, auto_cls_mask=None):
    
    # F of self attention
    def do_masked_self_attn(v, v_mask):
      v_norm = self.self_att_layer_norm(v)

      v_query, attn = self.self_attn(v_norm, v_norm, v_norm,
                                  mask=v_mask,
                                  layer_cache=layer_cache,
                                  type="self")
      self_attn_out = self.drop(v_query) + v # [doc_num*sent_num, 2*seq_len, hidden]
      return self_attn_out, attn
    # F of cross attention
    def do_cross_attn(v, q, m, attn_type="share", inner_type="context", return_resnet=True):
      if attn_type == "share":
        q_norm = self.enc_att_layer_norm(q)
        out, attn = self.context_attn(v, v,  q_norm,
                                    mask=m,
                                    layer_cache=layer_cache,
                                    type=inner_type)
      if attn_type == "g2src":
        q_norm = self.enc_att_layer_norm(q)
        out, attn = self.context_attn(v, v,  q_norm,
                                    mask=m,
                                    layer_cache=layer_cache,
                                    type=inner_type)
      if attn_type == "g2auto":
        q_norm = self.auto_enc_att_layer_norm(q)
        out, attn = self.auto_context_attn(v, v,  q_norm,
                                    mask=m,
                                    layer_cache=layer_cache,
                                    type=inner_type)
      if return_resnet:
        cross_attn_out = self.drop(out) + q
      else:
        cross_attn_out = out
      return cross_attn_out, attn
    
    dec_mask = None
    z = 0.0
    
    if step is None:
      if src_cls_hidden is not None:
        tgt_pad_mask = torch.cat((src_cls_mask, tgt_pad_mask), 2)
        inputs = torch.cat((src_cls_hidden.unsqueeze(1), inputs), 1)
      if auto_cls_hidden is not None:
        tgt_pad_mask = torch.cat((auto_cls_mask, tgt_pad_mask), 2) # [sent_num, 1, seq_len + 2]
        inputs = torch.cat((auto_cls_hidden.unsqueeze(1), inputs), 1)
      
      dec_mask = torch.gt(tgt_pad_mask +
                          self.mask[:, :tgt_pad_mask.size(-1),
                                    :tgt_pad_mask.size(-1)], 0) # [sent_num, 1, seq_len]
    if step == 0:
      if src_cls_hidden is not None:
        inputs = torch.cat((src_cls_hidden.unsqueeze(1), inputs), 1)
      if auto_cls_hidden is not None:
        inputs = torch.cat((auto_cls_hidden.unsqueeze(1), inputs), 1)
      
      
    if self.share_dec_cross_attn:
      auto_cross = src_cross = "share"
    else:
      auto_cross = "g2auto"
      src_cross = "g2src"

    # do self attention
    query, attn = do_masked_self_attn(inputs, dec_mask)

    # do cross attention when both gate mechanism and auto translation are available
    if memory_bank is not None and auto_trans_bank is None:
      mid, attn = do_cross_attn(memory_bank, query, src_pad_mask, attn_type=src_cross)
    if memory_bank is None and auto_trans_bank is not None:
      mid, attn = do_cross_attn(auto_trans_bank, query, auto_trans_mask, attn_type=auto_cross,inner_type="auto_context")
    
    if self.gated_auto_src:
      assert(self.only_fixed == 0 or auto_trans_bank is not None or self.use_auto_trans == 1)
      if self.doc_ctx_start:
        src_cross_o, attn = do_cross_attn(memory_bank, query, src_pad_mask, attn_type=src_cross, return_resnet=False)
        auto_cross_o, attn = do_cross_attn(auto_trans_bank, query, auto_trans_mask, attn_type=auto_cross,inner_type="auto_context",return_resnet=False)
        mid, z = self.gate_module(auto_cross_o, src_cross_o, return_gate_num=True)
        mid = self.drop(mid) + query
      if not self.doc_ctx_start:
        mid, attn = do_cross_attn(memory_bank, query, src_pad_mask, attn_type=src_cross)
    # do ffnn
    mid_norm = self.ffn_layer_norm(mid)
    output = self.feed_forward(mid_norm)
    output = self.drop(output) + mid
    
    if step is None or step == 0:
      if auto_cls_hidden is not None:
        output = output[:, 1:, :]
      if src_cls_hidden is not None:
        output = output[:, 1:, :]

    return output, attn, z

  def _get_attn_subsequent_mask(self, size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class TransformerDecoder(nn.Module):
  def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, model_opt):
    super(TransformerDecoder, self).__init__()
    # self.use_auto_trans = model_opt.use_auto_trans
    # self.decoder_cross_before = model_opt.decoder_cross_before
    # self.only_fixed = model_opt.only_fixed
    # self.gated_auto_src = model_opt.gated_auto_src
    # Basic attributes.
    self.decoder_type = 'transformer'
    self.num_layers = num_layers
    self.embeddings = embeddings
    # self.tgt_next_attn = tgt_next_attn
    # Decoder State
    self.state = {}
    
    # Build TransformerDecoder.
    self.transformer_layers = nn.ModuleList(
      [TransformerDecoderLayer(d_model, heads, d_ff, dropout, model_opt=model_opt, layer_idx=i)
       for i in range(num_layers)])
    self.gate_avg_num = model_opt.doc_context_layers
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def init_state(self, src=None, src_enc=None, src_mask=None, 
                segment_embeding=None, 
                auto_trans_bank=None, auto_trans_mask=None, 
                src_cls_hidden=None, auto_cls_hidden=None):
    """ Init decoder state """
    self.state["src"] = src
    self.state["src_enc"] = src_enc
    self.state["src_mask"] = src_mask
    
    self.state["auto_trans_bank"] = auto_trans_bank
    self.state["auto_trans_mask"] = auto_trans_mask
    
    self.state["segment_emb"] = segment_embeding
    self.state["src_cls_hidden"] = src_cls_hidden
    self.state["auto_cls_hidden"] = auto_cls_hidden
    self.state["cache"] = None

  def map_state(self, fn):
    def _recursive_map(struct, batch_dim=0):
      for k, v in struct.items():
        if v is not None:
          if isinstance(v, dict):
            _recursive_map(v)
          else:
            struct[k] = fn(v, batch_dim)

    self.state["src"] = fn(self.state["src"], 1)
    self.state["src_enc"] = fn(self.state["src_enc"], 1)
    
    if self.state["src_mask"] is not None:
      self.state["src_mask"] = fn(self.state["src_mask"], 0)
    
    if self.state["segment_emb"] is not None:
      self.state["segment_emb"] = fn(self.state["segment_emb"], 1)
    
    if self.state["auto_trans_mask"] is not None:
      self.state["auto_trans_mask"] = fn(self.state["auto_trans_mask"], 0)

    if self.state["auto_trans_bank"] is not None:
      self.state["auto_trans_bank"] = fn(self.state["auto_trans_bank"], 1)

    if self.state["src_cls_hidden"] is not None:
      self.state["src_cls_hidden"] = fn(self.state["src_cls_hidden"], 0)
    
    if self.state["auto_cls_hidden"] is not None:
      self.state["auto_cls_hidden"] = fn(self.state["auto_cls_hidden"], 0)
    
    # if self.state["src_cls_mask"] is not None:
    #   self.state["src_cls_mask"] = fn(self.state["src_cls_mask"], 0)
    
    # if self.state["auto_cls_mask"] is not None:
    #   self.state["auto_cls_mask"] = fn(self.state["auto_cls_mask"], 0)

      
    
    if self.state["cache"] is not None:
      _recursive_map(self.state["cache"])
    
  def detach_state(self):
    self.state["src"] = self.state["src"].detach()

  def forward(self, tgt, step=None, sent_num=None, beam_size=None):
    """
    See :obj:`onmt.modules.RNNDecoderBase.forward()`
    """
    # Target Inputs
    if step == 0:
      self._init_cache(self.num_layers)
    attns = {"std": []}
    tgt_words = tgt.transpose(0, 1)
    emb = self.embeddings(tgt, step=step, sent_num=sent_num)
    if self.state["segment_emb"] is not None:
      emb = self.state["segment_emb"] + emb
    assert emb.dim() == 3  # len x batch x embedding_dim
    
    # Output of Encoder
    src_memory_bank = self.state["src_enc"].transpose(0, 1).contiguous() \
                  if self.state["src_enc"] is not None else None 
    auto_trans_bank = self.state["auto_trans_bank"].transpose(0, 1).contiguous() \
                      if self.state["auto_trans_bank"] is not None else None 
    src_cls_hidden = self.state["src_cls_hidden"] \
                      if self.state["src_cls_hidden"] is not None else None 
    
    auto_cls_hidden = self.state["auto_cls_hidden"] \
                      if self.state["auto_cls_hidden"] is not None else None 
    # print(auto_cls_hidden.shape)
    # print(src_memory_bank.shape) 
    src_pad_mask = self.state["src_mask"]  # [B, 1, T_src]
    auto_trans_mask = self.state["auto_trans_mask"]
    # src_cls_mask = self.state["src_cls_mask"]  # [B, 1, T_src]
    # auto_cls_mask = self.state["auto_cls_mask"]
    # Input of Decoder
    output = emb.transpose(0, 1).contiguous()
    pad_idx = self.embeddings.word_padding_idx
    tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
    src_cls_mask = auto_cls_mask = src_pad_mask.logical_not().sum(2, keepdim=True) == 0
    z = 0.0    
    for i in range(self.num_layers):
      output, attn, z = self.transformer_layers[i](
        output,
        src_memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=(
          self.state["cache"]["layer_{}".format(i)]
          if step is not None else None),
        step=step, beam_size=beam_size,
        auto_trans_bank=auto_trans_bank, auto_trans_mask=auto_trans_mask,
        src_cls_hidden=src_cls_hidden, auto_cls_hidden=auto_cls_hidden,
        src_cls_mask=src_cls_mask, auto_cls_mask=auto_cls_mask)
      z = z + z

    z = z / self.gate_avg_num
    output = self.layer_norm(output)
    # Process the result and update the attentions.
    dec_outs = output.transpose(0, 1).contiguous()
    attn = attn.transpose(0, 1).contiguous()

    attns["std"] = attn

    # TODO change the way attns is returned dict => list or tuple (onnx)
    return dec_outs, attns, z

  def _init_cache(self, num_layers):
    self.state["cache"] = {}

    for l in range(num_layers):
      layer_cache = {
        "memory_keys": None,
        "memory_values": None
      }
      layer_cache["self_keys"] = None
      layer_cache["self_values"] = None
      layer_cache["auto_memory_keys"] = None
      layer_cache["auto_memory_values"] = None
      self.state["cache"]["layer_{}".format(l)] = layer_cache
