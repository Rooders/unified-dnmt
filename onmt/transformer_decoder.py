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
  def __init__(self, d_model, heads, d_ff, dropout, model_opt=None,layer_idx=None):
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



    
    # self.use_auto_trans = model_opt.use_auto_trans
    # self.decoder_cross_before = model_opt.decoder_cross_before
    # self.only_fixed = model_opt.only_fixed
    # self.gated_auto_src = model_opt.gated_auto_src
    # self.share_dec_cross_attn = model_opt.share_dec_cross_attn
    # self.self_attn = onmt.sublayer.MultiHeadedAttention(
    #   heads, d_model, dropout=dropout)
    # if self.share_dec_cross_attn:
    #   self.context_attn = onmt.sublayer.MultiHeadedAttention(
    #     heads, d_model, dropout=dropout)
    #   self.enc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # if not self.share_dec_cross_attn:
    #   self.src_context_attn = onmt.sublayer.MultiHeadedAttention(
    #     heads, d_model, dropout=dropout)
    #   self.src_enc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
      
    #   self.auto_context_attn = onmt.sublayer.MultiHeadedAttention(
    #     heads, d_model, dropout=dropout)
    #   self.auto_enc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # self.self_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # self.dropout = dropout
    # self.drop = nn.Dropout(dropout)

    if self.gated_auto_src:
      self.gate_module = onmt.sublayer.GateController(d_model)
    
    mask = self._get_attn_subsequent_mask(MAX_SIZE)
    # Register self.mask as a buffer in TransformerDecoderLayer, so
    # it gets TransformerDecoderLayer's cuda behavior automatically.
    self.register_buffer('mask', mask)
    # create the modules for a glimpse of next translation.
    # self.tgt_next_attn = tgt_next_attn
    # if self.tgt_next_attn > 0:
    #   self.next_sent_attn = onmt.sublayer.MultiHeadedAttention(
    #   heads, d_model, dropout=dropout)
    #   self.next_sent_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
      
    
  def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
              layer_cache=None, step=None, beam_size=None, auto_trans_bank=None, auto_trans_mask=None, mlm_decoder=False):
    
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
      if attn_type in ["share", "g2src"]:
        q_norm = self.enc_att_layer_norm(q)
        out, attn = self.context_attn(v, v,  q_norm,
                                    mask=m,
                                    layer_cache=layer_cache,
                                    type=inner_type)
      # if attn_type == "g2src":
      #   q_norm = self.src_enc_att_layer_norm(q)
      #   out, attn = self.src_context_attn(v, v,  q_norm,
      #                               mask=m,
      #                               layer_cache=layer_cache,
      #                               type=inner_type)
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
      if mlm_decoder:
        dec_mask = tgt_pad_mask
      else:
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(-1),
                                      :tgt_pad_mask.size(-1)], 0)
    if self.share_dec_cross_attn:
      auto_cross = src_cross = "share"
    else:
      auto_cross = "g2auto"
      src_cross = "g2src"

    # do self attention
    query, attn = do_masked_self_attn(inputs, dec_mask)
    # do cross attention when both gate mechanism and auto translation are available
    if self.doc_ctx_start:
      assert(self.only_fixed == 0 or auto_trans_bank is not None or self.use_auto_trans == 1)
      
      if self.gated_auto_src:
        src_cross_o, attn = do_cross_attn(memory_bank, query, src_pad_mask, attn_type=src_cross, return_resnet=False)
        auto_cross_o, attn = do_cross_attn(auto_trans_bank, query, auto_trans_mask, attn_type=auto_cross,inner_type="auto_context",return_resnet=False)
        mid, z = self.gate_module(src_cross_o, auto_cross_o, return_gate_num=True)
      else:
        mid, attn = do_cross_attn(auto_trans_bank, query, auto_trans_mask, attn_type=auto_cross)
        mid, attn = do_cross_attn(memory_bank, mid, src_pad_mask, attn_type=src_cross)
    
    if not self.doc_ctx_start:
      mid, attn = do_cross_attn(memory_bank, query, src_pad_mask, attn_type=src_cross)
    
    mid = self.drop(mid) + query
    # do cross attention when gate mechanism is not available but auto translation is
    
    # if not self.gated_auto_src and self.use_auto_trans and auto_trans_bank is not None:
    #   if self.decoder_cross_before:
        
    #   else:
    #     mid, attn = do_cross_attn(memory_bank, query, src_pad_mask, attn_type=src_cross)
    #     mid, attn = do_cross_attn(auto_trans_bank, mid, auto_trans_mask, attn_type=auto_cross)
    
    # do cross attention when only one of src and auto translation is available
    # if not self.use_auto_trans or self.only_fixed or auto_trans_bank is None:
    #   mid, attn = do_cross_attn(memory_bank, query, src_pad_mask)
    
    # do ffnn
    mid_norm = self.ffn_layer_norm(mid)
    output = self.feed_forward(mid_norm)
    output = self.drop(output) + mid
    
    return output, attn, z
      
      

    

    


    # if self.use_auto_trans and auto_trans_bank is not None and self.decoder_cross_before and not self.only_fixed and not self.gated_auto_src:
    #   query = do_cross_attn(query)
    #   query_norm = self.enc_att_layer_norm(query)
    #   auto_trans_mid, attn = self.context_attn(auto_trans_bank, auto_trans_bank, query_norm,
    #                               mask=auto_trans_mask,
    #                               layer_cache=layer_cache,
    #                               type="auto_context")
    
    #   query = self.drop(auto_trans_mid) + query
    #   # do encoding output attention
    #   query_norm = self.enc_att_layer_norm(query)
    #   mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
    #                                 mask=src_pad_mask,
    #                                 layer_cache=layer_cache,
    #                                 type="context")
    #   mid = self.drop(mid) + query
    
    # if self.use_auto_trans and auto_trans_bank is not None and not self.decoder_cross_before and not self.only_fixed and not self.gated_auto_src:
    #   query_norm = self.enc_att_layer_norm(query)
    #   src_mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
    #                                 mask=src_pad_mask,
    #                                 layer_cache=layer_cache,
    #                                 type="context")

    #   query = self.drop(src_mid) + query
      
    #   query_norm = self.enc_att_layer_norm(query)
    #   mid, attn = self.context_attn(auto_trans_bank, auto_trans_bank, query_norm,
    #                               mask=auto_trans_mask,
    #                               layer_cache=layer_cache,
    #                               type="auto_context")
    
    #   mid = self.drop(mid) + query
    
    # if self.use_auto_trans and auto_trans_bank is not None and self.gated_auto_src and not self.only_fixed:
    #   query_norm = self.enc_att_layer_norm(query)
    #   src_mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
    #                                 mask=src_pad_mask,
    #                                 layer_cache=layer_cache,
    #                                 type="context")

    #   # query = self.drop(src_mid) + query
      
    #   # query_norm = self.enc_att_layer_norm(query)
    #   mid, attn = self.context_attn(auto_trans_bank, auto_trans_bank, query_norm,
    #                               mask=auto_trans_mask,
    #                               layer_cache=layer_cache,
    #                               type="auto_context")
    
    #   mid, z = self.gate_module(src_mid, mid, return_gate_num=True)
    #   mid = self.drop(mid) + query


    # if not self.use_auto_trans or self.only_fixed or auto_trans_bank is None:
    #   query_norm = self.enc_att_layer_norm(query)
    #   mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
    #                                 mask=src_pad_mask,
    #                                 layer_cache=layer_cache,
    #                                 type="context")
    #   mid = self.drop(mid) + query

    # # do ffn
    # mid_norm = self.ffn_layer_norm(mid)
    # output = self.feed_forward(mid_norm)
    # output = self.drop(output) + mid

    # return output, attn, z

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

    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def init_state(self, src, src_enc, src_mask=None, segment_embeding=None, auto_trans_bank=None, auto_trans_mask=None):
    """ Init decoder state """
    self.state["src"] = src
    self.state["src_enc"] = src_enc
    self.state["src_mask"] = src_mask
    
    self.state["auto_trans_bank"] = auto_trans_bank
    self.state["auto_trans_mask"] = auto_trans_mask
    
    self.state["segment_emb"] = segment_embeding
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


    if self.state["cache"] is not None:
      _recursive_map(self.state["cache"])
    
  def detach_state(self):
    self.state["src"] = self.state["src"].detach()

  def forward(self, tgt, step=None, sent_num=None, beam_size=None, mlm_decoder=False):
    """
    See :obj:`onmt.modules.RNNDecoderBase.forward()`
    """
    if step == 0:
      self._init_cache(self.num_layers)

    # src = self.state["src"]
    memory_bank = self.state["src_enc"]
    # src_words = src.transpose(0, 1)
    tgt_words = tgt.transpose(0, 1)

    # Initialize return variables.
    attns = {"std": []}

    # Run the forward pass of the TransformerDecoder.

    emb = self.embeddings(tgt, step=step, sent_num=sent_num)
    # add the segment embeding during inference
    if self.state["segment_emb"] is not None:
      emb = self.state["segment_emb"] + emb
    
    assert emb.dim() == 3  # len x batch x embedding_dim

    output = emb.transpose(0, 1).contiguous()
    src_memory_bank = memory_bank.transpose(0, 1).contiguous()

    pad_idx = self.embeddings.word_padding_idx
    src_pad_mask = self.state["src_mask"]  # [B, 1, T_src]
    tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
    
    auto_trans_bank = self.state["auto_trans_bank"]
    
    if auto_trans_bank is not None:
      auto_trans_bank = auto_trans_bank.transpose(0, 1).contiguous()
    auto_trans_mask = self.state["auto_trans_mask"]


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
        step=step, beam_size=beam_size, auto_trans_bank=auto_trans_bank, auto_trans_mask=auto_trans_mask, mlm_decoder=mlm_decoder)
      z = z + z

    z = z / self.num_layers
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
