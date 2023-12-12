"""Base class for encoders and generic multi encoders."""
from pydoc import doc
import torch
import torch.nn as nn
import onmt
from utils.misc import aeq
from onmt.sublayer import PositionwiseFeedForward
from builtins import input


class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout, model_opt=None, layer_idx=None):
    super(TransformerEncoderLayer, self).__init__()
    self.use_ord_ctx = model_opt.use_ord_ctx
    self.use_auto_trans = model_opt.use_auto_trans
    self.cross_attn = model_opt.cross_attn
    self.doc_ctx_start = (model_opt.enc_layers - model_opt.doc_context_layers) <= layer_idx
    # self.cross_before = model_opt.cross_before
    self.share_enc_cross_attn = model_opt.share_enc_cross_attn
    # self-attention + layerNorm
    self.self_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
    self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    if self.use_ord_ctx and self.doc_ctx_start:
      # common context attention
      self.doc_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
      self.doc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    if self.cross_attn and self.doc_ctx_start: # and self.share_enc_cross_attn:
      self.cross_lang_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
      self.cross_lang_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
      if not self.share_enc_cross_attn:
        self.auto_cross_lang_attn = onmt.sublayer.MultiHeadedAttention(
        heads, d_model, dropout=dropout)
        self.auto_cross_lang_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    #   self.auto_src_attn = onmt.sublayer.MultiHeadedAttention(
    #   heads, d_model, dropout=dropout)
    #   self.auto_src_attn_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # feed-forward network + layerNorm
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    # appleid dropout for layers
    self.dropout = nn.Dropout(dropout)
  
  

  def forward(self, inputs=None, mask=None, doc_num=None, trans_inputs=None, trans_mask=None):
    
    def do_self_attn(in_p, m):
      in_p_norm = self.att_layer_norm(in_p)
      out_p, _ = self.self_attn(in_p_norm, in_p_norm, in_p_norm,
                                  mask=m)
      in_p = self.dropout(out_p) + in_p
      return in_p
    
    def do_ctx_attn(in_p, m):
      sent_hidden = in_p[:, 0, :].view(doc_num, -1, in_p.size(-1))
      # [doc_num, 1, sent_num]
      sent_mask = m[:, :, 0].view(doc_num, -1).unsqueeze(1)
      sent_hidden_norm = self.doc_att_layer_norm(sent_hidden)
      # [doc_num, sent_num, hidden]
      sent_output, _ = self.doc_attn(sent_hidden_norm, sent_hidden_norm, sent_hidden_norm,
                                   mask=sent_mask)
      # [doc_num * sent_num, 1, hidden]
      sent_output = sent_output.view(in_p.size(0), -1).unsqueeze(1)
      in_p = self.dropout(sent_output) + in_p
      return in_p

    def do_cross_attn(val, query, v_mask, attn_type="auto2src"):
      if attn_type in ["share", "src2auto"]:
        val_norm = self.cross_lang_attn_layer_norm(val)
        query_norm = self.cross_lang_attn_layer_norm(query)
        val_out, _ = self.cross_lang_attn(val_norm, val_norm, query_norm, mask=v_mask)
        
      # if attn_type == "src2auto":
      #   val_norm = self.src_auto_attn_norm(val)
      #   query_norm = self.src_auto_attn_norm(query)
      #   val_out, _ = self.src_auto_attn(val_norm, val_norm, query_norm, mask=v_mask)
      
      if attn_type == "auto2src":
        val_norm = self.auto_cross_lang_attn_layer_norm(val)
        query_norm = self.auto_cross_lang_attn_layer_norm(query)
        val_out, _ = self.auto_cross_lang_attn(val_norm, val_norm, query_norm, mask=v_mask)

      final_out = self.dropout(val_out) + query
      return final_out

    def do_ffnn(in_p):
      in_p_norm = self.att_layer_norm(in_p)
      out_p = self.feed_forward(in_p_norm)
      # [doc_num*sent_num , seq_len, hidden]
      in_p = self.dropout(out_p) + in_p
      return in_p
    
    if inputs is not None:
      inputs = do_self_attn(inputs, mask)
      if self.use_ord_ctx and self.doc_ctx_start:
        inputs = do_ctx_attn(inputs, mask)
    
    if trans_inputs is not None:
      trans_inputs = do_self_attn(trans_inputs, trans_mask)
      if self.use_ord_ctx and self.doc_ctx_start:
        trans_inputs = do_ctx_attn(trans_inputs, trans_mask)
    
    if self.doc_ctx_start and self.cross_attn:
      assert(inputs is not None and trans_inputs is not None)
      
      trans_attn_type, src_attn_type = "auto2src", "src2auto" if self.share_enc_cross_attn \
                                                                  else "share", "share"
      trans_inputs = do_cross_attn(inputs, trans_inputs, mask, attn_type=trans_attn_type)
      inputs = do_cross_attn(trans_inputs, inputs, trans_mask, attn_type=trans_attn_type)

    
    if inputs is not None:
      inputs = do_ffnn(inputs)

    if trans_inputs is not None:
      trans_inputs = do_ffnn(trans_inputs)
    
    return inputs, trans_inputs

class TransformerEncoder(nn.Module):

  def __init__(self, num_layers, d_model, heads, d_ff,
               dropout, embeddings, model_opt=None):
    super(TransformerEncoder, self).__init__()
    
    # self.paired_trans = paired_trans
    self.num_layers = num_layers
    self.embeddings = embeddings
    
    self.transformer = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout, model_opt=model_opt, layer_idx=i)
       for i in range(num_layers)])

    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def _check_args(self, src, lengths=None):
    _, n_batch = src.size()
    if lengths is not None:
      n_batch_, = lengths.size()
      aeq(n_batch, n_batch_)

  def forward(self, src=None, src_length=None, trans_emb=None, trans_mask=None):
    """ See :obj:`EncoderBase.forward()`"""
    # src: (src_seq_len, batch_size)
    if src is not None:
      self._check_args(src)
      padding_idx = self.embeddings.word_padding_idx
      emb = self.embeddings(src, sent_num=src_length.size(-1))
      out = emb.transpose(0, 1).contiguous()
      words = src.transpose(0, 1)
      mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
    else:
      emb, out, mask = None, None, None
    
    trans_out = trans_emb
    # Run the forward pass of every layer of the transformer.
    for i in range(self.num_layers):
      out, trans_out = self.transformer[i](inputs=out, mask=mask, \
                                     doc_num=src_length.size(0), \
                                     trans_inputs=trans_out, trans_mask=trans_mask)
    if out is not None:
      out = self.layer_norm(out) # [doc_num * sent_num, seq_len, hidden]
      out = out.transpose(0, 1).contiguous()
    
    if trans_out is not None:
      trans_out = self.layer_norm(trans_out)
      trans_out = trans_out.transpose(0, 1).contiguous()
      
    
    return emb, out, mask, trans_out, trans_mask




