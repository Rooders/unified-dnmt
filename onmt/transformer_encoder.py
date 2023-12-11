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
  
    
    
    
    # self.use_ord_ctx = model_opt.use_ord_ctx
    # self.use_auto_trans = model_opt.use_auto_trans
    # self.cross_attn = model_opt.cross_attn
    self.cross_before = model_opt.cross_before
    # self.share_enc_cross_attn = model_opt.share_enc_cross_attn
    # # self-attention + layerNorm
    # self.self_attn = onmt.sublayer.MultiHeadedAttention(
    #   heads, d_model, dropout=dropout)
    # self.att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # if self.use_ord_ctx:
    #   # common context attention
    #   self.doc_attn = onmt.sublayer.MultiHeadedAttention(
    #   heads, d_model, dropout=dropout)
    #   self.doc_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # if self.cross_attn and self.share_enc_cross_attn:
    #   self.cross_lang_attn = onmt.sublayer.MultiHeadedAttention(
    #   heads, d_model, dropout=dropout)
    #   self.cross_lang_att_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # if self.cross_attn and not self.share_enc_cross_attn:
    #   self.src_auto_attn = onmt.sublayer.MultiHeadedAttention(
    #   heads, d_model, dropout=dropout)
    #   self.src_auto_attn_norm = nn.LayerNorm(d_model, eps=1e-6)
    #   self.auto_src_attn = onmt.sublayer.MultiHeadedAttention(
    #   heads, d_model, dropout=dropout)
    #   self.auto_src_attn_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    # # feed-forward network + layerNorm
    # self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    # self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    # # appleid dropout for layers
    # self.dropout = nn.Dropout(dropout)
  
  

  def forward(self, inputs, mask, doc_num=None, auto_trans_inputs=None, auto_trans_mask=None):
    
    def do_self_attn(inputs, mask):
      input_norm = self.att_layer_norm(inputs)
      outputs, _ = self.self_attn(input_norm, input_norm, input_norm,
                                  mask=mask)
      inputs = self.dropout(outputs) + inputs
      return inputs
    
    def do_ctx_attn(inputs, mask):
      sent_hidden = inputs[:, 0, :].view(doc_num, -1, inputs.size(-1))
      # [doc_num, 1, sent_num]
      sent_mask = mask[:, :, 0].view(doc_num, -1).unsqueeze(1)
      sent_hidden_norm = self.doc_att_layer_norm(sent_hidden)
      # [doc_num, sent_num, hidden]
      sent_output, _ = self.doc_attn(sent_hidden_norm, sent_hidden_norm, sent_hidden_norm,
                                   mask=sent_mask)
      # [doc_num * sent_num, 1, hidden]
      sent_output = sent_output.view(inputs.size(0), -1).unsqueeze(1)
      inputs = self.dropout(sent_output) + inputs
      return inputs

    def do_cross_attn(val, query, v_mask, attn_type="share"):
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

    def do_ffnn(inputs):
      input_norm = self.att_layer_norm(inputs)
      outputs = self.feed_forward(input_norm)
      # [doc_num*sent_num , seq_len, hidden]
      inputs = self.dropout(outputs) + inputs
      return inputs
    

    inputs = do_self_attn(inputs, mask)
    if self.use_ord_ctx and self.doc_ctx_start:
      inputs = do_ctx_attn(inputs, mask)
      if not self.cross_before or auto_trans_inputs is None:
        inputs = do_ffnn(inputs)


    if self.use_auto_trans and auto_trans_inputs is not None:
      auto_trans_inputs = do_self_attn(auto_trans_inputs, auto_trans_mask)
      if self.use_ord_ctx and self.doc_ctx_start:
        auto_trans_inputs = do_ctx_attn(auto_trans_inputs, auto_trans_mask)
        if not self.cross_before:
          auto_trans_inputs = do_ffnn(auto_trans_inputs)
    
    if self.cross_attn and auto_trans_inputs is not None and self.doc_ctx_start:
      if self.share_enc_cross_attn:
        src_attn = auto_attn = "share"
      else:
        src_attn = "src2auto"
        auto_attn = "auto2src"
      auto_trans_inputs = do_cross_attn(inputs, auto_trans_inputs, mask, attn_type=auto_attn)
      inputs = do_cross_attn(auto_trans_inputs, inputs, auto_trans_mask, attn_type=src_attn)
      
    if self.cross_before:
      if auto_trans_inputs is not None:
        auto_trans_inputs = do_ffnn(auto_trans_inputs)
      inputs = do_ffnn(inputs)
      
    
    return inputs, auto_trans_inputs
    

    # if self.use_ord_ctx:
    #   # # [doc_num, sent_num, hidden]
    #   # sent_hidden = inputs[:, 0, :].view(doc_num, -1, inputs.size(-1))
    #   # # [doc_num, 1, sent_num]
    #   # sent_mask = mask[:, :, 0].view(doc_num, -1).unsqueeze(1)
    #   # sent_hidden_norm = self.doc_att_layer_norm(sent_hidden)
    #   # # [doc_num, sent_num, hidden]
    #   # sent_output, _ = self.doc_attn(sent_hidden_norm, sent_hidden_norm, sent_hidden_norm,
    #   #                              mask=sent_mask)
    #   # # [doc_num * sent_num, 1, hidden]
    #   # sent_output = sent_output.view(inputs.size(0), -1).unsqueeze(1)
    #   # inputs = self.dropout(sent_output) + inputs

    # if self.cross_attn and auto_trans_inputs is not None:
    #   input_norm = self.cross_lang_att_layer_norm(inputs)
    #   auto_trans_input_norm = self.cross_lang_att_layer_norm(auto_trans_inputs)
    #   auto_trans_inputs = self.cross_lang_attn(auto_trans_input_norm)
      
    
      
    # input_norm = self.att_layer_norm(inputs)
    
    # outputs = self.feed_forward(input_norm)
    # # [doc_num*sent_num , seq_len, hidden]
    # inputs = self.dropout(outputs) + inputs
    
    return inputs

class TransformerEncoder(nn.Module):

  def __init__(self, num_layers, d_model, heads, d_ff,
               dropout, embeddings, model_opt=None):
    super(TransformerEncoder, self).__init__()
    self.cross_out_encoder = model_opt.cross_out_encoder
    # self.paired_trans = paired_trans
    self.num_layers = num_layers
    self.embeddings = embeddings
    
    self.transformer = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout, model_opt=model_opt, layer_idx=i)
       for i in range(num_layers)])
    
    if self.cross_out_encoder:
      self.cross_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
      self.out_cross_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
      self.out_ffnn = PositionwiseFeedForward(d_model, d_ff, dropout)
      self.ffnn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
      self.dropout = nn.Dropout(dropout)
    
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
  def outer_cross_attn(self, src_bank, auto_trans_bank, src_mask, trans_mask):
    src_bank_norm = self.cross_attn_layer_norm(src_bank)
    auto_trans_bank_norm = self.cross_attn_layer_norm(auto_trans_bank)

    src_bank_out, _ = self.out_cross_attn(auto_trans_bank_norm, auto_trans_bank_norm, src_bank_norm, mask=trans_mask)
    src_bank = src_bank + self.dropout(src_bank_out)
    
    src_bank_norm = self.ffnn_layer_norm(src_bank)
    src_bank_out = self.out_ffnn(src_bank_norm)
    src_bank = self.dropout(src_bank_out) + src_bank

    
    auto_trans_bank_out, _ = self.out_cross_attn(src_bank_norm, src_bank_norm, auto_trans_bank_norm, mask=src_mask)
    auto_trans_bank = auto_trans_bank + self.dropout(auto_trans_bank_out)

    auto_trans_bank_norm = self.ffnn_layer_norm(auto_trans_bank)
    auto_trans_bank_out = self.out_ffnn(auto_trans_bank_norm)
    auto_trans_bank = self.dropout(auto_trans_bank_out) + auto_trans_bank
    

    return src_bank, auto_trans_bank


  def _check_args(self, src, lengths=None):
    _, n_batch = src.size()
    if lengths is not None:
      n_batch_, = lengths.size()
      aeq(n_batch, n_batch_)
  
  def paired_enc_out(self, out, mask, src_length):
    doc_num, sent_num = src_length.size(0), src_length.size(1)
    seq_len = out.size(1)
    out = out.view(doc_num, sent_num, seq_len, -1)
    first_pairs = torch.cat((out[:, -1, :, :], out[:, 0, :, :]), dim=1) # [doc_num, 1, 2*seq_len, hidden]
    rest_pairs = torch.cat((out[:, :-1, :, :], out[:, 1:, :, :]), dim=2) # [doc_num, sent_num-1, 2*seq_len, hidden]
    paired_out = torch.cat((first_pairs.unsqueeze(1), rest_pairs), dim=1) # [doc_num, sent_num, 2*seq_len, hidden]
    paired_out = paired_out.view(doc_num*sent_num, seq_len*2, -1)
    mask = mask.view(doc_num, sent_num, 1, -1)
    first_double_seq_mask = torch.cat((mask[:, -1, :, :], mask[:, 0, :, :]), dim=-1) # [doc_num, 1, 1, 2*seq_len]
    rest_double_seq_mask = torch.cat((mask[:, :-1, :, :], mask[:, 1:, :, :]), dim=-1) # [doc_num, sent_num-1, 1, 2*seq_len]
    paired_mask = torch.cat((first_double_seq_mask.unsqueeze(1), rest_double_seq_mask), dim=1) # [doc_num, sent_num, 1, 2*seq_len]
    paired_mask = paired_mask.view(doc_num*sent_num, 1, -1)
    return paired_out.transpose(0, 1).contiguous(), paired_mask
  

  def forward(self, src=None, src_length=None, auto_trans_emb=None, auto_trans_mask=None, only_trans_encoding=False):
    """ See :obj:`EncoderBase.forward()`"""
    # src: (src_seq_len, batch_size)
    
    self._check_args(src)
    padding_idx = self.embeddings.word_padding_idx
    emb = self.embeddings(src, sent_num=src_length.size(-1))
    out = emb.transpose(0, 1).contiguous()
    auto_trans_out = auto_trans_emb
    words = src.transpose(0, 1)
    mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
    # Run the forward pass of every layer of the transformer.
    for i in range(self.num_layers):
      if only_trans_encoding:
        out, auto_trans_out = self.transformer[i](auto_trans_emb, auto_trans_mask, src_length.size(0), None, None)
        auto_trans_out = out
      else:
        out, auto_trans_out = self.transformer[i](out, mask, src_length.size(0), auto_trans_out, auto_trans_mask)
    
    if self.cross_out_encoder and not only_trans_encoding:
      out, auto_trans_out = self.outer_cross_attn(out, auto_trans_out, mask, auto_trans_mask)
    
    out = self.layer_norm(out) # [doc_num * sent_num, seq_len, hidden]
    if auto_trans_out is not None:
      auto_trans_out = self.layer_norm(auto_trans_out)
      auto_trans_out = auto_trans_out.transpose(0, 1).contiguous()
    

    # Paired Translation Operation
    # if self.paired_trans:
    #   paired_out, paired_mask = self.paired_enc_out(out, mask, src_length)
    #   return emb, paired_out, paired_mask
    # else:
    return emb, out.transpose(0, 1).contiguous(), mask, auto_trans_out




