"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import random
import onmt.constants as Constants 

from onmt.transformer_encoder import TransformerEncoder
from onmt.transformer_decoder import TransformerDecoder

from onmt.embeddings import Embeddings
from utils.misc import use_gpu
from utils.logging import logger
from inputters.dataset import load_fields_from_vocab
import torch.nn.functional as F




class NMTModel(nn.Module):
  def __init__(self, encoder, decoder, model_opt, mlm_decoder=None):
    super(NMTModel, self).__init__()
    # modules
    self.encoder = encoder
    self.decoder = decoder
    self.mlm_decoder = mlm_decoder
    # options
    self.use_auto_trans = model_opt.use_auto_trans
    self.only_fixed = model_opt.only_fixed
    self.mlm_prob = model_opt.mlm_prob
    self.sentence_level = model_opt.sentence_level
    self.mlm_distill = model_opt.mlm_distill


  def get_embeding_and_mask_before_encoding(self, embeddings_layer, seq, src_length):
    
    padding_idx = embeddings_layer.word_padding_idx
    emb = embeddings_layer(seq, sent_num=src_length.size(-1))
    emb = emb.transpose(0, 1).contiguous()
    words = seq.transpose(0, 1)
    mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]

    return mask, emb
  
  def mlm_prob_sampling(self):
    prob_l = [i for i in range(15, 60, 1)]
    prob = random.sample(prob_l, 1)[0]
    return prob / 100

  def mask_tokens_with_spec_mask(self, inputs, spec_mask, enc_type="tran"):
    if enc_type == "tran":
      embeddings = self.decoder.embeddings
    elif enc_type == "mlm":
      embeddings = self.mlm_decoder.embeddings
    else:
      embeddings = self.encoder.embeddings
   
    labels = inputs.clone() # [bs, seq_len]
    masked_indices = spec_mask.bool()
    labels[~masked_indices] = embeddings.word_padding_idx  # We only compute loss on masked tokens
    inputs[masked_indices] = embeddings.mask_word_idx
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.transpose(0, 1).contiguous(), labels.transpose(0, 1).contiguous()

  def mask_tokens(self, inputs, mlm_probability=0.15, enc_type="tran"):
    # inputs: [bs, seq_len]
    if enc_type == "tran":
      embeddings = self.decoder.embeddings
    elif enc_type == "mlm":
      embeddings = self.mlm_decoder.embeddings
    else:
      embeddings = self.encoder.embeddings


    def special_mask(in_l, special_l):
      return [1 if val in special_l else 0 for val in in_l]
   
    labels = inputs.clone() # [bs, seq_len]
    device = inputs.device
    probability_matrix = torch.full(labels.shape, mlm_probability)# [bs, seq_len]
    special_tokens_mask = [special_mask(val, embeddings.special_tokens_idxs) for val in labels.tolist()] #[bs, seq_len]

    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0) # [bs, seq_len]
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = embeddings.word_padding_idx  # We only compute loss on masked tokens
    
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = embeddings.mask_word_idx

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(embeddings.word_vocab_size, labels.shape, dtype=torch.long, device=device)
    indices_random.to(device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.transpose(0, 1).contiguous(), labels.transpose(0, 1).contiguous()
  
  def noisy_input(self, inputs, noise_prob=0.15):
    embeddings = self.mlm_decoder.embeddings
    
    def special_mask(in_l, special_l):
      return [1 if val in special_l else 0 for val in in_l]
   
    labels = inputs.clone() # [bs, seq_len]
    device = inputs.device
    probability_matrix = torch.full(labels.shape, noise_prob)# [bs, seq_len]
    special_tokens_mask = [special_mask(val, embeddings.special_tokens_idxs) for val in labels.tolist()] #[bs, seq_len]

    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0) # [bs, seq_len]
    noise_indices = torch.bernoulli(probability_matrix).bool()
    random_words = torch.randint(embeddings.word_vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[noise_indices] = random_words[noise_indices]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs

  def forward(self, src, tgt, tgt_tran=None, src_lengths=None, only_nmt=False):
    # tgt = tgt[:-1]  # exclude last target from inputs
    # _, memory_bank, enc_mask = self.encoder(src, src_lengths)
    
    if self.use_auto_trans:
      tgt_tran_mask, tgt_tran_emb = self.get_embeding_and_mask_before_encoding(self.decoder.embeddings, tgt_tran, src_lengths)
      if self.only_fixed:
        _, memory_bank, enc_mask, auto_trans_out = self.encoder(src, src_length=src_lengths, auto_trans_emb=tgt_tran_emb, auto_trans_mask=tgt_tran_mask, only_trans_encoding=True)
      
      else:
        _, memory_bank, enc_mask, auto_trans_out = self.encoder(src, src_length=src_lengths, auto_trans_emb=tgt_tran_emb, auto_trans_mask=tgt_tran_mask)
      
    if not self.use_auto_trans or self.sentence_level:
      _, memory_bank, enc_mask, _ = self.encoder(src, src_length=src_lengths)
      tgt_tran_mask, auto_trans_out = None, None

    if self.only_fixed:
      self.decoder.init_state(tgt_tran, auto_trans_out, tgt_tran_mask)
    
    if self.use_auto_trans and not self.only_fixed:
      self.decoder.init_state(src, memory_bank, enc_mask, auto_trans_bank=auto_trans_out, auto_trans_mask=tgt_tran_mask)
    
    if self.sentence_level or not self.use_auto_trans:
      self.decoder.init_state(src, memory_bank, enc_mask)
    dec_out, attns, _ = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))
  
    if self.mlm_distill and not only_nmt:
      # tgt_tran_distill_mask, tgt_tran_distill_emb = self.get_embeding_and_mask_before_encoding(self.mlm_decoder.embeddings, tgt_tran, src_lengths)
      # _, distill_memory_bank, distill_enc_mask, noise_tgt_out = self.encoder(src, src_length=src_lengths, auto_trans_emb=tgt_tran_distill_emb, auto_trans_mask=tgt_tran_distill_mask)
      self.mlm_decoder.init_state(src, memory_bank, enc_mask, auto_trans_bank=auto_trans_out, auto_trans_mask=tgt_tran_mask)
      # self.mlm_decoder.init_state(src, distill_memory_bank, distill_enc_mask, auto_trans_bank=noise_tgt_out, auto_trans_mask=tgt_tran_distill_mask)
      mlm_prob = self.mlm_prob_sampling()
      mlm_inputs, mlm_labels = self.mask_tokens(tgt[1:].clone().transpose(0, 1), mlm_probability=mlm_prob, enc_type='mlm')
      mlm_dec_out, attn, _ = self.mlm_decoder(mlm_inputs, sent_num=src_lengths.size(-1), mlm_decoder=True)
    else:
      mlm_dec_out = None
      mlm_labels = None
    
    return dec_out, attns, mlm_dec_out, mlm_labels

  def forward_mlm_for_distillation(self, src, tgt, tgt_tran=None, src_lengths=None, mask_id=None):
    if self.use_auto_trans:
      tgt_tran_mask, tgt_tran_emb = self.get_embeding_and_mask_before_encoding(self.mlm_decoder.embeddings, tgt_tran, src_lengths)
      _, memory_bank, enc_mask, auto_trans_out = self.encoder(src, src_length=src_lengths, auto_trans_emb=tgt_tran_emb, auto_trans_mask=tgt_tran_mask)
      
    if not self.use_auto_trans or self.sentence_level:
      _, memory_bank, enc_mask, _ = self.encoder(src, src_length=src_lengths)
      tgt_tran_mask, auto_trans_out = None, None
    
    mlm_inputs, _ = self.mask_tokens_with_spec_mask(tgt[1:].clone().transpose(0, 1), spec_mask=mask_id.transpose(0, 1), enc_type='mlm')
    self.mlm_decoder.init_state(src, memory_bank, enc_mask, auto_trans_bank=auto_trans_out, auto_trans_mask=tgt_tran_mask)
    mlm_dec_out, _, _ = self.mlm_decoder(mlm_inputs, sent_num=src_lengths.size(-1), mlm_decoder=True)
    return mlm_dec_out


def build_embeddings(opt, word_dict, for_encoder=True):
  """
  Build an Embeddings instance.
  Args:
      opt: the option in current environment.
      word_dict(Vocab): words dictionary.
      feature_dicts([Vocab], optional): a list of feature dictionary.
      for_encoder(bool): build Embeddings for encoder or decoder?
  """
  if for_encoder:
    embedding_dim = opt.src_word_vec_size
  else:
    embedding_dim = opt.tgt_word_vec_size



  word_padding_idx = word_dict.stoi[Constants.PAD_WORD]
  mask_word_idx = word_dict.stoi[Constants.MASK_WORD]
  special_tokens_idxs = [word_dict.stoi[spec_tok] for spec_tok in [Constants.PAD_WORD, Constants.UNK_WORD, \
                                        Constants.SLU_WORD, Constants.SEG_WORD, Constants.BOS_WORD, Constants.EOS_WORD]]  
    
  num_word_embeddings = len(word_dict)

  return Embeddings(word_vec_size=embedding_dim,
                    position_encoding=opt.position_encoding,
                    segment_embedding=opt.segment_embedding,
                    dropout=opt.dropout,
                    word_padding_idx=word_padding_idx,
                    mask_word_idx=mask_word_idx,
                    special_tokens_idxs=special_tokens_idxs,
                    word_vocab_size=num_word_embeddings,
                    sparse=opt.optim == "sparseadam")


def build_encoder(opt, embeddings):
  """
  Various encoder dispatcher function.
  Args:
      opt: the option in current environment.
      embeddings (Embeddings): vocab embeddings for this encoder.
  """
  return TransformerEncoder(opt.enc_layers, opt.enc_rnn_size,
                            opt.heads, opt.transformer_ff,
                            opt.dropout, embeddings, opt)

def build_decoder(opt, embeddings):
  """
  Various decoder dispatcher function.
  Args:
      opt: the option in current environment.
      embeddings (Embeddings): vocab embeddings for this decoder.pingshi 
  """
  return TransformerDecoder(opt.dec_layers, opt.dec_rnn_size,
                     opt.heads, opt.transformer_ff,
                     opt.dropout, embeddings, opt)

def load_test_model(opt, dummy_opt, model_path=None):
  if model_path is None:
    model_path = opt.models[0]
  checkpoint = torch.load(model_path,
                        map_location=lambda storage, loc: storage)
  model_opt = checkpoint['opt']
  fields = load_fields_from_vocab(checkpoint['vocab'], model_opt, opt.tgt_tran)
  for arg in dummy_opt:
    if arg not in model_opt:
      model_opt.__dict__[arg] = dummy_opt[arg]
  model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint, model_opt)
  model.eval()
  model.generator.eval()
  return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, opt=None, is_train=False):
  """
  Args:
      model_opt: the option loaded from checkpoint.
      fields: `Field` objects for the model.
      gpu(bool): whether to use gpu.
      checkpoint: the model gnerated by train phase, or a resumed snapshot
                  model from a stopped training.
  Returns:
      the NMTModel.
  """

  # for backward compatibility
  if model_opt.enc_rnn_size != model_opt.dec_rnn_size:
    raise AssertionError("""We do not support different encoder and
                         decoder rnn sizes for translation now.""")

  # Build encoder.
  src_dict = fields["src"].vocab
  src_embeddings = build_embeddings(model_opt, src_dict)
  encoder = build_encoder(model_opt, src_embeddings)
  
  # Build decoder.
  tgt_dict = fields["tgt"].vocab
  tgt_embeddings = build_embeddings(model_opt, tgt_dict,
                                    for_encoder=False)

  # Share the embedding matrix - preprocess with share_vocab required.
  if model_opt.share_embeddings:
    # src/tgt vocab should be the same if `-share_vocab` is specified.
    if src_dict != tgt_dict:
      raise AssertionError('The `-share_vocab` should be set during '
                           'preprocess if you use share_embeddings!')

    tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

  decoder = build_decoder(model_opt, tgt_embeddings)
  # Build Generator.
  gen_func = nn.LogSoftmax(dim=-1)
  generator = nn.Sequential(
    nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].vocab), bias=False),
    gen_func
  )
  device = torch.device("cuda" if gpu else "cpu")
  # if use distillation, init a mlm model for use
  if model_opt.mlm_distill:
    # whole distill model
    # distill_tgt_embeddings = build_embeddings(model_opt, tgt_dict,
    #                                 for_encoder=False)
    cmlm_decoder = build_decoder(model_opt, tgt_embeddings)
    
    if model_opt.new_gen:
      cmlm_gen_func = nn.LogSoftmax(dim=-1)
      cmlm_generator = nn.Sequential(
        nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].vocab), bias=False),
        cmlm_gen_func)
    else:
      cmlm_generator = generator
    # decoder for joint training 
  else:
    cmlm_generator = None
    cmlm_decoder = None
  # Build NMTModel(= encoder + decoder).
  
  model = NMTModel(encoder, decoder, opt, cmlm_decoder)


  # Initiate the model
  if model_opt.param_init != 0.0:
    for p in model.parameters():
      p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    for p in generator.parameters():
      p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.new_gen and model_opt.mlm_distill:
      for p in cmlm_generator.parameters():
        p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    
  if model_opt.param_init_glorot:
    for p in model.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)
    for p in generator.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)
    if model_opt.new_gen and model_opt.mlm_distill:
      for p in cmlm_generator.parameters():
        if p.dim() > 1:
          xavier_uniform_(p)
        
  # Load the pretrained word vector
  if hasattr(model.encoder, 'embeddings'):
    model.encoder.embeddings.load_pretrained_vectors(
        model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
  if hasattr(model.decoder, 'embeddings'):
    model.decoder.embeddings.load_pretrained_vectors(
        model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
  
  if model_opt.share_decoder_embeddings:
    generator[0].weight = decoder.embeddings.word_lut.weight
  
  if model_opt.share_mlm_decoder_embeddings:
    cmlm_generator[0].weight = cmlm_decoder.embeddings.word_lut.weight
  
  # Load the model states from checkpoint.
  if checkpoint is not None:
    # This preserves backward-compat for models using customed layernorm
    def fix_key(s):
      s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                 r'\1.layer_norm\2.bias', s)
      s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                 r'\1.layer_norm\2.weight', s)
      return s

    checkpoint['model'] = \
      {fix_key(k): v for (k, v) in checkpoint['model'].items()}
    # end of patch for backward compatibility
    
    
    model.load_state_dict(checkpoint['model'], strict=False)
    # load NMT decoder for MLM decoder
  
    
    
    # init all cross attention including in encoder with decoder cross attention.  
    if model_opt.init_cross_sent and is_train:
      logger.info("init all cross attention module with the sentence-level cross attention from decoder ...")
      for i in range(model_opt.enc_layers):
        cur_ctx_attn = {}
        cur_ctx_norm = {}
        cur_self_attn = {}
        cur_self_norm = {}
        for k, v in checkpoint['model'].items():
          # extract the context attention from decoder
          if "."+ str(i) + "." in k and "context" in k and "decoder" in k:
            c_k = k.split("n.")[1]
            cur_ctx_attn[c_k] = v
          if "."+ str(i) + "." in k and "enc_att_layer_norm" in k and "decoder" in k:
            n_k = k.split(".")[-1]
            cur_ctx_norm[n_k] = v
          # extract the self attention from encoder
          if "."+ str(i) + "." in k and "encoder" in k and "self" in k:
            s_k = k.split("n.")[1]
            cur_self_attn[s_k] = v
          if "."+ str(i) + "." in k and ".att_layer_norm" in k:
            n_s_k = k.split(".")[-1]
            cur_self_norm[n_s_k] = v
        
        # init the params in doc attention with self attention in encoder
        if model_opt.use_ord_ctx:
          logger.info("init the params in doc attention with self attention in encoder ... ")
          model.encoder.transformer[i].doc_attn.load_state_dict(cur_self_attn, strict=False)
          model.encoder.transformer[i].doc_att_layer_norm.load_state_dict(cur_self_norm, strict=False)
        
        # init the params in cross attention in encoder with context attention in decoder
        if not model_opt.share_enc_cross_attn and model_opt.cross_attn:
          logger.info("init the params in not shared cross attention in encoder with context attention in decoder")
          model.encoder.transformer[i].src_auto_attn.load_state_dict(cur_ctx_attn, strict=False)
          model.encoder.transformer[i].auto_src_attn_norm.load_state_dict(cur_ctx_norm, strict=False)
          model.encoder.transformer[i].auto_src_attn.load_state_dict(cur_ctx_attn, strict=False)
          model.encoder.transformer[i].auto_src_attn_norm.load_state_dict(cur_ctx_norm, strict=False)
        if model_opt.share_enc_cross_attn and model_opt.cross_attn:
          logger.info("init the params in shared cross attention in encoder with context attention in decoder")
          model.encoder.transformer[i].cross_lang_attn.load_state_dict(cur_ctx_attn, strict=False)
          model.encoder.transformer[i].cross_lang_att_layer_norm.load_state_dict(cur_ctx_norm, strict=False)   
        
        # init the params in cross attention in decoder with context attention in decoder
        if not model_opt.share_dec_cross_attn:
          logger.info("init the params in not shared cross attention in decoder with context attention in decoder")
          model.decoder.transformer_layers[i].src_context_attn.load_state_dict(cur_ctx_attn, strict=False)
          model.decoder.transformer_layers[i].src_enc_att_layer_norm.load_state_dict(cur_ctx_norm, strict=False)
          model.decoder.transformer_layers[i].auto_context_attn.load_state_dict(cur_ctx_attn, strict=False)
          model.decoder.transformer_layers[i].auto_enc_att_layer_norm.load_state_dict(cur_ctx_norm, strict=False)
        
      if model_opt.mlm_distill:
        logger.info("init the mlm decoder with nmt decoder ... ")
        decoder_dict = model.decoder.state_dict()
        model.mlm_decoder.load_state_dict(decoder_dict, strict=False)
      

    generator.load_state_dict(checkpoint['generator'], strict=False)
  # else:
  #   if model_opt.param_init != 0.0:
  #     for p in model.parameters():
  #       p.data.uniform_(-model_opt.param_init, model_opt.param_init)
  #     for p in generator.parameters():
  #       p.data.uniform_(-model_opt.param_init, model_opt.param_init)
  #   if model_opt.param_init_glorot:
  #     for p in model.parameters():
  #       if p.dim() > 1:
  #         xavier_uniform_(p)
  #     for p in generator.parameters():
  #       if p.dim() > 1:
  #         xavier_uniform_(p)

  #   if hasattr(model.encoder, 'embeddings'):
  #     model.encoder.embeddings.load_pretrained_vectors(
  #         model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
  #   if hasattr(model.decoder, 'embeddings'):
  #     model.decoder.embeddings.load_pretrained_vectors(
  #         model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
  #pdb.set_trace()
  # Add generator to model (this registers it as parameter of model).
  model.generator = generator
  model.mlm_generator = cmlm_generator
  model.to(device)

  return model


def build_model(model_opt, opt, fields, checkpoint):
  """ Build the Model """
  logger.info('Building model...')
  model = build_base_model(model_opt, fields,
                           use_gpu(opt), checkpoint, opt=opt, is_train=True)
  logger.info(model)
  return model
