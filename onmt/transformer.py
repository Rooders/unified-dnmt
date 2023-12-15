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
from onmt.sublayer import Projector
from onmt.embeddings import Embeddings
from utils.misc import use_gpu
from utils.logging import logger
from inputters.dataset import load_fields_from_vocab
import torch.nn.functional as F




class NMTModel(nn.Module):
  def __init__(self, encoder, decoder, model_opt):
    super(NMTModel, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    # self.paired_trans = model_opt.paired_trans
    self.use_auto_trans = model_opt.use_auto_trans
    self.only_fixed = model_opt.only_fixed
    # self.multi_task_training = model_opt.multi_task_training
    #Options of making auto translation appraoch truth.
    # self.auto_truth_trans_kl = model_opt.auto_truth_trans_kl
    self.cross_attn = model_opt.cross_attn
    # self.src_mlm = model_opt.src_mlm
    # self.weight_trans_kl = model_opt.weight_trans_kl
    # self.use_z_contronl = model_opt.use_z_contronl
    self.distance_fc = model_opt.distance_fc
    self.only_trans = False if self.only_fixed or self.cross_attn else True
    # self.shift_num = model_opt.shift_num
    # self.fixed_trans = model_opt.fixed_trans
    if model_opt.use_affine:
      self.transfer_layer = Projector(model_opt.enc_rnn_size, model_opt.dropout)
    else:
      def return_s(input):
        return input
      self.transfer_layer = return_s
    
  def get_embeding_and_mask_before_encoding(self, embeddings_layer, seq, src_length):
    
    padding_idx = embeddings_layer.word_padding_idx
    emb = embeddings_layer(seq, sent_num=src_length.size(-1))
    emb = emb.transpose(0, 1).contiguous()
    words = seq.transpose(0, 1)
    mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]

    return mask, emb

  def avg_pooling_with_mask(self, ori_input, mask):
    # ori_input: [doc_num * sent_num, seq_len, hidden_size]
    # mask: [doc_num * sent_num, 1, seq_len]
    division = (1.0 - mask.float()).sum(dim=-1) # [doc_num * sent_num, 1]
    sum_input = (ori_input * (mask.squeeze(1).unsqueeze(-1).float())).sum(dim=1) # [doc_num * sent_num, hidden_size]
    loss_mask = division == 0
    avg_output = sum_input / division.masked_fill_(loss_mask, torch.finfo(torch.float).max) #[doc_num * sent_num, hidden_size]
    return avg_output, loss_mask


  def encoder_forward(self, task_type="trans", **args):
    if task_type == "unified_enc":
      tgt_tran_mask, tgt_tran_emb = self.get_embeding_and_mask_before_encoding(
                                         self.decoder.embeddings, args['tgt_tran'], args['lengths'])
      return self.encoder(args['src'], args['lengths'],
                          trans_emb=tgt_tran_emb, trans_mask=tgt_tran_mask)
  
    if task_type == "src_enc":
      return self.encoder(args['src'], args['lengths'])
    
    if task_type == "tgt_enc":
      tgt_tran_mask, tgt_tran_emb = self.get_embeding_and_mask_before_encoding(
                                         self.decoder.embeddings, args['tgt_tran'], args['lengths'])
      return self.encoder(src_length=args['lengths'], trans_emb=tgt_tran_emb, trans_mask=tgt_tran_mask)


  def auto_truth_dis_loss(self, v1, v2, distance_fc="euclidean"):
    
    if distance_fc == "euclidean":
      app_dis_loss = F.pairwise_distance(self.transfer_layer(v1), self.transfer_layer(v2), keepdim=True) # [doc_num * sent_num, 1]
    
    if distance_fc == "cosine":
      app_dis_loss = 1.0 - F.cosine_similarity(self.transfer_layer(v1), self.transfer_layer(v2)).unsqueeze(-1)
    
    if distance_fc == "dot_prod":
      app_dis_loss = 1.0 - F.sigmoid(torch.mul(self.transfer_layer(v2)), self.transfer_layer(v2)).sum(dim=-1, keepdim=True)
    
    return app_dis_loss

  def multitask_pass(self, src, tgt, tgt_tran, src_lengths):
    current_batch = random.choice([0, 1, 2])
    if current_batch == 0: # only fixed training
      tgt_tran_mask, tgt_tran_emb = self.get_embeding_and_mask_before_encoding(self.decoder.embeddings, tgt_tran, src_lengths)
      _, _, _, auto_trans_out, _ = self.encoder(src_length=src_lengths, trans_emb=tgt_tran_emb, trans_mask=tgt_tran_mask)
      self.decoder.init_state(auto_trans_bank=auto_trans_out, auto_trans_mask=tgt_tran_mask)
      dec_out, attns, z = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))
      return dec_out, attns, None, None, z

    if current_batch == 1: # only context-aware training
      _, memory_bank, enc_mask, _ = self.encoder(src, src_length=src_lengths)
      self.decoder.init_state(src, memory_bank, enc_mask)
      dec_out, attns, z = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))
      return dec_out, attns, None, None, z
    
    if current_batch == 2: # both training
      tgt_tran_mask, tgt_tran_emb = self.get_embeding_and_mask_before_encoding(self.decoder.embeddings, tgt_tran, src_lengths)
      _, memory_bank, enc_mask, auto_trans_out, _ = self.encoder(src, src_length=src_lengths, auto_trans_emb=tgt_tran_emb, auto_trans_mask=tgt_tran_mask)
      self.decoder.init_state(src, memory_bank, enc_mask, auto_trans_bank=auto_trans_out, auto_trans_mask=tgt_tran_mask)
      dec_out, attns, z = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))
      return dec_out, attns, None, None, z

  def forward(self, src, tgt, tgt_tran=None, src_lengths=None):
    # tgt = tgt[:-1]  # exclude last target from inputs
    # _, memory_bank, enc_mask = self.encoder(src, src_lengths)
    
    # multi_task training mode
    # if self.multi_task_training:
    #   return self.multitask_pass(src, tgt, tgt_tran, src_lengths)
    

    # app_dis_loss = 0.0
    # auto_cls_hidden = None
    # src_cls_hidden = None
    # encoder pass for making auto translaiton approach the truth translsation. / unified only
    if self.cross_attn:
      # tgt_enc = tgt.clone()
      # tgt_enc[0, :] = src[0, 0]
      # _, _, _, truth_out, truth_mask = self.encoder_forward(task_type="tgt_enc", \
      #                                 lengths=src_lengths, tgt_tran=tgt_enc)
      _, memory_bank, src_mask, trans_out, trans_mask= self.encoder_forward(task_type="unified_enc", \
                                      lengths=src_lengths, src=src, tgt_tran=tgt_tran)
      
      self.decoder.init_state(auto_trans_bank=trans_out, auto_trans_mask=trans_mask)
      repair_dec_out, attns, z = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))
      
      self.decoder.init_state(src, memory_bank, src_mask)
      trans_dec_out, attns, z = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))

      return {"repair_out":repair_dec_out, 
              "trans_out":trans_dec_out,
              "attns": attns}

      # if self.auto_truth_trans_kl:
      #   auto_avg_hidden, _ = self.avg_pooling_with_mask(trans_out.transpose(0, 1), trans_mask)
      #   auto_cls_hidden = self.transfer_layer(auto_avg_hidden) # [doc_, dim]
      #   if self.src_mlm:
      #     src_avg_hidden, _ = self.avg_pooling_with_mask(memory_bank.transpose(0, 1), src_mask)
      #     src_cls_hidden = self.transfer_layer(src_avg_hidden)
      # trans_avg_out, loss_mask = self.avg_pooling_with_mask(trans_out.transpose(0, 1), trans_mask)
      # truth_avg_out, _ = self.avg_pooling_with_mask(truth_out.transpose(0, 1), truth_mask)
      # app_dis_loss = self.auto_truth_dis_loss(trans_avg_out, truth_avg_out.detach(), self.distance_fc)
      # app_dis_loss = (app_dis_loss * (1 - loss_mask.float())).sum() / (1 - loss_mask.float()).sum()
      # self.decoder.init_state(src, memory_bank, src_mask, 
      #                         auto_trans_bank=trans_out, 
      #                         auto_trans_mask=trans_mask,
      #                         src_cls_hidden=src_avg_hidden,
      #                         auto_cls_hidden=auto_avg_hidden)
                              
    # encoder pass for DocRepair
    if self.only_fixed:
      _, _, _, trans_out, trans_mask = self.encoder_forward(task_type="tgt_enc", \
                           lengths=src_lengths, tgt_tran=tgt_tran)
      self.decoder.init_state(auto_trans_bank=trans_out, auto_trans_mask=trans_mask)
      dec_out, attns, z = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))
      return {"repair_out":dec_out,
              "trans_out":None,
              "attns": attns}
    # encoder pass for SentTrans or DocTrans
    
    if self.only_trans:
      _, memory_bank, src_mask, _, _ = self.encoder_forward(task_type="src_enc", \
                           lengths=src_lengths, src=src)
      self.decoder.init_state(src, memory_bank, src_mask)
      dec_out, attns, z = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))
      return {"repair_out":None, 
              "trans_out":dec_out,
              "attns": attns}

    # decoder pass
    # dec_out, attns, z = self.decoder(tgt[:-1], sent_num=src_lengths.size(-1))
    # approching loss scale use value of gating 
    # if self.use_z_contronl:
    #   app_dis_loss = z * app_dis_loss * self.weight_trans_kl 
    # else:
    #   app_dis_loss = app_dis_loss * self.weight_trans_kl

    # return dec_out, attns, src_cls_hidden, auto_cls_hidden, z


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

  # Build NMTModel(= encoder + decoder).
  device = torch.device("cuda" if gpu else "cpu")
  model = NMTModel(encoder, decoder, opt)

  # Build Generator.
  gen_func = nn.LogSoftmax(dim=-1)
  src_gen_func = nn.LogSoftmax(dim=-1)
  generator = nn.Sequential(
    nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].vocab), bias=False),
    gen_func
  )
  if model_opt.src_mlm:
    src_generator = nn.Sequential(
      nn.Linear(model_opt.dec_rnn_size, len(fields["src"].vocab), bias=False),
      src_gen_func
    )
  else:
    src_generator = None 


  # Initiate the model
  if model_opt.param_init != 0.0:
    for p in model.parameters():
      p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    for p in generator.parameters():
      p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if src_generator is not None:
      for p in src_generator.parameters():
        p.data.uniform_(-model_opt.param_init, model_opt.param_init)
  if model_opt.param_init_glorot:
    for p in model.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)
    for p in generator.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)
    if src_generator is not None:
      for p in src_generator.parameters():
        p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        
  # Load the pretrained word vector
  if hasattr(model.encoder, 'embeddings'):
    model.encoder.embeddings.load_pretrained_vectors(
        model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
  if hasattr(model.decoder, 'embeddings'):
    model.decoder.embeddings.load_pretrained_vectors(
        model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
  
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
        if not model_opt.share_enc_cross_attn:
          logger.info("init the params in not shared cross attention in encoder with context attention in decoder")
          model.encoder.transformer[i].src_auto_attn.load_state_dict(cur_ctx_attn, strict=False)
          model.encoder.transformer[i].auto_src_attn_norm.load_state_dict(cur_ctx_norm, strict=False)
          model.encoder.transformer[i].auto_src_attn.load_state_dict(cur_ctx_attn, strict=False)
          model.encoder.transformer[i].auto_src_attn_norm.load_state_dict(cur_ctx_norm, strict=False)
        else:
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
        

    generator.load_state_dict(checkpoint['generator'], strict=False)
    
  if model_opt.share_decoder_embeddings:
    generator[0].weight = model.decoder.embeddings.word_lut.weight
    if model_opt.src_mlm:
      src_generator[0].weight = model.encoder.embeddings.word_lut.weight
  
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
  model.src_generator = src_generator
  model.to(device)

  return model


def build_model(model_opt, opt, fields, checkpoint):
  """ Build the Model """
  logger.info('Building model...')
  model = build_base_model(model_opt, fields,
                           use_gpu(opt), checkpoint, opt=opt, is_train=True)
  logger.info(model)
  return model
