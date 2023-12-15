#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import configargparse
import onmt.opts as opts
import torch
import onmt.transformer as nmt_model
from inputters.dataset import build_dataset, OrderedIterator, make_features
from onmt.beam import Beam
from utils.misc import tile
import onmt.constants as Constants 
import time
from tkinter import _flatten

def build_translator(opt):
  dummy_parser = configargparse.ArgumentParser(description='translate.py')
  opts.model_opts(dummy_parser)
  dummy_opt = dummy_parser.parse_known_args([])[0]

  fields, model, model_opt = nmt_model.load_test_model(opt, dummy_opt.__dict__)
  
  translator = Translator(model, fields, opt, model_opt)

  return translator

class Translator(object):
  def __init__(self, model, fields, opt, model_opt, out_file=None):
    self.model = model
    self.fields = fields
    self.gpu = opt.gpu
    self.cuda = opt.gpu > -1
    self.device = torch.device('cuda' if self.cuda else 'cpu')
    self.decode_extra_length = opt.decode_extra_length
    self.decode_min_length = opt.decode_min_length
    self.beam_size = opt.beam_size
    self.min_length = opt.min_length
    self.pre_paired_trans = opt.pre_paired_trans
    self.minimal_relative_prob = opt.minimal_relative_prob
    self.inference_mode = opt.inference_mode
    self.out_file = out_file
    self.tgt_eos_id = fields["tgt"].vocab.stoi[Constants.EOS_WORD]
    self.tgt_bos_id = fields["tgt"].vocab.stoi[Constants.BOS_WORD]
    self.src_eos_id = fields["src"].vocab.stoi[Constants.EOS_WORD]
    # new options
    self.segment_embedding = model_opt.segment_embedding
    # self.paired_trans = model_opt.paired_trans
    self.use_ord_ctx = model_opt.use_ord_ctx
    self.sentence_level = model_opt.sentence_level
    self.use_auto_trans = model_opt.use_auto_trans
    self.cross_attn = model_opt.cross_attn
    self.cross_before = model_opt.cross_before
    self.decoder_cross_before = model_opt.decoder_cross_before
    self.only_fixed = model_opt.only_fixed
    
    print(self.inference_mode)
    # self.src_mlm = model_opt.src_mlm
    # self.auto_truth_trans_kl = model_opt.auto_truth_trans_kl
    # # print(self.auto_truth_trans_kl)
    # print(self.src_mlm)
    # self.shift_num = model_opt.shift_num
    # print(self.shift_num)
  
  def build_tokens(self, idx, side="tgt"):
    assert side in ["src", "tgt", "tgt_tran"], "side should be either src or tgt"
    vocab = self.fields[side].vocab
    if side == "tgt":
      eos_id = self.tgt_eos_id
    else:
      eos_id = self.src_eos_id
    tokens = []
    for tok in idx:
      if tok == eos_id:
        break
      if tok < len(vocab):
        tokens.append(vocab.itos[tok])
    return tokens  
  
  def translate(self, src_data_iter, tgt_data_iter, tgt_tran_data_iter, batch_size, out_file=None):
    data = build_dataset(self.fields,
                         src_data_iter=src_data_iter,
                         tgt_data_iter=tgt_data_iter,
                         auto_trans_iter=tgt_tran_data_iter,
                         sentence_level=self.sentence_level,
                         pre_paired_trans = self.pre_paired_trans,
                         use_filter_pred=False)
    
    def sort_translation(indices, translation):
      ordered_transalation = [None] * len(translation)
      for i, index in enumerate(indices):
        ordered_transalation[index] = translation[i]
      return ordered_transalation
    
    if self.cuda:
        cur_device = "cuda"
    else:
        cur_device = "cpu"

    data_iter = OrderedIterator(
      dataset=data, device=cur_device,
      batch_size=batch_size, train=False, sort=True,
      sort_within_batch=True, shuffle=True)
    start_time = time.time()
    print("Begin decoding ...")
    batch_count = 0
    all_translation = []
    if self.sentence_level:
      for batch in data_iter:
        hyps, scores = self.translate_batch(batch)
        assert len(batch) == len(hyps)
        batch_transtaltion = []
        for src_idx_seq, tran_idx_seq, score in zip(batch.src[0].transpose(0, 1), hyps, scores):
          src_words = self.build_tokens(src_idx_seq, side='src')
          src = ' '.join(src_words)
          tran_words = self.build_tokens(tran_idx_seq, side='tgt')
          tran = ' '.join(tran_words)
          batch_transtaltion.append(tran)
          print("SOURCE: " + src + "\nOUTPUT: " + tran + "\n")
        for index, tran in zip(batch.indices.data, batch_transtaltion):
          while (len(all_translation) <=  index):
            all_translation.append("")
          all_translation[index] = tran
        batch_count += 1
        print("batch: " + str(batch_count) + "...")
    else:
      for batch in data_iter:
        # if document-level, need to proprecess batch
        num_doc, num_sents = batch.src[0].size(0), batch.src[0].size(1)
        batch.src = list(batch.src)
        batch.src[0] = batch.src[0].view(num_doc * num_sents, -1).transpose(0, 1)  # (seq_len, sents_num)
        # batch.tgt = batch.tgt.view(num_doc * num_sents, -1).transpose(0, 1)  # (seq_len, sents_num)
        batch.src = tuple(batch.src)
        if self.use_auto_trans:
          batch.tgt_tran = batch.tgt_tran.view(num_doc * num_sents, -1).transpose(0, 1) #(seq_len, sents_num)
        
        hyps, scores = self.translate_batch(batch)
        #assert len(batch) == len(hyps)
        batch_transtaltion = []
        src_doc = []
        tran_doc = []
        if self.use_auto_trans:
          auto_tran_doc = []
          for src_idx_seq, auto_trans_seq, tran_idx_seq, score in zip(batch.src[0].transpose(0, 1), batch.tgt_tran.transpose(0, 1), hyps, scores):
            src_words = self.build_tokens(src_idx_seq, side='src')
            src_words = list(' '.join(src_words))
            src_doc.append(src_words)
            src_doc.append('\n')
            tran_words = self.build_tokens(tran_idx_seq, side='tgt')
            tran_words = list(' '.join(tran_words))
            tran_doc.append(tran_words)
            tran_doc.append('\n')
            auto_tran_words = self.build_tokens(auto_trans_seq, side='tgt_tran')
            auto_tran_words = list(' '.join(auto_tran_words))
            auto_tran_doc.append(auto_tran_words)
            auto_tran_doc.append('\n')
          src_doc = list(_flatten(src_doc))
          tran_doc = list(_flatten(tran_doc))
          auto_tran_doc = list(_flatten(auto_tran_doc))
          src = ''.join(src_doc)
          tran = ''.join(tran_doc)
          auto_tran = ''.join(auto_tran_doc)
          batch_transtaltion.append(tran)
          print("SOURCE: " + src + auto_tran + "\nOUTPUT: " + tran + "\n")
        else:
          for src_idx_seq, tran_idx_seq, score in zip(batch.src[0].transpose(0, 1), hyps, scores):
            src_words = self.build_tokens(src_idx_seq, side='src')
            src_words = list(' '.join(src_words))
            src_doc.append(src_words)
            src_doc.append('\n')
            tran_words = self.build_tokens(tran_idx_seq, side='tgt')
            tran_words = list(' '.join(tran_words))
            tran_doc.append(tran_words)
            tran_doc.append('\n')
          src_doc = list(_flatten(src_doc))
          tran_doc = list(_flatten(tran_doc))
          src = ''.join(src_doc)
          tran = ''.join(tran_doc)
          batch_transtaltion.append(tran)
          print("SOURCE: " + src + "\nOUTPUT: " + tran + "\n")
        for index, tran in zip(batch.indices.data, batch_transtaltion):
          while (len(all_translation) <=  index):
            all_translation.append("")
          all_translation[index] = tran
        batch_count += 1
        print("batch: " + str(batch_count) + "...")
    
    if out_file is not None:
      for tran in all_translation:
        if self.sentence_level:
          out_file.write(tran + '\n')
        else:
          out_file.write(tran)
    print('Decoding took %.1f minutes ...'%(float(time.time() - start_time) / 60.))
  
  def translate_batch(self, batch):
    def get_inst_idx_to_tensor_position_map(inst_idx_list):
      ''' Indicate the position of an instance in a tensor. '''
      return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}
    
    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
      ''' Collect tensor parts associated to active instances. '''

      _, *d_hs = beamed_tensor.size()
      n_curr_active_inst = len(curr_active_inst_idx)
      new_shape = (n_curr_active_inst * n_bm, *d_hs)

      beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
      beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
      beamed_tensor = beamed_tensor.view(*new_shape)

      return beamed_tensor
    
    def beam_decode_step(
      inst_dec_beams, len_dec_seq, inst_idx_to_position_map, n_bm):
      ''' Decode and update beam status, and then return active beam idx '''
      # len_dec_seq: i (starting from 0)

      def prepare_beam_dec_seq(inst_dec_beams):
        dec_seq = [b.get_last_target_word() for b in inst_dec_beams if not b.done]
        # dec_seq: [(beam_size)] * batch_size
        dec_seq = torch.stack(dec_seq).to(self.device)
        # dec_seq: (batch_size, beam_size)
        dec_seq = dec_seq.view(1, -1)
        # dec_seq: (1, batch_size * beam_size)
        return dec_seq

      def predict_word(dec_seq, n_active_inst, n_bm, len_dec_seq):
        # dec_seq: (1, batch_size * beam_size)
        dec_output, *_ = self.model.decoder(dec_seq, step=len_dec_seq)
        # dec_output: (1, batch_size * beam_size, hid_size)
        word_prob = self.model.generator(dec_output.squeeze(0))
        # word_prob: (batch_size * beam_size, vocab_size)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        # word_prob: (batch_size, beam_size, vocab_size)

        return word_prob

      def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
        active_inst_idx_list = []
        select_indices_array = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
          is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
          if not is_inst_complete:
            active_inst_idx_list += [inst_idx]
            select_indices_array.append(inst_beams[inst_idx].get_current_origin() + inst_position * n_bm)
        if len(select_indices_array) > 0:
          select_indices = torch.cat(select_indices_array)
        else:
          select_indices = None
        return active_inst_idx_list, select_indices

      n_active_inst = len(inst_idx_to_position_map)

      dec_seq = prepare_beam_dec_seq(inst_dec_beams)
      # dec_seq: (1, batch_size * beam_size)
      word_prob = predict_word(dec_seq, n_active_inst, n_bm, len_dec_seq)

      # Update the beam with predicted word prob information and collect incomplete instances
      active_inst_idx_list, select_indices = collect_active_inst_idx_list(
        inst_dec_beams, word_prob, inst_idx_to_position_map)
      
      if select_indices is not None:
        assert len(active_inst_idx_list) > 0
        self.model.decoder.map_state(
            lambda state, dim: state.index_select(dim, select_indices))

      return active_inst_idx_list
    
    def collate_active_info(
        src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
      # Sentences which are still active are collected,
      # so the decoder will not run on completed sentences.
      n_prev_active_inst = len(inst_idx_to_position_map)
      active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
      active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

      active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
      active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
      active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

      return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def collect_best_hypothesis_and_score(inst_dec_beams):
      hyps, scores = [], []
      for inst_idx in range(len(inst_dec_beams)):
        hyp, score = inst_dec_beams[inst_idx].get_best_hypothesis()
        hyps.append(hyp)
        scores.append(score)
        
      return hyps, scores
    
    with torch.no_grad():
      #-- Encode
      src_seq = make_features(batch, 'src')
      src_lengths = batch.src[-1]
      # src: (seq_len_src, batch_size)
      if self.segment_embedding:
        # 1, sent_num*doc_num, hidden
        tgt_seg_emb = self.model.decoder.embeddings.get_seg_emb(sent_num=src_lengths.size(-1), doc_num=src_lengths.size(0), seq_len=1)
      else:
        tgt_seg_emb = None
      # src_emb, src_enc, src_mask = self.model.encoder(src_seq, batch.src[-1])
      
      # auto_cls_hidden = None
      # src_cls_hidden = None
      # auto_avg_hidden = None
      # src_avg_hidden = None
      tgt_tran_mask = None
      enc_mask = None
      memory_bank = None
      auto_trans_out = None
      if self.use_auto_trans:
        tgt_tran = make_features(batch, 'tgt_tran')
        # tgt_tran_mask, tgt_tran_emb = self.model.get_embeding_and_mask_before_encoding(self.model.decoder.embeddings, tgt_tran, src_lengths)
        
        if not self.only_fixed:
          _, memory_bank, enc_mask, auto_trans_out, tgt_tran_mask = self.model.encoder_forward(task_type="unified_enc", \
                                      lengths=src_lengths, src=src_seq, tgt_tran=tgt_tran)
          # if self.auto_truth_trans_kl:
          #   auto_avg_hidden, _ = self.model.avg_pooling_with_mask(auto_trans_out.transpose(0, 1), tgt_tran_mask)
          #   auto_cls_hidden = self.model.transfer_layer(auto_avg_hidden) # [doc_, dim]
            
          #   if self.src_mlm:
          #     src_avg_hidden, _ = self.model.avg_pooling_with_mask(memory_bank.transpose(0, 1), enc_mask)
          #     src_cls_hidden = self.model.transfer_layer(src_avg_hidden)
          
          
          # if self.auto_truth_trans_kl:
          #   auto_cls_hidden = auto_trans_out[0, :, :] # [doc_, dim]
            
          #   if self.src_mlm:
          #     src_cls_hidden = memory_bank[0, :, :]
          #       # _, memory_bank, enc_mask, auto_trans_out, _ = self.model.encoder(src_seq, src_length=src_lengths, \
          #       #                                                               trans_emb=tgt_tran_emb, trans_mask=tgt_tran_mask)
        if self.only_fixed:
          _, _, _, auto_trans_out, tgt_tran_mask = self.model.encoder_forward(task_type="tgt_enc", \
                           lengths=src_lengths, tgt_tran=tgt_tran)
          # _, memory_bank, enc_mask, auto_trans_out, _ = self.model.encoder(src_length=src_lengths, \
          #                                                                trans_emb=tgt_tran_emb, trans_mask=tgt_tran_mask)
      
      if not self.use_auto_trans:
        # _, memory_bank, enc_mask, auto_trans_out, tgt_tran_mask = self.model.encoder(src_seq, src_length=src_lengths)
        _, memory_bank, enc_mask, _, _ = self.model.encoder_forward(task_type="src_enc", \
                           lengths=src_lengths, src=src_seq)

      # if self.use_auto_trans:
      #   tgt_tran = make_features(batch, 'tgt_tran')
      #   tgt_tran_mask, tgt_tran_emb = self.model.get_embeding_and_mask_before_encoding(self.model.decoder.embeddings, tgt_tran, src_lengths)
      #   _, tgt_tran_bank, tgt_tran_mask = self.model.encoder(tgt_tran_emb, src_length=src_lengths, mask=tgt_tran_mask, input_type="embeddings")
      #   if self.paired_trans:
      #     src_enc, src_mask = self.model.pair_tran_src_enc_out(tgt_tran_bank, src_enc, tgt_tran_mask, src_mask, self.shift_num, src_lengths)
      # if not self.use_auto_trans:


      # src_emb: (seq_len_src, batch_size, emb_size)
      # src_enc: (seq_len_src, batch_size, hid_size)
      
      if self.inference_mode == "trans":
        self.model.decoder.init_state(src_seq, memory_bank, enc_mask, segment_embeding=tgt_seg_emb)
      
      if self.inference_mode == "repair":
        self.model.decoder.init_state(segment_embeding=tgt_seg_emb, \
                                    auto_trans_bank=auto_trans_out, auto_trans_mask=tgt_tran_mask)
      # self.model.decoder.init_state(src_seq, memory_bank, enc_mask, segment_embeding=tgt_seg_emb)
      src_len = src_seq.size(0)
      
      #-- Repeat data for beam search
      n_bm = self.beam_size
      n_inst = src_seq.size(1)
      self.model.decoder.map_state(lambda state, dim: tile(state, n_bm, dim=dim))
      # src_enc: (seq_len_src, batch_size * beam_size, hid_size)
      
      #-- Prepare beams
      decode_length = src_len + self.decode_extra_length
      decode_min_length = 0
      if self.decode_min_length >= 0:
        decode_min_length = src_len - self.decode_min_length
      inst_dec_beams = [Beam(n_bm, decode_length=decode_length, minimal_length=decode_min_length, minimal_relative_prob=self.minimal_relative_prob, bos_id=self.tgt_bos_id, eos_id=self.tgt_eos_id, device=self.device) for _ in range(n_inst)]
      
      #-- Bookkeeping for active or not
      active_inst_idx_list = list(range(n_inst))
      inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
      
      #-- Decode
      for len_dec_seq in range(0, decode_length):
        active_inst_idx_list = beam_decode_step(
          inst_dec_beams, len_dec_seq, inst_idx_to_position_map, n_bm)
        
        if not active_inst_idx_list:
          break  # all instances have finished their path to <EOS>

        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        
        
    batch_hyps, batch_scores = collect_best_hypothesis_and_score(inst_dec_beams)
    return batch_hyps, batch_scores
      
