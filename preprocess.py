import configargparse
import glob
import os
import codecs
import gc

import torch
import torchtext.vocab
from collections import Counter, OrderedDict
from tkinter import _flatten
import onmt.constants as Constants
import onmt.opts as opts
from inputters.dataset import get_fields, build_dataset, make_text_iterator_from_file
from utils.logging import init_logger, logger

def save_fields_to_vocab(fields):
  """
  Save Vocab objects in Field objects to `vocab.pt` file.
  """
  vocab = []
  for k, f in fields.items():
    if f is not None and 'vocab' in f.__dict__:
      f.vocab.stoi = f.vocab.stoi
      vocab.append((k, f.vocab))
  return vocab

def build_field_vocab(field, counter, **kwargs):
  specials = list(OrderedDict.fromkeys(
    tok for tok in [Constants.UNK_WORD, Constants.PAD_WORD, Constants.BOS_WORD, Constants.EOS_WORD, 
                    Constants.SLU_WORD, Constants.SEG_WORD, Constants.MASK_WORD]
    if tok is not None))
  field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)

def merge_vocabs(vocabs, vocab_size=None, min_frequency=1):
  merged = sum([vocab.freqs for vocab in vocabs], Counter())
  return torchtext.vocab.Vocab(merged,
                               specials=[Constants.UNK_WORD, Constants.PAD_WORD,
                                         Constants.BOS_WORD, Constants.SLU_WORD, Constants.SEG_WORD, Constants.MASK_WORD],
                               max_size=vocab_size,
                               min_freq=min_frequency)    

def build_vocab(train_dataset_files, fields, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size, tgt_words_min_frequency, sentence_level):
  counter = {}

  for k in fields:
    counter[k] = Counter()

  # Load vocabulary
  for _, path in enumerate(train_dataset_files):
    dataset = torch.load(path)
    logger.info(" * reloading %s." % path)
    for ex in dataset.examples:
      for k in fields:
        val = getattr(ex, k, None)
        if not fields[k].sequential:
          continue
        if sentence_level:
          counter[k].update(val)
        else:
          for sentence_val in val:
            counter[k].update(sentence_val)
    dataset.examples = None
    gc.collect()
    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()
  if sentence_level:
    src_field = fields["src"]
    tgt_field = fields["tgt"]
    if "tgt_tran" in fields.keys():
      auto_trans_field = fields["tgt_tran"]
  else:
    src_field = fields["src"].nesting_field
    tgt_field = fields["tgt"].nesting_field
    if "tgt_tran" in fields.keys():
      auto_trans_field = fields["tgt_tran"].nesting_field
  
  


  build_field_vocab(tgt_field, counter["tgt"],
                     max_size=tgt_vocab_size,
                     min_freq=tgt_words_min_frequency)
  fields["tgt"].vocab = tgt_field.vocab
  
  logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))
  

  build_field_vocab(src_field, counter["src"],
                     max_size=src_vocab_size,
                     min_freq=src_words_min_frequency)
  fields["src"].vocab = src_field.vocab
  logger.info(" * src vocab size: %d." % len(fields["src"].vocab))
  
  if "tgt_tran" in fields.keys():
    fields["tgt_tran"].vocab = tgt_field.vocab
    logger.info(" use the auto translation, the vocab size of Auto Translation is set same with tgt vocab")
    logger.info(" * auto tgt vocab size: %d." % len(fields["tgt_tran"].vocab))


  # Merge the input and output vocabularies.
  if share_vocab:
    # `tgt_vocab_size` is ignored when sharing vocabularies
    logger.info(" * merging src and tgt vocab...")
    merged_vocab = merge_vocabs(
        [fields["src"].vocab, fields["tgt"].vocab],
        vocab_size=src_vocab_size,
        min_frequency=src_words_min_frequency)
    fields["src"].vocab = merged_vocab
    fields["tgt"].vocab = merged_vocab
    
    logger.info(" * src vocab size: %d." % len(fields["src"].vocab))
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))
    if "tgt_tran" in fields.keys():
      fields["tgt_tran"].vocab = merged_vocab
      logger.info(" * auto trans vocab size: %d." % len(fields["tgt_tran"].vocab))

  return fields

def parse_args():
  parser = configargparse.ArgumentParser(
    description='preprocess.py',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

  opts.config_opts(parser)
  opts.preprocess_opts(parser)

  opt = parser.parse_args()
  torch.manual_seed(opt.seed)

  return opt

def build_save_in_shards_using_shards_size(src_corpus, tgt_corpus, auto_trans_corpus,
                                           fields,
                                           corpus_type, opt):
  src_data = []
  tgt_data = []
  auto_trans_data = []
  if auto_trans_corpus is not None:
    with open(src_corpus, "r") as src_file:
      with open(tgt_corpus, "r") as tgt_file:
        with open(auto_trans_corpus, "r") as auto_trans_file:
          for s, t, a in zip(src_file, tgt_file, auto_trans_file):
            src_data.append(s)
            tgt_data.append(t)
            auto_trans_data.append(a)
    if not opt.sentence_level:
      src_len = sum([len(doc.split(" ||| ")) for doc in src_data])
      tgt_len = sum([len(doc.split(" ||| ")) for doc in tgt_data])
      auto_trans_len = sum([len(doc.split(" ||| ")) for doc in auto_trans_data])
      if auto_trans_len != tgt_len != src_len:
        raise AssertionError("Source, Target and Auto-trans should \
                              have the same length")
    else:
      if len(src_data) != len(tgt_data) != len(auto_trans_data):
        raise AssertionError("Source, Target and Auto-trans should \
                              have the same length")
  if auto_trans_corpus is None:
    with open(src_corpus, "r") as src_file:
      with open(tgt_corpus, "r") as tgt_file:
        for s, t in zip(src_file, tgt_file):
          src_data.append(s)
          tgt_data.append(t)

    if not opt.sentence_level:
      src_len = sum([len(doc.split(" ||| ")) for doc in src_data])
      tgt_len = sum([len(doc.split(" ||| ")) for doc in tgt_data])
      if tgt_len != src_len:
        raise AssertionError("Source, Target and Auto-trans should \
                              have the same length")
    else:
      if len(src_data) != len(tgt_data):
        raise AssertionError("Source, Target and Auto-trans should \
                              have the same length")

  num_shards = int(len(src_data) / opt.shard_size)
  for x in range(num_shards):
    logger.info("Splitting shard %d." % x)
    f = codecs.open(src_corpus + ".{0}.txt".format(x), "w",
                    encoding="utf-8")
    f.writelines(
            src_data[x * opt.shard_size: (x + 1) * opt.shard_size])
    f.close()
    f = codecs.open(tgt_corpus + ".{0}.txt".format(x), "w",
                    encoding="utf-8")
    f.writelines(
            tgt_data[x * opt.shard_size: (x + 1) * opt.shard_size])
    f.close()
    if auto_trans_corpus is not None:
      f = codecs.open(auto_trans_corpus + ".{0}.txt".format(x), "w",
                      encoding="utf-8")
      f.writelines(
              auto_trans_data[x * opt.shard_size: (x + 1) * opt.shard_size])
      f.close()
  num_written = num_shards * opt.shard_size

  if len(src_data) > num_written:
    logger.info("Splitting shard %d." % num_shards)
    f = codecs.open(src_corpus + ".{0}.txt".format(num_shards),
                    'w', encoding="utf-8")
    f.writelines(
            src_data[num_shards * opt.shard_size:])
    f.close()
    f = codecs.open(tgt_corpus + ".{0}.txt".format(num_shards),
                    'w', encoding="utf-8")
    f.writelines(
            tgt_data[num_shards * opt.shard_size:])
    f.close()
    if auto_trans_corpus is not None:
      f = codecs.open(auto_trans_corpus + ".{0}.txt".format(num_shards),
                      'w', encoding="utf-8")
      f.writelines(
              auto_trans_data[num_shards * opt.shard_size:])
      f.close()
  
  
  
  src_list = sorted(glob.glob(src_corpus + '.*.txt'))
  tgt_list = sorted(glob.glob(tgt_corpus + '.*.txt'))
  if auto_trans_corpus is not None:
    auto_trans_list = sorted(glob.glob(auto_trans_corpus + '.*.txt'))
  ret_list = []
  for index, src in enumerate(src_list):
    logger.info("Building shard %d." % index)
    src_iter = make_text_iterator_from_file(src)
    tgt_iter = make_text_iterator_from_file(tgt_list[index])
    if auto_trans_corpus is not None:
      auto_trans_iter = make_text_iterator_from_file(auto_trans_list[index])
    else:
      auto_trans_iter = None
     
    
    dataset = build_dataset(
      fields,
      src_iter,
      tgt_iter,
      auto_trans_iter,
      src_seq_length=opt.src_seq_length,
      tgt_seq_length=opt.tgt_seq_length,
      src_seq_length_trunc=opt.src_seq_length_trunc,
      tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
      sentence_level=opt.sentence_level, pre_paired_trans=opt.pre_paired_trans
    )

    pt_file = "{:s}_{:s}.{:d}.pt".format(
      opt.save_data, corpus_type, index)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    logger.info(" * saving %sth %s data shard to %s."
                % (index, corpus_type, pt_file))
    torch.save(dataset, pt_file)

    ret_list.append(pt_file)
    os.remove(src)
    os.remove(tgt_list[index])
    if auto_trans_corpus is not None:
      os.remove(auto_trans_list[index])
    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

  return ret_list

def store_vocab_to_file(vocab, filename):
  with open(filename, "w") as f:
    for i, token in enumerate(vocab.itos):
      f.write(str(i)+ ' ' + token + '\n')
    f.close()

def build_save_vocab(train_dataset, fields, opt):
  """ Building and saving the vocab """
  fields = build_vocab(train_dataset, fields,
                                 opt.share_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab_size,
                                 opt.tgt_words_min_frequency,
                                 opt.sentence_level)

  # Can't save fields, so remove/reconstruct at training time.
  vocab_file = opt.save_data + '_vocab.pt'
  torch.save(save_fields_to_vocab(fields), vocab_file)
  store_vocab_to_file(fields['src'].vocab, opt.save_data + '_src_vocab')
  store_vocab_to_file(fields['tgt'].vocab, opt.save_data + '_tgt_vocab')
    
def build_save_dataset(corpus_type, fields, opt):
  """ Building and saving the dataset """
  assert corpus_type in ['train', 'valid']

  if corpus_type == 'train':
    src_corpus = opt.train_src
    tgt_corpus = opt.train_tgt
    if opt.use_auto_trans:
      auto_trans_corpus = opt.train_auto_trans
    else:
      auto_trans_corpus = None
  else:
    src_corpus = opt.valid_src
    tgt_corpus = opt.valid_tgt
    if opt.use_auto_trans:
      auto_trans_corpus = opt.valid_auto_trans
    else:
      auto_trans_corpus = None

  if (opt.shard_size > 0):
    return build_save_in_shards_using_shards_size(src_corpus,
                                                  tgt_corpus,
                                                  auto_trans_corpus,
                                                  fields,
                                                  corpus_type,
                                                  opt)

  # We only build a monolithic dataset.
  # But since the interfaces are uniform, it would be not hard
  # to do this should users need this feature.
  src_iter = make_text_iterator_from_file(src_corpus)
  tgt_iter = make_text_iterator_from_file(tgt_corpus)
  if opt.use_auto_trans:
    auto_trans_iter = make_text_iterator_from_file(auto_trans_corpus)
  else:
    auto_trans_iter = None
  
  dataset = build_dataset(
    fields,
    src_iter,
    tgt_iter,
    auto_trans_iter,
    src_seq_length=opt.src_seq_length,
    tgt_seq_length=opt.tgt_seq_length,
    src_seq_length_trunc=opt.src_seq_length_trunc,
    tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
    sentence_level=opt.sentence_level, pre_paired_trans=opt.pre_paired_trans)
  
  # We save fields in vocab.pt seperately, so make it empty.
  dataset.fields = []

  pt_file = "{:s}_{:s}.pt".format(opt.save_data, corpus_type)
  logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))
  torch.save(dataset, pt_file)

  return [pt_file]

def main():
  opt = parse_args()
  if (opt.shuffle > 0):
    raise AssertionError("-shuffle is not implemented, please make sure \
                         you shuffle your data before pre-processing.")
  init_logger(opt.log_file)
  logger.info("Input args: %r", opt)
  logger.info("Extracting features...")

  logger.info("Building `Fields` object...")
  fields = get_fields(sentence_level=opt.sentence_level, use_auto_trans=opt.use_auto_trans)

  logger.info("Building & saving training data...")
  train_dataset_files = build_save_dataset('train', fields, opt)

  logger.info("Building & saving validation data...")
  build_save_dataset('valid', fields, opt)

  logger.info("Building & saving vocabulary...")

  build_save_vocab(train_dataset_files, fields, opt)

if __name__ == "__main__":
  main()
  
