import codecs
import os
import argparse
def get_sentence_num(file_name, seg_tok= " ||| "):
    with open(file_name, 'r', encoding='utf-8') as infile:
        for line in infile:
            sentences = line.strip().split(seg_tok)
            yield len(sentences), sentences
        infile.close()

def get_docs(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    lines = file.readlines()
    docs = []
    for line in lines:
      sents = line.strip().split(' ||| ')
      docs.append(sents)
    file.close()
    return docs

def sentslist(file_name):
    sentences = []
    blank = 0
    with open(file_name, 'r', encoding='utf-8') as infile:
        for line in infile:
            sentence = line.strip()
            if sentence == "":
              sentence = "This is noise"
              blank += 1
            sentences.append(sentence)
        infile.close()
    print(blank)
    return sentences

def main(args):
    sent_file = args.sent_file
    doc_file = args.doc_file
    out_file = args.out_file
    split_tok = args.split_tok
    mode = args.mode
    out = open(out_file, 'w', encoding='utf-8')
    
    if mode == 'doc2sent':
      for sentences in get_docs(doc_file):
        out.write("\n".join(sentences) + "\n")
      out.close()
    
    if mode == "sent2doc":
      start = 0
      sents = sentslist(sent_file)
      sent_nums = 0
      for sent_num, _ in get_sentence_num(doc_file, seg_tok=" ||| "):
        sent_nums += sent_num
        out.write(split_tok.join(sents[start:start+sent_num]) + "\n")
        start += sent_num
      out.close()
      print(sent_nums)
      print(len(sents))

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='klk')
    
   parser.add_argument("--sent_file", type=str, default="",
                         help="the sentence-level source file, every line contain a sentence ")
   parser.add_argument("--doc_file", type=str, default="", 
                         help="the sentence-level translation file, every line contain a sentence ")
   parser.add_argument("--out_file", type=str, required=True, 
                         help="the sentence-level alignment file, every line contain a src-trans pair aligment")
   parser.add_argument("--mode", type=str, required=True, 
                         help="the sentence-level alignment file, every line contain a src-trans pair aligment")
   
   parser.add_argument("--split_tok", type=str, default=" ||| ", 
                         help="the sentence-level alignment file, every line contain a src-trans pair aligment")
   
   args = parser.parse_args()
   main(args)


    