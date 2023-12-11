import argparse
from xml.etree.ElementInclude import default_loader
from xml.sax import parseString


def read_docs(file_name, split_tok="|||"):
   file = open(file_name, 'r', encoding="utf-8")
   docs = []
   for line in file.readlines():
      doc = line.split(split_tok)
      final_doc = ["this is a fake sentence" if sent.strip() == "" else sent.strip() for sent in doc]
      docs.append(final_doc)
   # docs = [line.split(split_tok) for line in file.readlines()]
   # final_doc = ["this is a fake sentence" if line.strip() == "" else line.strip() for line in docs]
   file.close()
   return docs

def mini_doc_bulider(src_docs, tgt_docs, tran_docs=None, max_length=512):
   assert(len(src_docs) == len(tgt_docs) == len(tran_docs))
   all_src_mini_docs = []
   all_tgt_mini_docs = []
   all_tran_mini_docs = []
   for i, (src_doc, tgt_doc, tran_doc) in enumerate(zip(src_docs, tgt_docs, tran_docs)):
     assert(len(src_doc) == len(tgt_doc) == len(tran_doc))
     min_src_docs = []
     min_src_doc = []
     min_tgt_docs = []
     min_tgt_doc = []
     min_tran_docs = []
     min_tran_doc = []
     upper_len = 0
     for src_sent, tgt_sent, tran_sent in zip(src_doc, tgt_doc, tran_doc):
       upper_len += max(len(src_sent.split()), len(tgt_sent.split()), len(tran_sent.split()))
       if upper_len <= max_length:
         min_src_doc.append(src_sent)
         min_tgt_doc.append(tgt_sent)
         min_tran_doc.append(tran_sent)
       if upper_len > max_length:
         min_src_docs.append(min_src_doc)
         min_tgt_docs.append(min_tgt_doc)
         min_tran_docs.append(min_tran_doc)
         min_src_doc = [src_sent]
         min_tgt_doc = [tgt_sent]
         min_tran_doc = [tran_sent]
         upper_len = max(len(src_sent.split()), len(tgt_sent.split()), len(tran_sent.split()))
     
     min_src_docs.append(min_src_doc)
     min_tgt_docs.append(min_tgt_doc)
     min_tran_docs.append(min_tran_doc)
     all_src_mini_docs.extend(min_src_docs)
     all_tgt_mini_docs.extend(min_tgt_docs)
     all_tran_mini_docs.extend(min_tran_docs)
   assert(len(all_src_mini_docs) == len(all_tgt_mini_docs) == len(all_tran_mini_docs))
   return all_src_mini_docs, all_tgt_mini_docs, all_tran_mini_docs

   


def main(args):
   src_docs = read_docs(args.src_doc_path)
   tgt_docs = read_docs(args.tgt_doc_path)
   tran_docs = read_docs(args.tran_doc_path)
   mini_src_docs, mini_tgt_docs, mini_tran_docs = mini_doc_bulider(src_docs, tgt_docs, tran_docs, args.max_length)
   src_mini_file = open(args.src_doc_path + ".mini", 'w', encoding='utf-8')
   tgt_mini_file = open(args.tgt_doc_path + ".mini", 'w', encoding='utf-8')
   tran_mini_file = open(args.tran_doc_path + ".mini", 'w', encoding='utf-8')
   
   mini_src_docs = [" ||| ".join(doc) for doc in mini_src_docs]
   mini_tgt_docs = [" ||| ".join(doc) for doc in mini_tgt_docs]
   mini_tran_docs = [" ||| ".join(doc) for doc in mini_tran_docs]
   src_mini_file.write("\n".join(mini_src_docs) + "\n")
   tgt_mini_file.write("\n".join(mini_tgt_docs) + "\n")
   tran_mini_file.write("\n".join(mini_tran_docs) + "\n")
   
if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Post-processing of paired translation")
   parser.add_argument("--src_doc_path", type=str, required=True, help="split token of paired translation")
   parser.add_argument("--tgt_doc_path", type=str, required=True, help="path of paired translation")
   parser.add_argument("--tran_doc_path", type=str, default="", help="path of file to store")
   parser.add_argument("--max_length", type=int, default=512, help="the max number of token in per document")
   
   args = parser.parse_args()
   main(args)



