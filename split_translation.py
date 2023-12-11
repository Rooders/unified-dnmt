import argparse
def main(args):
   trans = open(args.path_trans, 'r', encoding='utf-8')
   all_trans = [line.strip().split(args.split_tok)[-1] for line in trans]
   out_file = open(args.out_file, 'w', encoding='utf-8')
   out_file.write("\n".join(all_trans))
   trans.close()
   out_file.close()

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Post-processing of paired translation")
   parser.add_argument("--split_tok", type=str, default=" <seg> ", help="split token of paired translation")
   parser.add_argument("--path_trans", type=str, required=True, help="path of paired translation")
   parser.add_argument("--out_file", type=str, required=True, help="path of file to store")
   args = parser.parse_args()
   main(args)