import subprocess
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='.', type=str)
    args, _ = parser.parse_known_args()

    for doc in os.listdir(args.dir):
        doc_path = os.path.join(args.dir, doc)
        doc_name = doc.replace('.txt', '')

        # make folders
        txt_folder = os.path.join(args.dir, 'txt')
        os.makedirs(txt_folder, exist_ok=True)
        rtf_folder = os.path.join(args.dir, 'rtf')
        os.makedirs(rtf_folder, exist_ok=True)
        
        # convert to RTF
        output_doc_path = os.path.join(rtf_folder, doc.replace('.txt', '.rtf'))
        cmd = f"pandoc {doc_path} -f markdown -t rtf -s -o {output_doc_path}"
        print(cmd)
        subprocess.run(cmd, shell=True)

        # move .txt to txt folder
        os.rename(doc_path, os.path.join(txt_folder, doc))
