import os
import sys
import pandas as pd
import argparse
from difflib import SequenceMatcher


def compare(first, second):
    return SequenceMatcher(None, first, second).ratio()


def prep_vtt(vtt_file):
    # Format VTT text to big list of words
    vtt_text = []
    for l in vtt_file:
        if ">" not in l and ":" not in l and "WEBVTT" not in l and l != "":
            vtt_text.append(l.replace(f"[Music]", ""))

    full_list = ' '.join(vtt_text)
    return full_list.split()


def prep_csv(csv_file):
    csv_f = pd.read_csv(csv_file, sep="|", dtype = {
        'file': "string",
        'timestamps': "string",
        'filesize': "string",
        'sentence': "string"
    })
    return csv_f


def getMatches(csv, vtt, index, row, num):
    ''' Fuzzes the sequence (needle) to find within the larger sequence (haystack) based on the length and anchor point to get a better match.
    '''
    anchors = []
    try:
        # Search for the anchor
        found = [item for item in range(len(vtt)) if vtt[item] == csv[num]]
        for word in found:
            matches = [{'index':index, 'items':
                [
                    {'match':compare(row['sentence'], ' '.join(vtt[word:word+len(csv)])),
                        'sen':' '.join(vtt[word:word+len(csv)])},
                    {'match':compare(row['sentence'], ' '.join(vtt[word:word+len(csv)+1])),
                        'sen':' '.join(vtt[word:word+len(csv)+1])},
                    {'match':compare(row['sentence'], ' '.join(vtt[word:word+len(csv)+2])),
                        'sen':' '.join(vtt[word:word+len(csv)+2])},
                    {'match':compare(row['sentence'], ' '.join(vtt[word:word+len(csv)+3])),
                        'sen':' '.join(vtt[word:word+len(csv)+3])},
                    {'match':compare(row['sentence'], ' '.join(vtt[word:word+len(csv)-1])),
                        'sen':' '.join(vtt[word:word+len(csv)-1])},
                    {'match':compare(row['sentence'], ' '.join(vtt[word:word+len(csv)-2])),
                        'sen':' '.join(vtt[word:word+len(csv)-2])},
                    {'match':compare(row['sentence'], ' '.join(vtt[word:word+len(csv)-3])),
                        'sen':' '.join(vtt[word:word+len(csv)-3])}
                ]}]
            anchors.append([max(dVals["items"], key=lambda x: x["match"]) for dVals in matches][0])
    except Exception as e:
        print(f'Range issues with {num}')
    return anchors


def matchMaker(csv_f, vtt):
    for index, row in csv_f.iterrows():
        anchors = []
        length = len(csv_f.index)
        if not pd.isna(row['sentence']):
            split_row = row['sentence'].split()
            print(f"\nSENTENCE {index+1} of {length}: {row['sentence']}")
            anchors.extend(getMatches(split_row, vtt, index, row, 0))
            anchors.extend(getMatches(split_row, vtt, index, row, 1))
            anchors.extend(getMatches(split_row, vtt, index, row, 2))
        else:
            print(f"\nSENTENCE {index+1} of {length} is empty.")

        if anchors:
            anchor_max = max(anchors, key=lambda x: x["match"])
            print(f"MATCH {round(anchor_max['match'] * 100)}% SENTENCE {anchor_max['sen']}")
            csv_f.iat[index,3] = anchor_max['sen']
            csv_f.iat[index,2] = str(round(anchor_max['match']*100))
        else:
            print("FOUND NOTHING")
    return csv_f


def checkForCSV(VTT):
    # Format file name to search for the matching CSV
    basename = ''.join(e for e in VTT.replace('.en.vtt','').split('/')[-1] if e.isalnum())
    maybeVTT = f"media/{basename}.en.vtt"
    if os.path.isfile(maybeVTT):
        # Could provide redo option here
        print(f"{maybeVTT} already exists....skipping.")
        return ''
    if os.path.isfile(f"media/{basename}.csv"):
        csv_file = f"media/{basename}.csv"
        print(f"Found matching CSV: {csv_file}")
        return csv_file
    else:
        print(f"No matching CSV was found for media/{basename}.csv")
        return ''


def main(args):
    if args.dir:
        if os.path.isdir(args.dir):
            for stuff in os.listdir(args.dir):
                if stuff.endswith('.en.vtt'):
                    vtt_f = f"{args.dir}{stuff}"
                    print(f"Found {vtt_f}")
                    csv_file = checkForCSV(vtt_f)
                    if not csv_file:
                        continue
                    csv_f = prep_csv(csv_file)
                    vtt = prep_vtt(set(line.strip() for line in open(vtt_f)))
                    csv_f = matchMaker(csv_f, vtt)
                    vtt_csv = csv_file.replace('.csv', '-vtt.csv')
                    csv_f.to_csv(vtt_csv, sep="|", header=['file', 'timestamps', 'accuracy', 'sentence'], index=False)
                    print(f"FINISHED WRITING {vtt_csv}")
        else:
            sys.exit(f"{args.dir} is not a directory")
    else:
        #vtt_file = set(line.strip() for line in open(args.vtt))
        vtt = prep_vtt(set(line.strip() for line in open(args.vtt)))
        csv_file = checkForCSV(args.vtt)
        if not csv_file:
            sys.exit("No matching CSV file to work with.")
        csv_f = prep_csv(csv_file)
        vtt = prep_vtt(set(line.strip() for line in open(vtt_f)))
        csv_f = matchMaker(csv_f, vtt)
        vtt_csv = csv_file.replace('.csv', '-vtt.csv')
        csv_f.to_csv(vtt_csv, sep="|", header=['file', 'timestamps', 'accuracy', 'sentence'], index=False)
        print(f"FINISHED WRITING {vtt_csv}")


if __name__ == "__main__":
    '''Matches VTT subtitle text with transcribed DeepSpeech transcription from a pretrained model and provides accuracy percentage.
       Creates a CSV that can be used by annotate.py for corrections and confirmation.
       End goal is for quicker preparations of materials for DeepSpeech training.
    '''
    parser = argparse.ArgumentParser("Point to a folder or a VTT file to match with an already created CSV file in the media folder.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vtt", help="Path to vtt")
    group.add_argument("--dir", help="Path to directory with vtt files")
    args = parser.parse_args()
    main(args)
