import sys
import pandas as pd
import argparse
from difflib import SequenceMatcher

'''Matches VTT subtitle text with transcribed DeepSpeech transcription from a pretrained model and provides accuracy percentage.
   Creates a CSV that can be used by annotate.py for corrections and confirmation.
   End goal is for quicker preparations of materials for DeepSpeech training.
'''

parser = argparse.ArgumentParser("""Testing sequence in a sequence sentence comparison""")
parser.add_argument("--ttc", help="Path to ttc", required=True)
parser.add_argument("--csv", help="Path to csv", required=True)
args = parser.parse_args()
ttc_file = set(line.strip() for line in open(args.ttc))
ttc_csv = args.csv.replace('.csv', '-ttc.csv')

def compare(first, second):
    return SequenceMatcher(None, first, second).ratio()


# Format TTC text to big list of words
ttc_text = []
for l in ttc_file:
    if ">" not in l and ":" not in l and "WEBVTT" not in l and l != "":
        ttc_text.append(l.replace(f"[Music]", ""))

full_list = ' '.join(ttc_text)
ttc = full_list.split()

csv_f = pd.read_csv(args.csv, sep="|", dtype = {
        'file': "string",
        'timestamps': "string",
        'filesize': "string",
        'sentence': "string"
    })


def getMatches(csv, ttc, num):
    ''' Fuzzes the sequence (needle) to find within the larger sequence (haystack) based on the length and anchor point to get a better match.
    '''
    anchors = []
    try:
        # Search for the anchor
        found = [item for item in range(len(ttc)) if ttc[item] == csv[num]]

        for word in found:
            matches = [{'index':index, 'items':
                [
                    {'match':compare(row['sentence'], ' '.join(ttc[word:word+len(csv)])),
                        'sen':' '.join(ttc[word:word+len(csv)])},
                    {'match':compare(row['sentence'], ' '.join(ttc[word:word+len(csv)+1])),
                        'sen':' '.join(ttc[word:word+len(csv)+1])},
                    {'match':compare(row['sentence'], ' '.join(ttc[word:word+len(csv)+2])),
                        'sen':' '.join(ttc[word:word+len(csv)+2])},
                    {'match':compare(row['sentence'], ' '.join(ttc[word:word+len(csv)+3])),
                        'sen':' '.join(ttc[word:word+len(csv)+3])},
                    {'match':compare(row['sentence'], ' '.join(ttc[word:word+len(csv)-1])),
                        'sen':' '.join(ttc[word:word+len(csv)-1])},
                    {'match':compare(row['sentence'], ' '.join(ttc[word:word+len(csv)-2])),
                        'sen':' '.join(ttc[word:word+len(csv)-2])},
                    {'match':compare(row['sentence'], ' '.join(ttc[word:word+len(csv)-3])),
                        'sen':' '.join(ttc[word:word+len(csv)-3])}
                ]}]
            anchors.append([max(dVals["items"], key=lambda x: x["match"]) for dVals in matches][0])
    except Exception as e:
        print(f'Range issues with {num}')
    return anchors


for index, row in csv_f.iterrows():
    anchors = []
    length = len(csv_f.index)
    if not pd.isna(row['sentence']):
        split_row = row['sentence'].split()
        print(f"\nSENTENCE {index+1} of {length}: {row['sentence']}")
        anchors.extend(getMatches(split_row, ttc, 0))
        anchors.extend(getMatches(split_row, ttc, 1))
        anchors.extend(getMatches(split_row, ttc, 2))
    else:
        print(f"\nSENTENCE {index+1} of {length} is empty.")

    if anchors:
        anchor_max = max(anchors, key=lambda x: x["match"])
        print(f"MATCH {round(anchor_max['match'] * 100)}% SENTENCE {anchor_max['sen']}")
        csv_f.iat[index,3] = anchor_max['sen']
        csv_f.iat[index,2] = str(round(anchor_max['match']*100))
    else:
        print("FOUND NOTHING")

csv_f.to_csv(ttc_csv, sep="|", header=['file', 'timestamps', 'accuracy', 'sentence'], index=False)
print(f"FINISHED WRITING {ttc_csv}")
