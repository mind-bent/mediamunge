import os
import re
import io
import sys
import wave
import ffpb
import numpy as np
import pandas as pd
import argparse
import subprocess
import matplotlib.pyplot as plt
from pydub import AudioSegment
from difflib import SequenceMatcher
from deepspeech import Model


def compare(first, second):
    return SequenceMatcher(None, first, second).ratio()


def getMatches(csv, vtt, index, row, num):
    anchors = []
    try:
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


def videosplice(args, cut_times, adir):
    timeframes = ','.join(map(str, cut_times))
    print(f'Splicing video at {timeframes}')
    file_extension = os.path.splitext(args.input)[1][1:]
    process = subprocess.run([
        'ffmpeg',
        '-v',
        'warning',
        '-i',
        args.input,
        '-c',
        'copy',
        '-map',
        '0',
        '-f',
        'segment',
        '-segment_times',
        timeframes,
        f'{adir}/sentence_%03d.{file_extension}'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def process_silence(args, silences, wav_file, adir, wav=None, source_sr=None, encoder=None, speaker_embed=None, similarities=None):    
    # Loop through the silences (start time, end time, type) and try to find silences smaller than args.max_duration seconds and bigger than one second greedily
    # by trying to cut it at the biggest silence. Thus we will skip the first and last audio sample, but we don't care.
    sent_index, i, lost_seconds = 0, 0, 0
    cut_times = []  # (audio, file_name, timestamps)
    full_audio = AudioSegment.from_file(wav_file, 'wav')
    if args.splice_video:
        all_times = []
    while i < len(silences):
         # By default this looks at the audio between 2 and 7 seconds from where the silence starts
        start_period = silences[i][0] + args.min_duration
        end_period = silences[i][0] + args.max_duration
        j, last_med_silence, last_short_silence, last_long_silence = 1, None, None, None
        while i + j < len(silences) and silences[i+j][0] < start_period:
            j += 1
        while i + j < len(silences) and silences[i+j][0] < end_period:
            if silences[i+j][2] == 0:
                last_short_silence = j
            elif silences[i+j][2] == 1:
                last_med_silence = j
            else:
                last_long_silence = j
                break
            j += 1
        
        if last_long_silence is None:
            if last_med_silence is not None:
                j = last_med_silence
            elif last_short_silence is not None:
                j = last_short_silence
            else:
                if i+1 < len(silences):
                    lost_seconds += (silences[i+1][0]+silences[i+1][1])/2 - (silences[i][0]+silences[i][1])/2
                i += 1
                continue
        
        # 50% of silence duration as a margin for safety, sec to ms
        sent_start = (silences[i][0] + silences[i][1]) / 2
        sent_end = (silences[i+j][0] + silences[i+j][1]) / 2

        if args.remove_bad_segments:
            from resemblyzer.hparams import sampling_rate

            sent_wav = wav[int(sent_start * sampling_rate):int(sent_end * sampling_rate)]
            sent_embed = encoder.embed_utterance(sent_wav, rate=16)
            similarities.append(sent_embed @ speaker_embed)

        cut_times.append((full_audio[sent_start*1000:sent_end*1000], f"sentence_{sent_index}.wav", f"{sent_start}-{sent_end}"))
        if args.splice_video:
            all_times.append(sent_start)
        i += j
        sent_index += 1

    if args.splice_video:
        videosplice(args, all_times, adir)

    print(f"{lost_seconds : .2f} lost seconds")

    if args.remove_bad_segments:
        return cut_times, similarities
    else:
        return cut_times
    

def get_silence(args):
    # find out long (2), medium (1) and small (0) silence types
    # (type, noise_tol, noise_dur) for long, medium and small silences
    silence_params = [(2, -50, .5), (1, -35, .3), (0, -25, .15)]
    silences = []
    # Iterate through the list of grouped silence parameters (silence type, noise tolerance, noise duration) hard coded above
    print(f"Gathering {len(silence_params)} rounds of silence points based on {silence_params}")
    current_round = 1
    for sil_type, noise_tol, noise_dur in silence_params:
        print(f'Round {current_round} of {len(silence_params)}')
        print(f'Calculating silences based on Silence Type: {sil_type}, Noise Tolerance: {noise_tol}, Noise Duration: {noise_dur}')
        process = subprocess.run([
            'ffmpeg',
            '-i',
            args.input,
            '-af',
            f'silencedetect=noise={noise_tol}dB:d={noise_dur}',
            '-f',
            'null',
            '-'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        # Grab the silence start and end times from the results and map it with the silence type
        curr_silences = re.findall("\[silencedetect .* silence_start: (\d+(?:.\d*))\\n\[silencedetect .* silence_end: (\d+(?:.\d*))", process.stderr)
        silences.extend([(float(s[0]), float(s[1]), sil_type) for s in curr_silences])
        current_round += 1

    if len(silences) < 1:
        sys.exit('There was no silence to be found. Try another file or tweak the parameters.')
    else:
        print(f'Found {len(silences)} silence points to sort through')

    # Sort the list of silence times by the first column (start time)
    silences.sort(key=lambda x: x[0])

    return silences


def main(args, adir, csv):
    if not os.path.exists(adir):
        os.makedirs(adir)

    params_list = [item for param in args.wav_args.split("-")[1:] for item in f"-{param}".split(" ")[:2]]

    for item in os.listdir(args.model_dir):
        if '.pbmm' in item:
            modelPath =  f"{args.model_dir}{item}"
        if '.scorer' in item:
            scorerPath =  f"{args.model_dir}{item}"
    
    try:
        modelPath
        scorerPath
        ds = Model(modelPath)
        ds.enableExternalScorer(scorerPath)
        print("Model has been loaded.")
    except Exception:
        print("Are the model file paths correct?")
        sys.exit()

    # Format wav file name
    wav_file = f"{''.join(e for e in ''.join(args.input.split('/')[-1].split('.')[:-1]) if e.isalnum())}.wav"
    wav_file = f"{'/'.join(args.input.split('/')[:-1])}/{wav_file}"

    # Convert video to mono 14k audio wave file
    print(f"Converting media file {args.input} to {wav_file}")
    ffpb.main(
        argv=[
            "-i",
            args.input,
            "-ac",
            "1",
            "-ar",
            "16000",
            wav_file
        ]
    )

    silences = get_silence(args)

    # Filter out unwanted audio
    if args.remove_bad_segments:
        import librosa
        import matplotlib.pyplot as plt
        from resemblyzer import normalize_volume, VoiceEncoder
        from resemblyzer.hparams import sampling_rate, audio_norm_target_dBFS
        from pydub.playback import play
        wav, source_sr = librosa.load(wav_file, sr=None)
        wav = librosa.resample(wav, source_sr, sampling_rate)
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
        
        speaker_wav = wav[int(args.speaker_segment[0] * sampling_rate):int(args.speaker_segment[1] * sampling_rate)]
    
        print("Playing the selected audio segment at given offsets to check if it is alright")
        audio = AudioSegment.from_wav(wav_file)
        play(audio[int(args.speaker_segment[0]*1000):int(args.speaker_segment[1]*1000)])
        if input("Is this correct? (y/n)\n") != "y":
            exit(0)
            
        encoder = VoiceEncoder("cpu")
        speaker_embed = encoder.embed_utterance(speaker_wav)
        similarities = []
        cut_times, similarities = process_silence(args, silences, wav_file, adir, wav, source_sr, encoder, speaker_embed, similarities)
    else:
        cut_times = process_silence(args, silences, wav_file, adir)

    if args.remove_bad_segments:
        # selects the similarity threshold at which we will remove audio
        print("Find a separation threshold on the histogram between speeches spoken by your speaker (closer to 1) and others (closer to 0). Then close the figure")
        plt.hist(similarities, bins=50)
        plt.title("Histogram of the similarities (The higher the better)")
        plt.show()
        thr = -1
        
        while thr < 0 or thr > 1:
            str_thr = input("Please enter a valid threshold\n") 
            try:
                thr = float(str_thr)
            except ValueError as e:
                print("Value provided was not a float!")


    # Create a nested dict to convert to csv with first two column headers
    csv_file = {"file": [], "timestamps": []}

    # Add all the audio file names to the file column
    for i, (audio, file_name, timestamps) in enumerate(cut_times):
        if args.remove_bad_segments and similarities[i] < thr:
            continue
        print(f"Creating File Name: {file_name} Time Stamps {timestamps}")
        csv_file["file"].append(f"{file_name}")
        csv_file["timestamps"].append(timestamps)
        audio.export(os.path.join(adir, file_name), format="wav", parameters=params_list)
    
    csv_file["filesize"] = [""] * len(csv_file["file"])  # adding an empty column for wav file size
    csv_file["sentence"] = [""] * len(csv_file["file"])  # adding an empty column for transcription
    pd.DataFrame(csv_file).to_csv(csv, sep="|", index=False)
    files = pd.read_csv(csv, sep="|", dtype="string")
    print(f"Transcribing text that will be added to {csv}")
    for i in range(len(files)):
        file_name = os.path.join(adir, files.iat[i, 0])
        file_size = os.path.getsize(file_name)
        files.iat[i, 2] = str(file_size)
        fin = wave.open(file_name, 'rb')
        frames = fin.readframes(fin.getnframes())
        audio = np.frombuffer(frames, np.int16)
        text = ds.stt(audio)
        print(text)
        files.iat[i, 3] = text

    files.to_csv(csv, sep="|", index=False)
    print(f"FINISHED WRITING {csv}")

    if args.vtt:
        print(f"Using VTT file to help with annotation")
        vtt_file = set(line.strip() for line in open(args.vtt))
        vtt_csv = csv.replace('.csv', '-vtt.csv')

        # Format VTT text to big list of words
        vtt_text = []
        for l in vtt_file:
            if ">" not in l and ":" not in l and "WEBVTT" not in l and l != "":
                vtt_text.append(l.replace(f"[Music]", ""))
        full_list = ' '.join(vtt_text)
        vtt = full_list.split()

        # CSV should be written so open it up for sentence matching
        csv_f = pd.read_csv(csv, sep="|", dtype = {
                'file': "string",
                'timestamps': "string",
                'filesize': "string",
                'sentence': "string"
            })

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

        csv_f.to_csv(vtt_csv, sep="|", header=['file', 'timestamps', 'accuracy', 'sentence'], index=False)
        print(f"FINISHED WRITING {vtt_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Extracts audio, and splits it into smaller files""")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="One media file to work with")
    group.add_argument("--input_dir", help="Media directory to work with. Looks for mp4, mkv, webm, mp3")
    parser.add_argument("--model_dir", help="Path to directory containing the model .pbmm  and .scorer files", required=True)
    parser.add_argument("--splice_video", help="Splice video along with audio", action='store_true')
    parser.add_argument("--wav_args", help="List of arguments of the wav created files as string", default="-acodec pcm_s16le -ac 1 -ar 16000")
    parser.add_argument("--max_duration", help="Maximum duration (in seconds) a clip can last", default=7, type=int)
    parser.add_argument("--min_duration", help="Minimum duration (in seconds) a clip can last", default=2, type=int)
    parser.add_argument("--vtt", help="Path to VTT for CSV correcting and comparing. Not needed if input_dir is used (automatically checks).")
    parser.add_argument("--remove_bad_segments", action="store_true",
        help="Use this to automatically remove sentences not spoken by a speaker of interest (must be specified using the 'speaker_segment' argument")
    parser.add_argument("--speaker_segment", nargs=2, type=float, 
        help="Start and end time of a sample spoken by a speaker (seconds)")
    args = parser.parse_args()

    if args.input_dir:
        if os.path.isdir(args.input_dir):
            print(f"Using files in {args.input_dir}")
        else:
            sys.exit(f"This input directory: {args.input_dir} does not exist.")

        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        for item in os.listdir(args.input_dir):
            print(f"FOUND {args.input_dir}{item}")
            if [item.endswith(ext) for ext in ['mkv','mp4','mp3','webm']]: 
                basename = ''.join(e for e in '.'.join(item.split('.')[:-1]) if e.isalnum())
                adir = f"media/{basename}_audio/"
                csv = f"media/{basename}.csv"
                vtt_file = f"{''.join(item.split('.')[:-1])}.en.vtt"
                if os.path.isfile(f"{args.input_dir}{vtt_file}"):
                    vtt_f = f"{args.input_dir}{vtt_file}"
                    print(f"FOUND VTT FILE: {vtt_f}")
                else:
                    vtt_f = None
                new_args = Namespace(
                    input=f"{args.input_dir}{item}",
                    vtt=vtt_f,
                    splice_video=args.splice_video,
                    wav_args=args.wav_args,
                    model_dir=args.model_dir,
                    max_duration=args.max_duration,
                    min_duration=args.min_duration,
                    remove_bad_segments=False,
                )
                main(new_args, adir, csv)

    else:
        basename = ''.join([e for e in '.'.join(args.input.split('.')[:-1]).split('/')[-1] if e.isalnum()])
        adir = f"media/{basename}_audio/"
        csv = f"media/{basename}.csv"
        main(args, adir, csv)
