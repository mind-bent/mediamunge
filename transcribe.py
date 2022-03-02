import os
import re
import io
import sys
import wave
import ffpb
import numpy as np
import pandas as pd
import librosa
import argparse
import subprocess
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
from deepspeech import Model
from resemblyzer import normalize_volume, VoiceEncoder
from resemblyzer.hparams import sampling_rate, audio_norm_target_dBFS
from resemblyzer.hparams import sampling_rate, audio_norm_target_dBFS

modelPath =  '../models/v0.9.3/deepspeech-0.9.3-models.pbmm'
scorerPath = '../models/v0.9.3/deepspeech-0.9.3-models.scorer'

try:
    ds = Model(modelPath)
    ds.enableExternalScorer(scorerPath)
except Exception:
    print("Are the model file paths correct?")


def videosplice(args, cut_times):
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
        f'{args.audio_folder}/sentence_%03d.{file_extension}'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    print(f'RESULTS {process.stdout} and {process.stderr}')


def process_silence(args, silences, wav_file, wav=None, source_sr=None, encoder=None, speaker_embed=None, similarities=None):    
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
            sent_wav = wav[int(sent_start * sampling_rate):int(sent_end * sampling_rate)]
            sent_embed = encoder.embed_utterance(sent_wav, rate=16)
            similarities.append(sent_embed @ speaker_embed)

        cut_times.append((full_audio[sent_start*1000:sent_end*1000], f"sentence_{sent_index}.wav", f"{sent_start}-{sent_end}"))
        if args.splice_video:
            all_times.append(sent_start)
        i += j
        sent_index += 1

    if args.splice_video:
        videosplice(args, all_times)

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
        print('There was no silence to be found. Try another file or tweak the parameters.')
        sys.exit()
    else:
        print(f'Found {len(silences)} silence points to sort through')

    # Sort the list of silence times by the first column (start time)
    silences.sort(key=lambda x: x[0])

    return silences


def main(args):
    if not os.path.exists(args.audio_folder):
        os.makedirs(args.audio_folder)
    params_list = [item for param in args.wav_args.split("-")[1:] for item in f"-{param}".split(" ")[:2]]
    # Convert video to mono 14k audio wave file
    wav_file = args.input.split('.')[0].split("-")[0].replace(" ", "") +".wav"
    print(f"Converting video file to {wav_file}")
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
        cut_times, similarities = process_silence(args, silences, wav_file, wav, source_sr, encoder, speaker_embed, similarities)
    else:
        cut_times = process_silence(args, silences, wav_file)

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
        print(f"Creating File Name: {file_name} Time Stamps {timestamps}")
        if args.remove_bad_segments and similarities[i] < thr:
            continue
        csv_file["file"].append(os.path.abspath(args.audio_folder) +"/"+ file_name)
        csv_file["timestamps"].append(timestamps)
        audio.export(os.path.join(args.audio_folder, file_name), format="wav", parameters=params_list)
    
    csv_file["size"] = [""] * len(csv_file["file"])  # adding an empty column for wav file size
    csv_file["sentence"] = [""] * len(csv_file["file"])  # adding an empty column for transcription
    pd.DataFrame(csv_file).to_csv(args.out_csv, sep="|", index=False)
    files = pd.read_csv(args.out_csv, sep="|", dtype="string")
    print(f"Transcribing text that will be added to {args.out_csv}")
    for i in range(len(files)):
        file_name = os.path.join(args.audio_folder, files.iat[i, 0])
        file_size = os.path.getsize(file_name)
        files.iat[i, 2] = str(file_size)
        fin = wave.open(file_name, 'rb')
        frames = fin.readframes(fin.getnframes())
        audio = np.frombuffer(frames, np.int16)
        text = ds.stt(audio)
        print(text)
        files.iat[i, 3] = text

    files.to_csv(args.out_csv, sep="|", index=False)
    print(f"{args.out_csv} has been written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Extracts audio, and splits it into smaller files""")
    parser.add_argument("--input", help="big video file", required=True)
    parser.add_argument("--audio_folder", help="folder that will contain smaller the audio files", required=True)
    parser.add_argument("--out_csv", help="name of output csv file, will contain names of each chopped wav file and will be pupulated with their transcript", required=True)
    parser.add_argument("--splice_video", help="Splice video along with audio", action='store_true')
    parser.add_argument("--wav_args", help="list of arguments of the wav created files as string", default="-acodec pcm_s16le -ac 1 -ar 16000")
    parser.add_argument("--max_duration", help="maximum duration (in seconds) a clip can last", default=7, type=int)
    parser.add_argument("--min_duration", help="minimum duration (in seconds) a clip can last", default=2, type=int)
    parser.add_argument("--remove_bad_segments", action="store_true",
        help="use this to automatically remove sentences not spoken by a speaker of interest (must be specified using the 'speaker_segment' argument")
    parser.add_argument("--speaker_segment", nargs=2, type=float, 
        help="start and end time of a sample spoken by a speaker (seconds)")
    args = parser.parse_args()
    print("SNAKES!")
    main(args)
    