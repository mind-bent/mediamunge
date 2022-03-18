import os
import pandas as pd
import argparse
import threading
from tkinter import *
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Label
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime


class PlayAudioSample(threading.Thread):
    """plays the sound corresponding to an audio sample in its own thread"""
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name
    
    def run(self):
        sound = AudioSegment.from_wav(self.file_name)
        play(sound)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Manually completes the transcriptions of audio files, with some help.""")
    parser.add_argument("--audio_folder", help="Folder that contains the audio files.", required=True)
    parser.add_argument("--csv", help="Name of the csv file that is to be filled with the files transcriptions.", required=True)
    parser.add_argument("--ttc", help="Name of the converted ttc to csv file (from transcribe.py) that can be used to help with transcription.")
    parser.add_argument("--jump", help="Junp to this row in the CSV file.", default=0, type=int)
    args = parser.parse_args()
    
    files = pd.read_csv(args.csv, sep="|", dtype = {
        'file': "string",
        'timestamps': "string",
        'size': "string",
        'sentence': "string"
    })

    if args.ttc:
        ttc_files = pd.read_csv(args.ttc, sep="|", dtype = {
            'file': "string",
            'timestamps': "string",
            'accuracy': "string",
            'sentence': "string"
        })

    offsets_deleted_sentences = []
    window = Tk()
    window.title(f"Transcription of {args.csv}")
    WIN_SIZE = 1300
    window.geometry(f'{WIN_SIZE}x500')
    
    instructions = Label(
            window,
            text="Audio gets played automatically.\nTranscribe in the topmost text area.\nctrl-n for next sentence, ctrl-p for previous, ctrl-m to match, ctrl-u to undo, ctrl-d to delete, ctrl-r to repeat playback"
    )
    instructions.grid(row=0, columnspan=4)
    
    transcription = scrolledtext.ScrolledText(window, width=130, height=4)
    transcription.grid(row=1, columnspan=4, pady=30)

    if args.ttc:
        percent = Label(window, width=50, height=2)
        percent.grid(row=2, columnspan=4, pady=30)

    ttc_scription = scrolledtext.ScrolledText(window, width=130, height=3)
    ttc_scription.grid(row=3, columnspan=4, pady=30)
    
    def prepare_next_turn(fwd):
        """Loads next file or ends the program"""
        global current_offset, audio_player
        if fwd:
            current_offset += 1
        else:
            current_offset -= 1
        progress_bar["value"] = current_offset
        if current_offset < len(files):
            transcription.delete("1.0", END)
            ttc_scription.delete("1.0", END)
            sent = files.sentence.iat[current_offset]
            if args.ttc:
                ttc_sent = ttc_files.sentence.iat[current_offset]
            if isinstance(sent, str) and sent != "":
                transcription.insert("1.0", sent)
            if args.ttc:
                if isinstance(ttc_sent, str) and ttc_sent != "":
                    ttc_scription.insert("1.0", ttc_sent)
                    percent.config(text = f"If subtitles were provided,\nthis is the accuracy from the matched subs below: {ttc_files.accuracy.iat[current_offset]}%")
            audio_player = PlayAudioSample(os.path.join(args.audio_folder, files.file.iat[current_offset])).start()
            transcription.focus()
        else:
            window.destroy()

    def press_undo():
        """Places the original text from the CSV back in the top text box"""
        global current_offset
        sent = files.sentence.iat[current_offset]
        transcription.delete("1.0", END)
        transcription.insert("1.0", sent)

    def press_match():
        """Takes matched TTC text in the bottom text box and swaps it with the CSV text on in the top text box"""
        ttc_text = ttc_scription.get("1.0", END).replace("\n", "")
        transcription.delete("1.0", END)
        transcription.insert("1.0", ttc_text)

    def press_previous():
        """Modifies csv with text content and prepares for next turn"""
        files.iat[current_offset, 3] = transcription.get("1.0", END).replace("\n", "")
        if args.ttc:
            ttc_files.iat[current_offset, 3] = ttc_scription.get("1.0", END).replace("\n", "")
        window.title(f"Current wav file is {files.iat[current_offset-1, 0].split('/')[-1]}")
        prepare_next_turn(False)

    def press_next():
        """Modifies csv with text content and prepares for next turn"""
        files.iat[current_offset, 3] = transcription.get("1.0", END).replace("\n", "")
        if args.ttc:
            ttc_files.iat[current_offset, 3] = ttc_scription.get("1.0", END).replace("\n", "")
        try:
            window.title(f"Current wav file is {files.iat[current_offset+1, 0].split('/')[-1]}")
        except Exception as e:
            print("THIS IS THE END")
        prepare_next_turn(True)
        
    def press_delete():
        """Adds current phrase offset to instance of deleted phrases and prepares for next turns"""
        offsets_deleted_sentences.append(current_offset)
        prepare_next_turn(True)
        
    def press_repeat():
        """Repeats the previous audio file"""
        PlayAudioSample(os.path.join(args.audio_folder, files.file.iat[current_offset])).start()
        
    button_match = Button(window, text="Match", command=press_match, bg="white")
    button_match.grid(row=4, column=1)
    button_undo = Button(window, text="Undo", command=press_undo, bg="yellow")
    button_undo.grid(row=4, column=2)
    button_delete = Button(window, text="Delete", command=press_delete, bg="red")
    button_delete.grid(row=5, column=0)
    button_repeat = Button(window, text="Repeat", command=press_repeat, bg="white")
    button_repeat.grid(row=5, column=1)
    button_prev = Button(window, text="Previous", command=press_previous, bg="orange")
    button_prev.grid(row=5, column=2)
    button_next = Button(window, text="Next", command=press_next, bg="green")
    button_next.grid(row=5, column=3)
    window.bind('<Control-d>', lambda _: press_delete())
    window.bind('<Control-r>', lambda _: press_repeat())
    window.bind('<Control-p>', lambda _: press_previous())
    window.bind('<Control-n>', lambda _: press_next())
    window.bind('<Control-m>', lambda _: press_match())
    window.bind('<Control-u>', lambda _: press_undo())
    
    progress_bar = ttk.Progressbar(window, style='blue.Horizontal.TProgressbar', length=WIN_SIZE, maximum=len(files))
    progress_bar.grid(row=6, columnspan=4)
    window.grid_rowconfigure(3, weight=1)  # so that progress bar is at the bottom
    
    current_offset = args.jump - 1  # will be incremented or decremented by prepare_next_turn
    prepare_next_turn(True)
    window.mainloop()
    
    # Deletes wav files to delete
    for i in offsets_deleted_sentences:
        file_name = os.path.join(args.audio_folder, files.file.iat[i])
        os.remove(file_name)
        print(f"{file_name} was deleted")
        
    index_to_keep = [i for i in range(len(files)) if i not in set(offsets_deleted_sentences)]
    files = files.iloc[index_to_keep]
    if args.ttc:
        indextokeep = [i for i in range(len(ttc_files)) if i not in set(offsets_deleted_sentences)]
        ttc_files = ttc_files.iloc[indextokeep]
    
    print("Saved modified csv file")
    try:
        print(f"LAST WAV FILE WAS: {files.iat[current_offset, 0].split('/')[-1]}")
        print(f"LAST LINE: {current_offset}")
    except Exception as e:
        print(f'LINE {len(files)-1} IS THE LAST SENTENCE')
        print(f'LINE {current_offset} IS THE LAST LINE?')

    files.to_csv(args.csv, sep="|", index=False)
    if args.ttc:
        ttc_files.to_csv(args.ttc, sep="|", index=False)

