# mediamunge
Processing media files to prepare for DeepSpeech training, for easier referencing and note gathering.
Current media files it handles: mp4, mkv, webm, mp3

Point transcribe.py to a media file or directory.
1. Cleans up name (used for folder and file names) and converts media to a wav file (mono, 16k), placing it in the same folder as the media.
2. Uses three rounds (db levels and durations hard coded) to capture silence points.
3. Splits wav file (uses ffmpeg) at audio points using optional min & max durations (default is 2 & 7 seconds).
4. Saves split wav files to the media folder.
5. Optionally splits media file (uses ffmpeg) into segments based on captured silence points, and saves them to the media folder.
6. Transcribes each spliced wav file, using a pretrained DeepSpeech model.
7. Creates a CSV with the wav file path, time stamps, file size, and transcription.
8. Optionally takes subtitle files (ttc), does some cleanup and matches sentences with the transcription, providing accuracy percentages.
9. If subtitles were included, it creates another CSV with wav file path, time stamps, accuracy, and matched subtitle sentence.

Use annotate.py as a tool to listen to the wav files and fix anything that has been transcribed incorrectly.
If subs have been processed, the matched sentence will be listed with the accuracy percentage.

This started with looking at DeepSpeech and eventually finding this repo https://github.com/NatGr/annotate_audio
and swapping the GCP STT part with DeepSpeech in order to not have to subscribe/pay/use the cloud and instead learn STT
in a standalone environment.

PreTrained models for DeepSpeech can be pulled down from here:

https://github.com/mozilla/DeepSpeech/releases

This code currently points to:

deepspeech-0.9.3-models.pbmm

deepspeech-0.9.3-models.scorer

Some information on this can be referenced here:

https://lindevs.com/speech-to-text-using-deepspeech/

The idea here is to eventually get enough media to use for DeepSpeech training as explained here:

https://mozilla.github.io/deepspeech-playbook/DATA_FORMATTING.html

https://mozilla.github.io/deepspeech-playbook/DATA_FORMATTING.html#preparing-your-data-for-training

https://mozilla.github.io/deepspeech-playbook/TRAINING.html

https://github.com/mozilla/DeepSpeech/blob/master/doc/TRAINING.rst

https://medium.com/visionwizard/train-your-own-speech-recognition-model-in-5-simple-steps-512d5ac348a5

https://www.assemblyai.com/blog/deepspeech-for-dummies-a-tutorial-and-overview-part-1/

https://www.assemblyai.com/blog/python-project-lecture-summaries/

https://deepspeech.readthedocs.io/en/r0.9/TRAINING.html#training-a-model


Tune pre-trained model: 

https://pythonawesome.com/a-keras-implementation-of-speach-to-text-architectures/

https://keras.io/keras_tuner/

https://keras.io/examples/audio/ctc_asr/


This process uses ffmpeg for manipulating media.
ffmpeg references:

https://ottverse.com/trim-cut-video-using-start-endtime-reencoding-ffmpeg/
