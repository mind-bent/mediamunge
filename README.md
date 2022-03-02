# mediamunge
Processing media files for easier referencing and note gathering
Point transcribe.py to media, it will convert it to wav, search for silent points, and chop it up while
saving everything to a csv file so it can be evaluated or annotated more.
Use annotate.py as a tool to listen to the wav files and fix anything that has been transcribed incorrectly.

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
