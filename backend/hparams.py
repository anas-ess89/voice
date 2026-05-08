## Mel-filterbank
mel_window_length = 25      # ms
mel_window_step = 10        # ms
mel_n_channels = 40

## Audio
sampling_rate = 16000
partials_n_frames = 160     # 1600 ms

## Voice Activation Detection
vad_window_length = 30      # ms
vad_moving_average_width = 8
vad_max_silence_length = 6

## Audio volume normalization
audio_norm_target_dBFS = -30

## Model parameters (LSTM)
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

## Verification thresholds
verification_threshold = 0.75
high_confidence_threshold = 0.80

## Training
learning_rate = 0.001
batch_size = 32
num_epochs = 10