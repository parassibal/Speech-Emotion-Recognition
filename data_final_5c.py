import csv
import librosa
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

phase = ['train', 'val', 'test']

for p in phase:
    paths = []
    texts = []
    emos = []
    emo_one = []
    emotions = {'Neutral': 0, 'Anger': 1, 'Frustration': 2, # 'Excited': 5,
                'Sadness': 3, 'Happiness': 4}
    with open(f'./{p}.csv') as f:
        freader = csv.reader(f)
        t = 0
        for i in freader:
            if t >= 1:
                if i[4] == 'Excited':
                    i[4] = 'Happiness'
                if i[4] in emotions:
                    paths.append('./' + i[3])
                    texts.append(i[1])
                    emos.append(eval(i[2]))
                    emo_one.append(i[4])
            t += 1

    # emo_label = np.zeros((len(paths), len(emotions)))
    emo_label = []
    for i, v in enumerate(emo_one):
        emo_label.append(int(emotions[v]))
    emo_label = np.array(emo_label)
    print(len(emo_label))
    np.save('./IEMOCAP_data/data_5c/emo_5label_'+p+'.npy', emo_label)

    maxLen = 0
    for i in texts:
        temp = i.count(' ') + 1
        if temp > maxLen:
            maxLen = temp
    ids = []
    masks = []
    for i in texts:
        encoded = tokenizer.encode_plus(i, add_special_tokens=True, max_length=maxLen, pad_to_max_length=True,
            return_attention_mask=True)
        ids.append(encoded['input_ids'])
        masks.append(encoded['attention_mask'])

    np.save('./IEMOCAP_data/data_5c/ids_5c_'+p+'.npy', np.array(ids))
    np.save('./IEMOCAP_data/data_5c/masks_5c_'+p+'.npy', np.array(masks))

    mel_allpath = []
    mel_db_allpath = []
    for i in range(len(paths)):
        if i % 100 == 0:
            print(f"{i}/{len(paths)}")
        data_all, sampling_rate = librosa.load(paths[i])
        mel_spectrogram_all = librosa.feature.melspectrogram(y=data_all, sr=sampling_rate, hop_length=512, n_fft=2048,
                                                             n_mels=256, fmax=sampling_rate / 2)
        mel_spect_all = np.abs(mel_spectrogram_all)
        mel_spectrogram_db_all = librosa.power_to_db(mel_spect_all, ref=np.max)
        mel_allpath.append(mel_spectrogram_all)
        mel_db_allpath.append(mel_spectrogram_db_all)

    print(mel_spectrogram_db_all.shape)

    for i, v in enumerate(mel_allpath):
        np.save(f'./IEMOCAP_data/data_5c/spec_5c_{p}/{str(i).zfill(4)}_{emo_one[i]}.npy', v)

    for i, v in enumerate(mel_db_allpath):
        np.save(f'./IEMOCAP_data/data_5c/db_spec_5c_{p}/{str(i).zfill(4)}_{emo_one[i]}.npy', v)

