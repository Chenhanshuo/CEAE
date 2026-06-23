# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:30:56 2025

@author: 17471
"""

import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def generate_pcma_dataset():
    def load_rml_data(filename):
        Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')
        all_mods = list(set([key[0] for key in Xd.keys()]))
        mods = sorted(all_mods)

        snrs = np.arange(-20, 19, 2).astype(float)
        X, lbl = [], []
        for mod in mods:
            for snr in snrs:
                key = (mod, snr)
                if key in Xd:
                    signals = Xd[key]
                    X.append(signals)
                    for _ in range(signals.shape[0]):
                        lbl.append((mod, snr))
                else:
                    print(f"警告: 数据集缺少组合 {key}")
        return np.vstack(X), lbl

    X_origin, lbl_origin = load_rml_data("RML2016.10a_dict.pkl")

    Xd_pcma = defaultdict(list)

    mods = sorted(list(set([lbl[0] for lbl in lbl_origin])))
    snrs = sorted(list(set([lbl[1] for lbl in lbl_origin])))

    for mod in tqdm(mods, desc='Modulations'):
        for snr in tqdm(snrs, desc='SNRs', leave=False):
            mask = np.array([(lbl[0] == mod) & (lbl[1] == snr) for lbl in lbl_origin])
            signals = X_origin[mask]

            if len(signals) == 0:
                print(f"跳过 {mod}-{snr}dB（无原始信号）")
                continue


            for _ in tqdm(range(1000), desc='Samples', leave=False):
                idx = np.random.choice(len(signals), 2, replace=True)
                s1 = signals[idx[0]]
                s2 = signals[idx[1]]
                combined = s1 + s2

                # 添加噪声（保持原始逻辑）
                # signal_power = np.mean(np.abs(combined)**2)
                # noise_power = signal_power * 10**(-snr/10)
                # noise = np.sqrt(noise_power/2) * (
                #     np.random.randn(*combined.shape) +
                #     1j*np.random.randn(*combined.shape)
                # )
                pcma_signal = combined  # + noise

                # 使用元组作为字典键
                key = (mod, snr)
                Xd_pcma[key].append(pcma_signal)

    for key in Xd_pcma:
        Xd_pcma[key] = np.array(Xd_pcma[key])
        print(f"生成组合 {key}: {len(Xd_pcma[key])} 个样本")

    with open('HSDMRD_dataset.pkl', 'wb') as f:
        pickle.dump(dict(Xd_pcma), f, protocol=4)

    return Xd_pcma


if __name__ == '__main__':
    generate_pcma_dataset()