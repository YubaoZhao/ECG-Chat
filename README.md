# ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis

This is a repository for reproducing the paper **ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis** [[Paper](https://arxiv.org/abs/2408.08849)] 

## Usage
### Prepare Datasets
We use 5 public datasets in our model, they can be downloaded from:
* [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
* [Champan-Shaoxing-Ningbo (CSD)](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
* [Shandong Provincial Hospital (SPH)](https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802/1)
* [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
* [CPSC2018](http://2018.icbeb.org/Challenge.html)

We provided the preprocessing code of these datasets, including extracting waveform data, converting to WFDB format, etc. Take the SPH dataset as an example:
```shell
python ./data/preprocess/preprocess_sph.py --data-dir /path/to/sph
```
You can also copy the .csv files in `data/` to your datasets folders. But you still need to convert the data format in SPH and CPSC2018 to WFDB format. 

The translated version of PTB-XL dataset is got from [Fairseq-signals](https://github.com/Jwoo5/fairseq-signals).

The ECG-Instruct datasets of ECG-Chat are provided in `llava/playground/data/`. `ecg_instruct_45k.json` is the combination of `diagnosis.json` and `conversation.json`. We also shared our prompts to build this two datasets in `llava/playground/data/prompts/`.

Due to the large size, the files `new_record_list.csv` in MIMIC-IV-ECG and pretraining dataset `pretrain_mimic.json` in our project can be downloaded [here](https://www.dropbox.com/scl/fo/ccq5dxmdgg4shf02yjn8c/ANOQ1Hzj4KwHqa1b9r80uzc?rlkey=teysp3v6hg6o9uko2i4zbbjpn&st=exu3i9oo&dl=0).
### Train the Models
To train and evaluate the ECG CoCa model, please use the scripts in `open_clip/`.

To pretraining and fine-tuning the ECG-Chat model, please use the scripts in `llava/`.

The codes for report generation evaluation and RAG are coming soon.

The ECG data augmentation methods implementation comes from [torch_ecg](https://github.com/DeepPSP/torch_ecg). We also used the CKEPE prompt proposed in [MERL](https://github.com/cheliu-computation/MERL-ICML2024) to evaluate the zero-shot classification ability of our model.

## Citation
If you think that our work is useful to your research, please cite using this BibTeX:
```bibtex
@misc{zhao2024ecgchatlargeecglanguagemodel,
      title={ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis}, 
      author={Yubao Zhao and Tian Zhang and Xu Wang and Puyu Han and Tong Chen and Linlin Huang and Youzhu Jin and Jiaju Kang},
      year={2024},
      eprint={2408.08849},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2408.08849}, 
}
```
If you have questions about this repo, please submit an issue or contact [yubaozhao01@gmail.com](mailto:yubaozhao01@gmail.com).

## Acknowledgement
- [OpenCLIP](https://github.com/mlfoundations/open_clip): the codebase we used to build our ECG CoCa model.
- [LLaVA](https://github.com/haotian-liu/LLaVA): we used the code and architecture of LLaVA to build our ECG-Chat model.