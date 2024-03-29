<br/>
<h1 align="center">DistilProtBert</h1>
<br/>

Official DistilProtBert implementation, a distilled version of ProtBert-UniRef100 model.

Check out our paper [DistilProtBert: A distilled protein language model used to distinguish between real proteins and their randomly shuffled counterparts](https://doi.org/10.1093/bioinformatics/btac474) for more details.

Model is available at Hugging Face model [page](https://huggingface.co/yarongef/DistilProtBert).

Pretraining dataset: [UniRef50](https://github.com/yarongef/DistilProtBert/blob/main/Datasets/UniRef50.ipynb)

This repository is based on ProtBert-UniRef100 implementation from [ProtTrans](https://github.com/agemagician/ProtTrans) repository.

## **Model details**
|   **Model**    | **# of parameters** | **# of hidden layers** |            **Pretraining dataset**             | **# of proteins** | **Pretraining hardware** |
|:--------------:|:-------------------:|:----------------------:|:----------------------------------------------:|:-----------------:|:------------------------:|
|    ProtBert    |        420M         |           30           | [UniRef100](https://www.uniprot.org/downloads) |       216M        |      512 16GB TPUs       |
| DistilProtBert |        230M         |           15           | [UniRef50](https://www.uniprot.org/downloads)  |        43M        |     5 v100 32GB GPUs     |

## **Evaluation tasks**

### Secondary structure prediction (Q3)
Please follow [SS3 fine-tuning notebook](https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTune-SS3.ipynb) and change *model_init* first argument to 'yarongef/DistilProtBert'.
      
    def model_init():
        return AutoModelForTokenClassification.from_pretrained('yarongef/DistilProtBert',
                                                               num_labels=len(unique_tags),
                                                               id2label=id2tag,
                                                               label2id=tag2id,
                                                               gradient_checkpointing=False)

Datasets can be found at [ProtTrans](https://github.com/agemagician/ProtTrans) repository.

#### Results
|     **MODEL**      | **CASP12** | **TS115** | **CB513** |
|:------------------:|:----------:|:---------:|:---------:|
| ProtBert-UniRef100 |    0.75    |   0.83    |   0.81    |
|   DistilProtBert   |    0.72    |   0.81    |   0.79    |

---------------------------------

### Membrane vs water soluble (Q2)
Please follow [MS fine-tuning notebook](https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTuning-MS.ipynb) and change *model_init* first argument to 'yarongef/DistilProtBert'.

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained('yarongef/DistilProtBert')

Datasets can be found at [ProtTrans](https://github.com/agemagician/ProtTrans) repository.

#### Results
|     **MODEL**      | **DeepLoc** |
|:------------------:|:-----------:|
| ProtBert-UniRef100 |    0.89     |  
|   DistilProtBert   |    0.86     | 

----------------------------------

### Distinguish between proteins and their k-let shuffled versions 

[Notebook](https://github.com/yarongef/DistilProtBert/blob/main/Datasets/Human%20Proteome.ipynb) for creating the datasets.

Datasets used for training and test: [singlets](https://huggingface.co/datasets/yarongef/human_proteome_singlets), [doublets](https://huggingface.co/datasets/yarongef/human_proteome_doublets) and [triplets](https://huggingface.co/datasets/yarongef/human_proteome_triplets).

Training:
- Extract the relevant k-let <ins>__training set__</ins> features via [feature extraction notebook](https://github.com/yarongef/DistilProtBert/blob/main/Feature%20Extraction/Feature%20Extraction.ipynb)
- Run the following [training script](https://github.com/yarongef/DistilProtBert/blob/main/Train/real_vs_shuffled.py)

Inference:
- Download the relevant k-let classification model from [here](https://www.dropbox.com/sh/221eiziowdg5m5e/AADh_f8DO_Tn9r56S1QbpyaHa?dl=0)
- Extract the relevant k-let <ins>__test set__</ins> features via [feature extraction notebook](https://github.com/yarongef/DistilProtBert/blob/main/Feature%20Extraction/Feature%20Extraction.ipynb)
- Run [inference notebook](https://github.com/yarongef/DistilProtBert/blob/main/Inference/Inference.ipynb)

_Singlet_
|    **Model**   | **AUC** |
|:--------------:|:-------:|
|      LSTM      |   0.71  |
|    ProtBert    |   0.93  |
| DistilProtBert |   0.92  |

_Doublet_
|    **Model**   | **AUC** |
|:--------------:|:-------:|
|      LSTM      |   0.68  |
|    ProtBert    |   0.92  |
| DistilProtBert |   0.91  |

_Triplet_
|    **Model**   | **AUC** |
|:--------------:|:-------:|
|      LSTM      |   0.61  |
|    ProtBert    |   0.92  |
| DistilProtBert |   0.87  |

## **Contact**
[Create an issue](https://github.com/yarongef/DistilProtBert/issues) to report bugs,
propose new functions or ask for help.

## **License**
[MIT License](https://github.com/yarongef/DistilProtBert/blob/main/LICENSE)

## **Citation**
If you use this code or one of our pretrained models for your publication, please cite our paper:
```
@article {
	author = {Geffen, Yaron and Ofran, Yanay and Unger, Ron},
	title = {DistilProtBert: A distilled protein language model used to distinguish between real proteins and their randomly shuffled counterparts},
	year = {2022},
	doi = {10.1093/bioinformatics/btac474},
	URL = {https://doi.org/10.1093/bioinformatics/btac474},
	journal = {Bioinformatics}
}
```
