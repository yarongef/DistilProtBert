# DistilProtBert
Official DistilProtBert implementation, a distilled version of ProtBert-UniRef100 model.

Model available at Hugging Face model [page](https://huggingface.co/yarongef/DistilProtBert).

Pretraining dataset: [UniRef50](https://www.uniprot.org/downloads)

Datasets for real versus shuffled protein sequences classification task are available at:
 - [Singlets](https://huggingface.co/datasets/yarongef/human_proteome_singlets)
 - [Doublets](https://huggingface.co/datasets/yarongef/human_proteome_doublets)
 - [Triplets](https://huggingface.co/datasets/yarongef/human_proteome_triplets)

This repository is based on ProtBert-UniRef100 implementation from [ProtTrans repo](https://github.com/agemagician/ProtTrans).

# **Evaluation tasks**

### Secondary structure predicion (Q3)
Please follow [SS3 fine tuning notebook](https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTune-SS3.ipynb) and change *model_init* first argument to 'yarongef/DistilProtBert'.
      
    def model_init():
        return AutoModelForTokenClassification.from_pretrained('yarongef/DistilProtBert',
                                                               num_labels=len(unique_tags),
                                                               id2label=id2tag,
                                                               label2id=tag2id,
                                                               gradient_checkpointing=False)

#### Results
|    **MODEL**   | **CASP12** | **TS115** | **CB513** |
|:--------------:|:----------:|:---------:|:---------:|
|    ProtBert-UniRef100    |    0.75    |    0.83   |    0.81   |
| DistilProtBert |    0.72    |    0.81   |    0.79   |

---------------------------------

### Membrane vs water soluble (Q2)
Please follow [MS fine tuning notebook](https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTuning-MS.ipynb) and change *model_init* first argument to 'yarongef/DistilProtBert'

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained('yarongef/DistilProtBert')
 
#### Results
|    **MODEL**   | **DeepLoc** |
|:--------------:|:----------:|
|    ProtBert-UniRef100    |    0.89    |  
| DistilProtBert |    0.86    | 

----------------------------------

### Real versus shuffled proteins classification

  1. Download the relevant k-let classification model from [here](https://www.dropbox.com/sh/221eiziowdg5m5e/AADh_f8DO_Tn9r56S1QbpyaHa?dl=0)
  2. Extract the relevant k-let test set features via [feature extraction notebook](https://github.com/yarongef/DistilProtBert/blob/main/Feature%20Extraction/Feature%20Extraction.ipynb)
  3. Run [inference notebook](https://github.com/yarongef/DistilProtBert/blob/main/Inference/Inference.ipynb)

#### Singlets
ddd

#### Doublets
ddd

#### Triplets
ddd
