# DistilProtBert
Official DistilProtBert implementation, a distilled version of ProtBert-UniRef100 model.

Model available at Hugging Face [page](https://huggingface.co/yarongef/DistilProtBert).

Pretraining dataset: [UniRef50](https://www.ebi.ac.uk/uniprot/download-center)

This repository is based on ProtBert-UniRef100 implementation from [ProtTrans repo](https://github.com/agemagician/ProtTrans).

**Evaluation tasks**
========================
Secondary structure predicion (Q3):
  1. Please follow [SS3 fine tuning notebook](https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTune-SS3.ipynb) 
  2. Update stage 11 to:
      

    def model_init():
        return AutoModelForTokenClassification.from_pretrained('yarongef/DistilProtBert',
                                                               num_labels=len(unique_tags),
                                                               id2label=id2tag,
                                                               label2id=tag2id,
                                                               gradient_checkpointing=False)



Results:
|    **MODEL**   | **CASP12** | **TS115** | **CB513** |
|:--------------:|:----------:|:---------:|:---------:|
|    ProtBert-UniRef100    |    0.75    |    0.83   |    0.81   |
| DistilProtBert |    0.72    |    0.81   |    0.79   |

---------------------------------

Membrane vs water soluble (Q2) task:
  1. Please follow [MS fine tuning notebook](https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTuning-MS.ipynb)
  2. Set *model_name* to 'yarongef/DistilProtBert'
 
Results:
|    **MODEL**   | **DeepLoc** |
|:--------------:|:----------:|
|    ProtBert-UniRef100    |    0.89    |  
| DistilProtBert |    0.86    | 

----------------------------------

Real versus shuffled proteins classification task:
  1. Download singlet/doublet/triplet classification models from [here](https://www.dropbox.com/sh/221eiziowdg5m5e/AADh_f8DO_Tn9r56S1QbpyaHa?dl=0)
  1. Extract test set features via [feature extraction notebook](https://github.com/yarongef/DistilProtBert/blob/main/Real%20vs.%20Shuffled%20Classification%20task/Feature%20Extraction.ipynb)
  2. Run [inference notebook](https://github.com/yarongef/DistilProtBert/blob/main/Real%20vs.%20Shuffled%20Classification%20task/Inference.ipynb)
