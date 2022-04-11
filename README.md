# DistilProtBert
DistilProtBert implementation, a distilled version of ProtBert model.

Model available at https://huggingface.co/yarongef/DistilProtBert.

For secondary structure predicion (Q3) task:
  1. Please follow https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTune-SS3.ipynb
  2. Change the model name to: 'yarongef/DistilProtBert'

For membrane vs water soluble (Q2) task:
  1. Please follow https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/ProtBert-BFD-FineTuning-MS.ipynb
  2. Change the model name to: 'yarongef/DistilProtBert'

For real versus shuffled proteins classification task:
  1. Extract DistilProtBert features as shown in feature extraction notebook.
  2. Run inference notebook.
