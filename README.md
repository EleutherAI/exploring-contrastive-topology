# exploring-contrastive-topology

Before running a Google Colab, run the following:

```
!git clone https://github.com/EleutherAI/exploring-contrastive-topology
%cd ./exploring-contrastive-topology
!sed -i "s/pandas==.*/pandas>=1.3/" requirements.txt
%pip install -r requirements.txt
!git lfs install
!git clone https://huggingface.co/datasets/cat-state/clip-embeddings
%cd ./clip-embeddings
!git lfs pull
%pip install -r requirements.txt
%cd ..
```
