# Models

Repository which serves to train the CANINE models.

The main two files are:

- `multiclass.py` - trains the multiclass CANINE model on the OpenLID dataset.
- `multilabel.py` - trains the multilabel CANINE models, both the base one and the one with negative sampling depending whether the `--negative-sampling` flag is provided.

The tokenizers for the models are located in the `trainer_output` directory.

Evaluations run by these two scripts:

- `flores_evaluation.py` - outputs the evaluations on the FLORES-200 dataset. The specific parameters are written in the comments in the file.
- `our_evaluation.py` - outputs evaluations on CHALIS. The specific parameters are written in the comments inside the source code.

Other notes:

- list of FLORES labels: [link](https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200)
- pycld3 package requires `protoc` - [The protocol buffers compiler](https://github.com/protocolbuffers/protobuf)

## Install

- run `pip install -r requirements.txt` and `pip install -r requirements-pytorch.txt`, then finally `./setup.sh`
