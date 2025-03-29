## **CAA on LLM: GPT2-Pytorch**



## Quick Start

1. download GPT2 pre-trained model in Pytorch which huggingface/pytorch-pretrained-BERT already made! (Thanks for sharing! it's help my problem transferring tensorflow(ckpt) file to Pytorch Model!)
```shell
$ git clone https://github.com/GodricLee/CAAonLLM && cd CAAonLLM
# download huggingface's pytorch model 
$ curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
# setup requirements, if using mac os, then run additional setup as descibed below
$ pip install -r requirements.txt
```
2. You can fine tune the model like this.

```shell
$ python main.py --train train_data.txt
```

3. Now, You can run like this.

- Text from Book 1984, George Orwell

```shell
$ python main.py --text "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."
```

2. Also You can Quick Starting in [Google Colab](https://colab.research.google.com/github/graykode/gpt-2-Pytorch/blob/master/GPT2_Pytorch.ipynb)



## Option

- `--text` : sentence to begin with.
- `--quiet` : not print all of the extraneous stuff like the "================"
- `--nsamples` : number of sample sampled in batch when multinomial function use
- `--unconditional` : If true, unconditional generation.
- `--batch_size` : number of batch size
- `--length` : sentence length (< number of context)
- `--temperature`:  the thermodynamic temperature in distribution `(default 0.7)`
- `--top_k`  : Returns the top k largest elements of the given input tensor along a given dimension. `(default 40)`

See more detail option about `temperature` and `top_k` in [here](https://github.com/openai/gpt-2#gpt-2-samples)



## Dependencies

- Pytorch 0.41+
- regex 2017.4.5
- 

## Author

- GodricLee



## License

- OpenAi/GPT2 follow MIT license, huggingface/pytorch-pretrained-BERT is Apache license. 
- I follow MIT license with original GPT2 repository



## Acknowledgement

- Tae Hwan Jung(Jeff Jung) @graykode
- Email : [nlkey2022@gmail.com](mailto:nlkey2022@gmail.com)