# Topic-Dialog-Summ

Pytorch implementation of the AAAI-2021 paper: [Topic-Oriented Spoken Dialogue Summarization for Customer Service with Saliency-Aware Topic Modeling](https://arxiv.org/pdf/2012.07311).

The code is partially referred to https://github.com/nlpyang/PreSumm.

## Requirements

* Python 3.6 or higher
* torch==1.1.0
* pytorch-transformers==1.1.0
* torchtext==0.4.0
* rouge==0.3.2
* tensorboardX==2.1
* nltk==3.5
* gensim==3.8.3

## Environment

* Tesla V100 16GB GPU
* CUDA 10.2

## Data Format

Each json file is a data list that includes dialogue samples. The format of a dialogue sample is shown as follows:

```
{"session": [
    // Utterance
    {
     // Chinese characters
     "content": ["请", "问", "有", "什", "么", "可", "以", "帮", "您"],
     // Chinese Words
     "word": ["请问", "有", "什么", "可以", "帮", "您"],
     // Role info (Agent)
     "type": "客服"
    },

    {"content": ["我", "想", "退", "货"],
     "word": ["我", "想", "退货"],
     // Role info (Customer)
     "type": "客户"}, 
    
    ...
 ],
 "summary": ["客", "户", "来", "电", "要", "求", "退", "货", "。", ...]
}
```

## Usage

1. Download BERT checkpoints.

	The pretrained BERT checkpoints can be found at:
	
	* Chinese BERT: https://github.com/ymcui/Chinese-BERT-wwm
	* English BERT: https://github.com/google-research/bert

	Put BERT checkpoints into the directory **bert** like this:

	```
	--- bert
	  |
	  |--- chinese_bert
	     |
	     |--- config.json
	     |
	     |--- pytorch_model.bin
	     |
	     |--- vocab.txt
	```

2. Pre-train word2vec embeddings

    ```
    PYTHONPATH=. python ./src/train_emb.py -data_path json_data -emb_size 100 -emb_path pretrain_emb/word2vec
    ```

3. Data Processing

	```
	PYTHONPATH=. python ./src/preprocess.py -raw_path json_data -save_path bert_data -bert_dir bert/chinese_bert -log_file logs/preprocess.log -emb_path pretrain_emb/word2vec -tokenize -truncated -add_ex_label
	```

4. Pre-train the pipeline model (Ext + Abs)

	```
	PYTHONPATH=. python ./src/train.py -data_path bert_data/ali -bert_dir bert/chinese_bert -log_file logs/pipeline.topic.train.log -sep_optim -topic_model -split_noise -pretrain -model_path models/pipeline_topic
	```

5. Train the whole model with RL

    ```
    PYTHONPATH=. python ./src/train.py -data_path bert_data/ali -bert_dir bert/chinese_bert -log_file logs/rl.topic.train.log -model_path models/rl_topic -topic_model -split_noise -train_from models/pipeline_topic/model_step_80000.pt -train_from_ignore_optim -lr 0.00001 -save_checkpoint_steps 500 -train_steps 30000
    ```

6. Validate

	```
	PYTHONPATH=. python ./src/train.py -mode validate -data_path bert_data/ali -bert_dir bert/chinese_bert -log_file logs/rl.topic.val.log -alpha 0.95 -model_path models/rl_topic -topic_model -split_noise -result_path results/val
	```

7. Test

	```
	PYTHONPATH=. python ./src/train.py -mode test -data_path bert_data/ali -bert_dir bert/chinese_bert -test_from models/rl_topic/model_step_30000.pt -log_file logs/rl.topic.test.log -alpha 0.95 -topic_model -split_noise -result_path results/test
	```

## Data

Our dialogue summarization dataset is collected from [Alibaba customer service center](https://114.1688.com/kf/contact.html). All dialogues are incoming calls in Mandarin Chinese that take place between a customer and a service agent. For the security of private information from customers, we performed the data desensitization and converted words to IDs. As a result, the data cannot be directly used in our released codes and other pre-trained models like BERT, but the dataset still provides some statistical information.

The desensitized data is available at 
[Google Drive](https://drive.google.com/file/d/1X3-C9vTYfk43T5NIEvRsdRIJkN1RuG7b/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1AvkGnerKpQHUNbwkz9kO7A) (extract code: t6nx).
