import collections
import json
import logging
import math
from io import open
import args

from pytorch_pretrained_bert.tokenization import BertTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

'''
从结构体中分离出: domain context qas 它们在序列上一一对应
context: 案例内容
qas: 包含了五个小问题
domain: 案件类型
'''

def read_squad_examples(input_file, is_training=True, version_2_with_negative=True):
    """
    :param input_file: 待读取文件训练集路径
    :param is_training: 是否为训练集
    :return:
    """
    with open(input_file, "r", encoding='utf-8') as reader:

        # dataset: 数据集中所有案例
        dataset = json.load(reader)["data"]
        print(len(dataset)) #2000

        examples = []

        # 分析提取dataset中的每个案例
        for item in dataset:
            for paragraph in item['paragraphs']:
                text = paragraph['context']
                qas = paragraph['qas']


                doc_tokens = []
                char_to_word_offset = []

                # 方便之后抽出答案
                for word in text:
                    doc_tokens.append(word)
                    char_to_word_offset.append(len(doc_tokens) - 1)


                # qas是一个案例下的所有问题，qa是所有问题中的一个问题
                for qa in qas:
                    qa_id = qa['id']
                    question = qa['question']

                    # 参数自定义
                    start_position = None
                    end_position = None
                    answer = None
                    is_impossible = False
                    is_yes= False
                    is_no = False

                    # 假如读入的数据集属于训练数据
                    if is_training:
                        if version_2_with_negative:
                            if qa['is_impossible'] == 'false':
                                is_impossible = False
                            else:
                                is_impossible = True

                        # for training, each question should have exactly 1 answer
                        if (len(qa['answers']) != 1) and (not is_impossible):
                            continue

                        if not is_impossible:
                            ans = qa['answers'][0]
                            answer = ans['text']

                            answer_start = ans['answer_start']
                            answer_length = len(answer)
                            start_position = char_to_word_offset[answer_start]
                            end_position = char_to_word_offset[answer_start + answer_length - 1]
                            real_answer = "".join(doc_tokens[start_position:end_position + 1])

                            clean_answer = " ".join(whitespace_tokenize(answer))
                            print(start_position, end_position, real_answer, clean_answer)

                            # 如果抽取出来的答案 与 材料提供的数据 不能匹配
                            if real_answer.find(clean_answer) == -1:
                                if (clean_answer == 'YES'):
                                    is_yes = True
                                    answer = 'YES'
                                    start_position = -1
                                    end_position = -1
                                elif clean_answer == 'NO':
                                    is_no = True
                                    answer = 'NO'
                                    start_position = -1
                                    end_position = -1
                                else:
                                    logger.warning("could not find answer: '%s' vs. '%s'", real_answer, clean_answer)
                                    continue
                        else:
                            start_position = -1
                            end_position = -1
                            answer = ""

                    # if training
                # for qa in qas
            # for paragraph
        # for item







if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    #生成训练数据
    examples = read_squad_examples(input_file=args.train_set, version_2_with_negative=True)