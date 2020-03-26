import collections
import json
import logging
import math
from io import open
import args
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)


# 把example定义成一个对象了，而不是一个dict
class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For example without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qa_id,
                 question_text,
                 doc_tokens,
                 answer = None,
                 start_position = None,
                 end_position = None,
                 is_impossible = None,
                 is_yes = None,
                 is_no = None):
        self.qa_id = qa_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.answer = answer
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.is_yes = is_yes
        self.is_no = is_no

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        cur = ""
        cur += "qa_id: %s" % (self.qa_id)
        cur += ", question: %s" % (self.question_text)
        cur += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))

        if self.start_position:
            cur += ", start_position: %d" % (self.start_position)
        if self.end_position:
            cur += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            cur += ", is_impossible: %d" % (self.is_impossible)
        return cur


class InputFeatures(object):
    """A single set of features of data."""


'''
从结构体中分离出: domain context qas 它们在序列上一一对应
context: 案例内容
qas: 包含了五个小问题
domain: 案件类型
'''
# read_squad_example 负责从json中读取数据，并进行一些处理，但是处理后的结果并不能输入进BERT模型中
def read_squad_examples(input_file, is_training=True, version_2_with_negative=True):
    """
    :param input_file: 待读取文件训练集路径
    :param is_training: 是否为训练集
    :return:
    """
    with open(input_file, "r", encoding='utf-8') as reader:

        # dataset: 数据集中所有案例
        dataset = json.load(reader)["data"]
        #print(len(dataset)) #2000

        examples = []

        # 分析提取dataset中的每个案例
        for item in dataset:
            for paragraph in item['paragraphs']:
                content_text = paragraph['context']
                qas = paragraph['qas']


                doc_tokens = []
                char_to_word_offset = []

                # doc_token是
                for word in content_text:
                    doc_tokens.append(word)
                    char_to_word_offset.append(len(doc_tokens) - 1)


                # qas是一个案例下的所有问题，qa是所有问题中的一个问题
                for qa in qas:
                    qa_id = qa['id']
                    question_text = qa['question']

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
                    example = SquadExample(
                        qa_id=qa_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        answer=answer,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                        is_yes=is_yes,
                        is_no=is_no,
                    )
                    examples.append(example)
                    """
                    example's key: qa_id: ... , question: ... 
                    """
                # for qa in qas
            # for paragraph
        # for item
        return examples

"""
convert_example_to_features任务
1. input_id
2. input_mask
3. segment_id
4. label_id
"""
# 使用该函数将examples处理成能够输入到bert中的格式，主要是截断、padding和token转换为id等
def convert_example_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training=True):

    # unk_token 就是生词
    features, unk_tokens = [], {}

    for(ex_index, example) in enumerate(examples):

        #对句子a分词
        query_tokens = tokenizer.tokenize(example.question_text) # ['在', '原', '告', '处', '投', '保', '的', '人', '投', '了', '什', '么', '保', '险', '？']
        doc_tokens = example.doc_tokens
        all_doc_tokens = []

        for(i, token) in enumerate(doc_tokens):
            #
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_doc_tokens.append(sub_token)

            # for sub_token in sub_tokens:
            #     all_doc_tokens.append(sub_token)

            if "[UNK]" in token:
                if token in unk_tokens:
                    unk_tokens[token] += 1
                else:
                    unk_tokens[token] = 1

        start_position = None
        end_position = None

        # 训练集 & 不可回答任务
        if is_training and example.is_impossible:
            start_position = end_position = -1

        # 训练集 & 可回答任务
        if is_training and not example.is_impossible:
            if example.is_yes or example.is_no :
                start_position = end_position = -1
            else:
                start_position = example.start_position
                end_position = example.end_position

        # The -3 accounts for [CLS] [SEP] [SEP]
        max_tokens_for_doc = args.max_seq_length - len(query_tokens) - 3

        # 问题截断
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0: max_query_length]

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        '''
        doc_stride 滑动窗口长度
        由于某些法律文本长度过长，不考虑直接截断的可能。 如，我们规定最长不超过512，但实际上，有些文本长达900多字。
        因此需要设定一个滑动窗口，从文本的左端向右端滑动，移动的个数为doc_stride
        这样bert的输入则变成了，query+滑动窗口内部文字
        '''
        doc_spans = []
        start_offset = 0
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])

        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset #剩余长度
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        #print(len(doc_spans)) # 3
        #print(doc_spans) # [DocSpan(start=0, length=494), DocSpan(start=256, length=494), DocSpan(start=512, length=459)]

        for (doc_span_idx, doc_span) in enumerate(doc_spans):
            tokens = []
            segment_ids = []

            # initial
            tokens.append("[CLS]")
            segment_ids.append(0)

            # add query's token
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)

            # add segment tag
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length): # 相对长度
                split_token_position = doc_span.start + i
                tokens.append(all_doc_tokens[split_token_position])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            print(input_ids)
            input_mask = [1] * len(input_ids)

        break

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    #生成训练数据
    examples = read_squad_examples(input_file=args.train_set, version_2_with_negative=True)

    #特征转换
    features = convert_example_to_features(examples=examples, tokenizer=tokenizer, max_seq_length=args.max_seq_length, doc_stride=256, max_query_length=args.max_query_length)