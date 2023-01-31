import torch
import random
import numpy as np
import pandas as pd
import copy
import io
import lmdb
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.datasets.data_utils import expand_forMC
from src.utils.load_save import LOGGER
from torch.utils.data import Dataset
import pdb


def load_templates(csv_dir):
    qa_templates = pd.read_csv(csv_dir+'QA_templates.csv',header=None)
    qa_template_dict = {}
    for i in range(len(qa_templates)):
        qtype,temp,qtemp,atemp=qa_templates.iloc[i][0],qa_templates.iloc[i][1],qa_templates.iloc[i][2],qa_templates.iloc[i][3]
        if qtype not in qa_template_dict:
            qa_template_dict[qtype]={}
        qa_template_dict[qtype][temp]={}
        qa_template_dict[qtype][temp]['question'],qa_template_dict[qtype][temp]['answer'] = qtemp,atemp
    return qa_template_dict


def get_qa_temp(template_id, qa_template_dict):
    qtype, temp = template_id.split('_')[0], template_id.split('_')[1]
    Question, Answer = qa_template_dict[qtype][temp]['question'], qa_template_dict[qtype][temp]['answer']
    return Question + ' ' + Answer


def parse_semantic_dic(QA_temp, words):
    fixed_words = [i for i in QA_temp.split(' ') if '[' not in i]
    blanks = [i for i in QA_temp.split(' ') if '[' in i]
    print(fixed_words, blanks)

    fixed_words_ids = [i for i in range(len(words)) if words[i] in fixed_words]
    blank_ids = [i for i in range(len(words)) if i not in fixed_words_ids]
    print(blank_ids)

    semantic_dict = {}
    if len(blanks) == len(blank_ids):
        for i, blank in enumerate(blanks):
            semantic_dict[blank] = [blank_ids[i]]
    else:
        phrase_ids = [idx for i, idx in enumerate(blank_ids) if i+1 != len(blank_ids) and idx+1 == blank_ids[i+1]]
        print(phrase_ids)

        idx = 0
        for blank in blanks:
            if blank_ids[idx] not in phrase_ids:
                semantic_dict[blank] = [blank_ids[idx]]
            else:
                semantic_dict[blank] = [blank_ids[idx], blank_ids[idx+1]]
                idx = idx + 1
            idx = idx + 1
    print(semantic_dict)
    return semantic_dict


class VideoQADataset(Dataset):
    """ This should work for both train and test (where labels are not available).
    task_type: str, one of [action, frameqa, transition]
        where action and transition are multiple-choice QA,
            frameqa is opened QA similar to VQA.
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    return_label: bool, whether return label in __getitem__
    random_sample_clips:
    """
    open_ended_qa_names = ["frameqa", "count", "msrvtt_qa", "msvd_qa"]

    def __init__(self, task_type, datalist, tokenizer, vid_lmdb_dir,
                 max_txt_len=20, ans2label=None,
                 is_train=True, return_label=True):
        self.task_type = task_type
        self.is_train = is_train
        self.return_label = return_label
        self.ans2label = ans2label
        # self.num_labels = len(ans2label)
        # self.label2ans = {v: k for k, v in ans2label.items()}
        self.qid2data = {d["question_id"]: d for group in datalist for d in group[1]}

        self.datalist = datalist
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.env = lmdb.open(
            vid_lmdb_dir, readonly=True,
            create=False)  # readahead=not _check_distributed()
        self.txn = self.env.begin(buffers=True)
        self.frame_encoding_dict = torch.load(f'{vid_lmdb_dir}/edge_encoding_dict.pytorch')


    def __len__(self):
        return len(self.datalist)

    def _load_video(self, video_id):
        """Load video.
        Apply transformation to the sampled frames.
        Returns:
            torch.float, in [0, 255], (n_frm=T, c, h, w)
        """
        dump = self.txn.get(video_id.encode('utf-8'))
        with io.BytesIO(dump) as reader:
            dump = np.load(reader, allow_pickle=True)
            embedding_dict = {'object': torch.tensor(dump['object']), 'relation': torch.tensor(dump['relation']),
                              'frame': torch.tensor(dump['frame']), 'action': torch.tensor(dump['action'])}

        return embedding_dict, self.frame_encoding_dict[video_id]

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 3
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            # vid_name = 'vid' + str(vid_id)
            vid_name = str(vid_id)
            # print(vid_name)
            embedding_dict, frame_encoding = self._load_video(vid_name)  # dump dictionary
            # vid_frm_array = torch.zeros_like(vid_frm_array)
            # Select a random video if the current video was not able to access.
            if embedding_dict is None:
                LOGGER.info(f"Failed to load examples with video: {vid_name}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            examples = [self._get_single_example(e) for e in examples]
            return dict(
                vid=embedding_dict,
                frame_info=frame_encoding,
                examples=examples,
                n_examples=len(examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data):
        example = dict(
            q_str=data["question"],
            question_id=data["question_id"],
            label=data["answer"]
        )
        if self.task_type in ["action", "transition"]:
            example["options_str_list"] = data["options"]
        elif self.task_type in self.open_ended_qa_names:
            if self.return_label:
                example["label"] = self.ans2label[example["label"]]
        if not self.return_label:
            example["label"] = None
        return example

    def evaluate_tgif_qa(self, results):
        """
        Args:
            results: list(dict),
              each dict is
                {
                    "question_id": int,
                    "answer": int or float, either answer_idx (int)
                }
        Returns:
            TGIF-QA score
        """
        preds = []
        gts = []
        # for frameQA
        answer_types = []
        answer_type2idx = dict(
            frameqa={"object": 0, "number": 1, "color": 2, "location": 3},
            msrvtt_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
            msvd_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])}
        )

        qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
        if self.task_type in self.open_ended_qa_names:  # convert ans_idx, int --> str
            qid2pred_ans = {k: self.label2ans[v] for k, v in qid2pred_ans.items()}

        for qid, pred_ans in qid2pred_ans.items():
            preds.append(pred_ans)

            gt_data = self.qid2data[qid]
            gt_ans = gt_data["answer"]
            if self.task_type in self.open_ended_qa_names:
                answer_types.append(answer_type2idx[self.task_type][gt_data["answer_type"]])
            gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        metrics = dict()
        # preds and gts are array of strings
        metrics["overall_acc"] = float(np.mean(preds == gts))
        if self.task_type in self.open_ended_qa_names:
            answer_types = np.array(answer_types)
            ratios = dict()
            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                answer_type_mask = answer_types == ans_type_idx
                answer_type_corrects = (
                        preds[answer_type_mask] == gts[answer_type_mask])
                metrics[f"{ans_type}_acc"] = float(
                    np.mean(answer_type_corrects)) if len(answer_type_corrects) != 0 else 0
                ratios[f"{ans_type}_ratio"] = [
                    1. * len(answer_type_corrects) / len(answer_types),
                    len(answer_type_corrects)]
            metrics["ratios"] = ratios
        return metrics


class VideoQACollator(object):
    def __init__(self, tokenizer, max_length=20, task_type='action', n_options=4, txt_lmdb_dir='storage/txt_db/star'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.n_options = n_options

    def collate_batch(self, batch):
        # pdb.set_trace()
        # v_collate = default_collate
        visual_inputs = [d["vid"] for d in batch]  # <list> (B, dict)
        frame_info = [d["frame_info"] for d in batch]  # <list> (B, list)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        # print(text_examples)
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        if self.task_type in ["action", "transition"]:
            text_str_list = flat_list_of_lists(
                [[d["q_str"] + " " + d["options_str_list"][i] for i in range(self.n_options)]
                 for d in text_examples]
            )  # (B * n_options, ) !!!
        else:
            text_str_list = [d["q_str"] for d in text_examples]  # (B, )

        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)
        # print(text_input_mask.shape)

        labels = default_collate([int(d["label"]) for d in text_examples]) \
            if text_examples[0]["label"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]
        return dict(
            visual_inputs=visual_inputs,  # (B, dict)
            frame_info=frame_info,  # (B, list)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,
            labels=labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
