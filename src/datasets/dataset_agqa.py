import torch
import random
import numpy as np
import copy
import io
import lmdb
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from torch.utils.data import Dataset
import pdb


class AGQADataset(Dataset):
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
    # open_ended_qa_names = ["frameqa", "msrvtt_qa"]

    def __init__(self, datalist, tokenizer, vid_lmdb_dir,
                 max_txt_len=20, ans2label=None,
                 is_train=True, return_label=True):
        self.is_train = is_train
        self.return_label = return_label
        self.ans2label = ans2label["answer_token_to_idx"]
        self.num_labels = len(ans2label)
        self.label2ans = {v: k for k, v in ans2label["answer_token_to_idx"].items()}
        self.qid2data = {d["question_id"]: d for group in datalist for d in group[1]}

        self.datalist = datalist
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.env = lmdb.open(
            vid_lmdb_dir, readonly=True,
            create=False)  # readahead=not _check_distributed()
        self.txn = self.env.begin(buffers=True)
        self.frame_encoding_dict = torch.load(f'{vid_lmdb_dir}/frame_encoding_dict.pytorch')
        self.idx_encoding_dict = torch.load(f'{vid_lmdb_dir}/idx_encoding_dict.pytorch')

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
        # return embedding_dict, self.idx_encoding_dict[video_id]

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 3
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            embedding_dict, frame_encoding = self._load_video(vid_id)  # dump dictionary
            # vid_frm_array = torch.zeros_like(vid_frm_array)
            # Select a random video if the current video was not able to access.
            if embedding_dict is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
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

        if self.return_label:
            try:
                example["label"] = self.ans2label[example["label"]]
            except:
                example["label"] = 0
        else:
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
        qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
        qid2pred_ans = {k: self.label2ans[v] for k, v in qid2pred_ans.items()}

        for qid, pred_ans in qid2pred_ans.items():
            preds.append(pred_ans)

            gt_data = self.qid2data[qid]
            gt_ans = gt_data["answer"]
            gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        metrics = dict()
        # preds and gts are array of strings
        metrics["overall_acc"] = float(np.mean(preds == gts))
        return metrics

    def evaluate_tgif_qa_all_metric(self, results):
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
        question_types = {'global': [], 'semantic': [], 'structural': [], 'ans_type': []}
        question_type2idx = dict(
            reasoning={"obj-rel": 0, "rel-act": 1, "obj-act": 2, "superlative": 3, "sequencing": 4, "exists": 5, "duration-comparison": 6, "action-recognition": 7},
            semantic={"object": 0, "relation": 1, "action": 2},
            structural={"query": 0, "compare": 1, "choose": 2, "logic": 3, "verify": 4},
            ans_type={"binary": 0, "open": 1}
        )

        qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
        qid2pred_ans = {k: self.label2ans[v] for k, v in qid2pred_ans.items()}

        for qid, pred_ans in qid2pred_ans.items():
            preds.append(pred_ans)

            gt_data = self.qid2data[qid]
            gt_ans = gt_data["answer"]
            for k, v in question_type2idx.items():
                if k != 'reasoning':
                    question_types[k].append(v[gt_data[k]])
                else:
                    question_types['global'].append([v[x] for x in gt_data['global']])
            gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        metrics = dict()
        # preds and gts are array of strings
        print('pred: ', preds)
        print('gts: ', gts)
        metrics["overall_acc"] = float(np.mean(preds == gts))
        for metric, question_type in question_types.items():
            question_type = np.array(question_type)
            ratios = dict()
            if metric != 'global':
                for q_type, q_type_idx in question_type2idx[metric].items():
                    q_type_mask = question_type == q_type_idx
                    q_type_corrects = (
                            preds[q_type_mask] == gts[q_type_mask])
                    metrics[f"{q_type}_acc"] = float(
                        np.mean(q_type_corrects)) if len(q_type_corrects) != 0 else 0
                    ratios[f"{q_type}_ratio"] = [
                        1. * len(q_type_corrects) / len(question_type),
                        len(q_type_corrects)]
                metrics[f"{metric}_ratios"] = ratios
            else:
                for q_type, q_type_idx in question_type2idx['reasoning'].items():
                    q_type_mask = [q_type_idx in x for x in question_type]
                    # q_type_mask = [np.any(x) for x in q_type_mask]
                    q_type_corrects = (
                            preds[q_type_mask] == gts[q_type_mask])
                    metrics[f"{q_type}_acc"] = float(
                        np.mean(q_type_corrects)) if len(q_type_corrects) != 0 else 0
                    ratios[f"{q_type}_ratio"] = [
                        1. * len(q_type_corrects) / len(question_type),
                        len(q_type_corrects)]
                # pdb.set_trace()
                metrics[f"{metric}_ratios"] = ratios
        return metrics


class VideoQACollator(object):
    def __init__(self, tokenizer, max_length=20):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate_batch(self, batch):
        # v_collate = default_collate
        visual_inputs = [d["vid"] for d in batch]  # <list> (B, dict)
        frame_info = [d["frame_info"] for d in batch]  # <list> (B, list)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.

        text_str_list = [d["q_str"] for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

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
