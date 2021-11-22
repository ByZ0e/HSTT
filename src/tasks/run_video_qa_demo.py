import json
import pdb

import torch
import os
import time
import random
import math
import pandas as pd
from transformers import BertConfig, BertTokenizerFast

import sys, os
# sys.path.insert(0, '../')
# sys.path.append(os.path.abspath('..'))
syspath = sys.path
print(syspath)
from src.modeling.e2e_model import ClipBert
from src.modeling.modeling import ClipBertForSequenceClassification

from src.datasets.dataset_agqa import AGQADataset, VideoQACollator
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import mk_input_group
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import load_jsonl, load_json, save_json, get_rounded_percentage
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer

from tqdm import tqdm
from os.path import join
from easydict import EasyDict as edict
from apex import amp
from torch.utils.data.distributed import DistributedSampler
# import horovod.torch as hvd
# from src.utils.distributed import all_gather_list
from collections import defaultdict
from scripts.save_lmdb import sta_vid

question_type_keys = ['global', 'ans_type', 'semantic', 'structural']

# sampled_vids = sta_vid('scripts/balanced_test_frames.txt')


def mk_agqa_dataloader(raw_datalist, lmdb_dir, cfg, tokenizer, is_train=True, return_label=True):
    if not cfg.all_metric:
        questions, answers, video_names, question_ids = raw_datalist
        datalist = []
        for i, q in enumerate(questions):
            # video_names[i] == 'CESHR'
            if video_names[i] == 'V6ZO7' or video_names[i] == 'TX1HS' or video_names[i] == '2XXH8' or video_names[i] == 'CESHR':
                continue
            # if cfg.sampled:
            #     if video_names[i] not in sampled_vids:
            #         continue
            d = dict(
                question=q,
                vid_id=video_names[i],
                answer=answers[i],  # int or str
                question_id=question_ids[i]  # be careful, it is not unique across splits
            )
            datalist.append(d)
    else:
        datalist = raw_datalist
    LOGGER.info(f"datalist {len(datalist)}")

    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["vid_id"]].append(d)
    LOGGER.info(f"grouped {len(grouped)}")

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group=cfg.max_n_example_per_group if is_train else 1,  # force 1 in eval,
        is_train=is_train
    )
    LOGGER.info(f"group_datalist {len(group_datalist)}")

    if cfg.all_metric:
        cfg.ans2label_path = cfg.ans2label_path.replace('csv', 'txt')
    ans2label = load_json(cfg.ans2label_path)
    print('load ans2label from ', cfg.ans2label_path)
    dataset = AGQADataset(
        datalist=group_datalist,
        tokenizer=tokenizer,
        vid_lmdb_dir=lmdb_dir,
        ans2label=ans2label,
        max_txt_len=cfg.max_txt_len,
        is_train=is_train,
        return_label=return_label
    )
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    sampler = DistributedSampler(dataset, num_replicas=0, rank=0, shuffle=is_train)
    vqa_collator = VideoQACollator(tokenizer=tokenizer,
                                   max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vqa_collator.collate_batch)
    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")

    anno_path = cfg.train_datasets[0].txt
    csv_data = pd.read_csv(anno_path)
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    question_ids = list(csv_data['key'])

    split = int(cfg.data_ratio * len(questions))

    # make sure split doesnt split a video
    while (video_names[split - 1] == video_names[split]):
        split = split + 1

    train_questions = questions[:split]
    train_answers = answers[:split]
    train_video_names = video_names[:split]
    train_question_ids = question_ids[:split]

    val_questions = questions[split:]
    val_answers = answers[split:]
    val_video_names = video_names[split:]
    val_question_ids = question_ids[split:]

    LOGGER.info(f"Loaded data size {len(train_questions)}")

    # anno_txt_file = ('storage/video_db/test_balanced.txt')
    # with open(anno_txt_file) as f:
    #     anno_dict = json.load(f)
    anno_dict = {}

    train_loader = mk_agqa_dataloader(
        raw_datalist=[train_questions, train_answers, train_video_names, train_question_ids],
        lmdb_dir=cfg.train_datasets[0].vid,
        cfg=cfg, tokenizer=tokenizer, is_train=True
    )
    val_loader = mk_agqa_dataloader(
        raw_datalist=[val_questions, val_answers, val_video_names, val_question_ids],
        lmdb_dir=cfg.val_datasets[0].vid,
        cfg=cfg, tokenizer=tokenizer, is_train=False, return_label=False
    )
    train_loader = PrefetchLoader(train_loader)
    val_loader = PrefetchLoader(val_loader)
    return train_loader, val_loader


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    add_attr_list = [
        "num_labels", "classifier", "cls_hidden_scale",
        "loss_type",
    ]
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])

    transformer_model_cls = ClipBertForSequenceClassification

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")
    model = ClipBert(
        model_cfg, transformer_cls=transformer_model_cls)
    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    else:
        LOGGER.info(f"Loading cnn weights from {cfg.detectron2_weights_path}")
        LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
        model.load_separate_ckpt(
            bert_weights_path=cfg.bert_weights_path)
    # if cfg.freeze_cnn:
    #     model.freeze_cnn_backbone()
    model.to(device)

    LOGGER.info("Setup model done!")
    return model


@torch.no_grad()
def validate(model, val_loader, cfg, train_global_step, eval_score=True):
    """use eval_score=False when doing inference on test sets where answers are not available"""
    model.eval()

    loss = 0.
    n_ex = 0
    qa_results = []
    st = time.time()
    debug_step = 5
    pbar = tqdm(total=len(val_loader))
    for val_step, batch in enumerate(val_loader):
        # forward pass
        question_ids = batch["question_ids"]
        bsz = len(question_ids)
        # used to make visual feature copies
        del batch["question_ids"]
        # add visual part into the mini batch and perform inference
        mini_batch = dict()
        for k, v in batch.items():
            mini_batch[k] = v

        n_ex += len(question_ids)
        # multi-frame test, scores across frames of the same video will be pooled together
        pool_method = cfg.score_agg_func

        logits = []
        losses = []

        outputs = model(mini_batch)  # dict
        logits.append(outputs["logits"].cpu())
        _loss = outputs["loss"].sum().item() if isinstance(
            outputs["loss"], torch.Tensor) else 0
        losses.append(_loss)
        loss += (sum(losses))

        logits = torch.stack(logits)  # (num_frm, B, 5)
        if pool_method == "mean":
            logits = logits.mean(0)  # (B, 5)
        elif pool_method == "max":
            logits = logits.max(0)[0]  # (B, 5)
        elif pool_method == "lse":
            logits = logits.permute(1, 0, 2).contiguous()  # (B, num_frm, 5), pooling will be done in CE
            logits = torch.logsumexp(logits, dim=1)  # torch.exp alone might be too large and unstable
        else:
            raise ValueError(f"Invalid value for pool_method, "
                             f"got {pool_method}, expect one of [`mean`, `max`, `lse`]")

        if cfg.task in ["action", "transition", "frameqa", "msrvtt_qa"]:
            # cross entropy
            pred_labels = logits.max(dim=-1)[1].data.tolist()
        else:
            # mse
            preds = (logits + 0.5).long().clamp(min=1, max=10)
            pred_labels = preds.data.squeeze().tolist()
        for qid, pred_label in zip(question_ids, pred_labels):
            qa_results.append(dict(
                question_id=qid,
                answer=pred_label,
                data=val_loader.dataset.qid2data[qid]
            ))
        pbar.update(1)
        if cfg.debug and val_step >= debug_step:
            break

    if cfg.debug:
        LOGGER.info(qa_results[:10])

    n_ex_per_rank = all_gather_list(n_ex)
    loss = sum(all_gather_list(loss))
    n_ex = sum(all_gather_list(n_ex))


    # average loss for each example
    val_log = {f'valid/loss': float(loss / n_ex)}
    if eval_score:
        LOGGER.info(f"QA Task [{cfg.task}], "
                    f"{len(qa_results)} qa_results,"
                    f"3 examples here: {qa_results[:3]}")
        if cfg.all_metric:
            vqa_scores = val_loader.dataset.evaluate_tgif_qa_all_metric(qa_results)
        else:
            vqa_scores = val_loader.dataset.evaluate_tgif_qa(qa_results)
        # print(f"{hvd.rank()}: {vqa_scores}")

        # Gather scores
        scores_per_rank = all_gather_list(vqa_scores)
        gathered_scores = {}
        for metric in question_type_keys:
            if f"{metric}_ratios" in scores_per_rank[0]:
                gathered_ratios = {
                    k: [0, 0] for k, _ in scores_per_rank[0][f"{metric}_ratios"].items()}
                # Gather ratios
                for rank_id in range(len(n_ex_per_rank)):
                    current_ratios = scores_per_rank[rank_id][f"{metric}_ratios"]
                    for k, v in current_ratios.items():
                        gathered_ratios[k][1] += v[1]
                for k, v in gathered_ratios.items():
                    gathered_ratios[k][0] = get_rounded_percentage(
                        1. * v[1] / n_ex)
                gathered_scores[f"{metric}_ratios"] = gathered_ratios

        # FIXME: Gather scores become complicated due to np.mean and dict format.
        for scores_k, _ in vqa_scores.items():
            # Only consider the acc.
            if "ratio" in scores_k and scores_k != 'duration-comparison_acc':
                continue
            gathered_v = 0
            for rank_id, n in enumerate(n_ex_per_rank):
                curr_acc = scores_per_rank[rank_id][scores_k] * n
                # print(scores_k, ':', curr_acc)
                gathered_v += curr_acc
            gathered_v = gathered_v * 1. / n_ex

            if cfg.task in ["action", "transition", "frameqa", "msrvtt_qa"]:
                gathered_scores[scores_k] = get_rounded_percentage(
                    gathered_v)
            else:
                gathered_scores[scores_k] = round(gathered_v, 2)

        for k, v in gathered_scores.items():
            if "ratio" not in k:
                val_log[f'valid/{k}'] = v
        # pdb.set_trace()
    else:
        LOGGER.info("eval_score = False, no scores are calculated.")
        gathered_scores = 0

    TB_LOGGER.log_scalar_dict(val_log)
    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds."
                f"{gathered_scores}")

    model.train()
    return qa_results, gathered_scores


def start_inference(cfg):
    set_random_seed(cfg.seed)
    # n_gpu = hvd.size()
    # device = torch.device("cuda", hvd.local_rank())
    # torch.cuda.set_device(hvd.local_rank())
    # if hvd.rank() != 0:
    #     LOGGER.disabled = True

    inference_res_dir = join(
        cfg.output_dir,
        f"results_{cfg.test_datasets[0].name}/"
        f"step_{cfg.inference_model_step}_use_idx_demo_{cfg.inference_n_clips}_{cfg.score_agg_func}_{'txt' if cfg.inference_metric else 'csv'}"
    )

    os.makedirs(inference_res_dir, exist_ok=True)
    save_json(cfg, join(inference_res_dir, "raw_args.json"),
              save_pretty=True)

    # LOGGER.info("device: {} n_gpu: {}, rank: {}, "
    #             "16-bits training: {}".format(
    #                 device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    # overwrite cfg with stored_cfg,
    # but skip keys containing the keyword 'inference'
    stored_cfg_path = join(cfg.output_dir, "log/args.json")
    stored_cfg = edict(load_json(stored_cfg_path))
    for k, v in cfg.items():
        if k in stored_cfg and "inference" not in k:
            setattr(cfg, k, stored_cfg[k])

    # setup models
    cfg.model_config = join(cfg.output_dir, "log/model_config.json")
    e2e_weights_path = join(
        cfg.output_dir, f"ckpt/model_step_{cfg.inference_model_step}.pt")
    cfg.e2e_weights_path = e2e_weights_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = setup_model(cfg, device=device)
    model.eval()

    # FIXME separate scaling for each loss
    model = amp.initialize(
        model, enabled=cfg.fp16, opt_level='O2')

    global_step = 0
    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    cfg.data_ratio = 1.

    anno_path = cfg.inference_txt_db
    print('Loading data from', anno_path)

    # questions, answers, video_names, question_ids = [], [], [], []
    # with open(anno_path, 'r') as f:
    #     js = f.read()
    # anno_data = json.loads(js)
    # for question_id, anno in anno_data.items():
    #     questions.append(anno['question'])
    #     video_names.append(anno['video_id'])
    #     answers.append(anno['answer'])
    #     question_ids.append(question_id)

    # anno_txt_file = ('storage/txt_db/test_balanced.txt')
    # with open(anno_txt_file) as f:
    #     anno_dict = json.load(f)
    if cfg.do_inference:
        cfg.all_metric = cfg.inference_metric
    if cfg.all_metric:
        print('=======load txt=======')
        with open(anno_path, 'r') as f:
            js = f.read()
        anno_data = json.loads(js)
        raw_datalist = []
        for question_id, anno in anno_data.items():
            if 'V6ZO7' in question_id or 'TX1HS' in question_id or '2XXH8' in question_id or question_id == 'CESHR':
                continue
            d = dict(
                question=anno['question'],
                vid_id=anno['video_id'],
                answer=anno['answer'],  # int or str
                question_id=question_id  # be careful, it is not unique across splits
            )
            d.update({key: anno[key] for key in question_type_keys})
            raw_datalist.append(d)
        LOGGER.info(f"Loaded data size {len(raw_datalist)}")
    else:
        print('=======load csv=======')
        csv_data = pd.read_csv(anno_path)
        questions = list(csv_data['question'])
        answers = list(csv_data['answer'])
        video_names = list(csv_data['gif_name'])
        question_ids = list(csv_data['key'])
        LOGGER.info(f"Loaded data size {len(questions)}")
        raw_datalist = [questions, answers, video_names, question_ids]

    val_loader = mk_agqa_dataloader(
        raw_datalist=raw_datalist,
        lmdb_dir=cfg.inference_vid_db,
        cfg=cfg, tokenizer=tokenizer, is_train=False,
        return_label=False
    )
    val_loader = PrefetchLoader(val_loader)

    LOGGER.info(cfg)
    LOGGER.info("Starting inference...")
    LOGGER.info(f"  Batch size = {cfg.inference_batch_size}")

    LOGGER.info(f'Step {global_step}: start validation')
    qa_results, qa_scores = validate(
        model, val_loader, cfg, global_step,
        eval_score=True)  # cfg.inference_split == "val"

    save_json(cfg, join(inference_res_dir, "merged_args.json"),
              save_pretty=True)
    save_json(qa_scores, join(inference_res_dir, "scores.json"),
              save_pretty=True)

    # ###### Saving with Horovod ####################
    # dummy sync
    _ = None
    all_gather_list(_)
    # if n_gpu > 1:
    #     # with retrial, as azure blob fails occasionally.
    #     max_save_load_trial = 10
    #     save_trial = 0
    #     while save_trial < max_save_load_trial:
    #         try:
    #             LOGGER.info(f"Save results trial NO. {save_trial}")
    #             save_json(
    #                 qa_results,
    #                 join(inference_res_dir, f"results_rank{hvd.rank()}.json"))
    #             break
    #         except Exception as e:
    #             save_trial += 1

    # dummy sync
    _ = None
    all_gather_list(_)
    # join results
    save_json(
        qa_results,
        join(inference_res_dir, f"results_all.json"))
    LOGGER.info(f'all results written')


if __name__ == '__main__':
    input_cfg = shared_configs.get_video_qa_args()
    if input_cfg.do_inference:
        start_inference(input_cfg)
