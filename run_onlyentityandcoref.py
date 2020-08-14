# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """


import argparse
import glob
import logging
import os
import random
import copy

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
#, get_entities
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


from bert_onlyentityandcoref import BertForTokenClassification

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    #BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    #XLMRobertaConfig,
    #XLMRobertaForTokenClassification,
    #XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from utils_ner_relation_coref_span import convert_examples_to_features, get_labels, get_span_labels, convert_span_labels, get_rel_labels, read_examples_from_file, get_entities, bucket_values


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    #"xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    training = torch.tensor([1])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        #model.to('cuda')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "span" not in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility

    np.set_printoptions(threshold=100000)
    
    for epoch_num in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            '''print(batch[0].size())
            print(batch[3].size())
            print(batch[4].size())
            print(batch[7].size())'''

            span_indices = batch[4].cpu().numpy()
            span_labels = batch[6].cpu().numpy()
            rel_indices = batch[7].cpu().numpy()
            rel_labels = batch[8].cpu().numpy()
            span_mask = batch[5]

            #batch_size = [0]*len(span_labels)
            batch_size = 0
            
            for batch_num, subbatch_span_labels in enumerate(span_labels):
                for subbatch_num, span_label in enumerate(subbatch_span_labels):
                    if span_label.sum() == -1 * len(span_label):
                        break
                    #batch_size[batch_num] += 1
                    batch_size += 1
            '''print(batch_size)

            print(span_indices.shape)
            print(span_labels.shape)
            print(rel_indices.shape)
            print(rel_labels.shape)
            print(span_mask.size())'''
                
            #sentence_span_indices = batch[9].cpu().numpy()

            #print(batch[3].size())
        
            #new_rel_labels = np.ones((batch[4].size(1), batch[4].size(2), batch[4].size(2)))*-1
            #new_rel_labels = np.zeros((len(span_labels), max(batch_size), batch[4].size(2), batch[4].size(2)))
            new_rel_labels = np.zeros((batch_size, batch[4].size(2), batch[4].size(2)))
            
            #print(new_rel_labels.shape)
            
            '''for batch_num, subdoc_spans in enumerate(span_indices):
                for subbatch_num, sent_spans in enumerate(subdoc_spans):
                    print(sent_spans)

                    span_indices = []
                    for span_num, [span_ind1, span_ind2] in enumerate(sent_spans):
                        #if span_ind1 == -1 or span_ind2 == -1:
                        if span_labels[batch_num, subbatch_num, span_num] == -1:
                            continue
                        span_indices.append(span_num)
                    #sent_indices = np.arange(span_ind1, span_ind2+1)
                    span_indices = span_indices[:, np.newaxis]
                    new_rel_labels[subbatch_num, span_indices, span_indices.transpose(1, 0)]  = 0'''
                    
            '''print(new_rel_labels.mean())
            if new_rel_labels.mean().item() == -1:
                new_rel_labels[0, 0, 0] = 0'''

            
            
            #print(rel_indices[0, :batch_size])
            #print(rel_labels[0, :batch_size])
            
            '''for batch_num, (subbatch_rel_label, subbatch_rel_index) in enumerate(zip(rel_labels, rel_indices)):
                for subbatch_num, (sent_labels, sent_indices) in enumerate(zip(subbatch_rel_label[:batch_size], subbatch_rel_index[:batch_size])):
                    for label, [span_ind1, span_ind2] in zip(sent_labels, sent_indices):
                        #if span_ind1 != -1 and span_ind2 != -1:
                        if label > 0:
                            #is an entity!!
                            new_rel_labels[subbatch_num, span_ind1, span_ind2] = label
                            #new_rel_labels[batch_num, span_ind1, span_ind2] = -1
                            new_rel_labels[subbatch_num, span_ind2, span_ind1] = -1'''
                        
                            #print(str(batch_num)+"\t"+str(span_ind1)+"\t"+str(span_ind2)+"\t"+str(label))

            #print(new_rel_labels)
                        
            '''for batch_num, pairwise_relations in enumerate(new_rel_labels):
                for ent1, rel_entities in enumerate(pairwise_relations):
                    for ent2, relation_type in enumerate(rel_entities):
                        if batch[5][batch_num][ent1] == 0 or batch[5][batch_num][ent2] == 0:
                            new_rel_labels[batch_num, span_ind1, span_ind2] = -1'''

            newer_rel_labels = new_rel_labels
            '''for p, (sliced, sliced_span_mask) in enumerate(zip(new_rel_labels, span_mask[0][:batch_size])):
                sliced_span_mask = sliced_span_mask.unsqueeze(-1)
                mask_entry = sliced_span_mask & sliced_span_mask.transpose(0, 1).unsqueeze(0)
                mask_entry = mask_entry.detach().cpu().numpy()
                sliced[~mask_entry] = -1
                newer_rel_labels[p] = sliced'''
                                        
            #Place -1's

            coref_indices = batch[9].cpu().numpy()
            coref_labels = batch[10].cpu().numpy()

            '''print(coref_labels.shape)
            print(coref_indices.shape)
            print(coref_labels.shape)
            print(batch[4].size(2))
            print("+++++")'''
            
            #new_coref_labels = np.zeros((batch_size, batch_size))   
            
            new_coref_labels = np.zeros((batch_size * batch[4].size(2), batch_size * batch[4].size(2)+1))   
            
            coref_dicts = dict()
            coref_sentlevel_dicts = dict()

            #for batch_num, (coref_indices2, coref_labels2) in enumerate(zip(coref_indices, coref_labels)):
            for batch_index, (coref_index1, coref_label1) in enumerate(zip(coref_indices[0], coref_labels[0])):
                for index, label in zip(coref_index1, coref_label1):
                    if label == 0 or label == -1:
                        continue
                    if label in coref_dicts.keys():
                        coref_dicts[label].append(batch_index * batch[4].size(2) + index[0])
                    else:
                        coref_dicts[label] = []
                        coref_dicts[label].append(batch_index * batch[4].size(2) + index[0])


            for batch_index, (coref_index1, coref_label1) in enumerate(zip(coref_indices[0], coref_labels[0])):
                for index, label in zip(coref_index1, coref_label1):
                    if label == 0 or label == -1:
                        continue
                    if batch_index not in coref_sentlevel_dicts.keys():
                        coref_sentlevel_dicts[batch_index] = dict()
                    
                    if label in coref_sentlevel_dicts[batch_index].keys():           
                        coref_sentlevel_dicts[batch_index][label].append(index[0])
                    else:
                        coref_sentlevel_dicts[batch_index][label] = []
                        coref_sentlevel_dicts[batch_index][label].append(index[0])

            '''for i in coref_sentlevel_dicts.keys():
                print(coref_sentlevel_dicts[i])
                print(coref_indices[0][i])
                print(coref_labels[0][i])
                print("=====")'''
                
            coref_chains = dict()
            for batch_index in coref_sentlevel_dicts.keys():
                if batch_index not in coref_chains.keys():
                    coref_chains[batch_index] = dict()
                for coref_label in coref_sentlevel_dicts[batch_index].keys():
                    for ent1 in coref_sentlevel_dicts[batch_index][coref_label]:
                        coref_chains[batch_index][ent1] = []
                        for ent2 in coref_sentlevel_dicts[batch_index][coref_label]:
                            #if ent1 != ent2:
                            coref_chains[batch_index][ent1].append(ent2)

            '''for i in coref_chains.keys():
                print(coref_sentlevel_dicts[i])                
                print(coref_chains[i])
                print("========")'''
                
            for batch_num, (subbatch_rel_label, subbatch_rel_index) in enumerate(zip(rel_labels, rel_indices)):
                for subbatch_num, (sent_labels, sent_indices) in enumerate(zip(subbatch_rel_label[:batch_size], subbatch_rel_index[:batch_size])):
                    for label, [span_ind1, span_ind2] in zip(sent_labels, sent_indices):
                        #if span_ind1 != -1 and span_ind2 != -1:
                        if label > 0:
                            #is an entity!!
                            new_rel_labels[subbatch_num, span_ind1, span_ind2] = label
                            #new_rel_labels[batch_num, span_ind1, span_ind2] = -1
                            #new_rel_labels[subbatch_num, span_ind2, span_ind1] = -1
                            new_rel_labels[subbatch_num, span_ind2, span_ind1] = label
                            #coref_chains can be used to assign all the other values with -1
                            '''if span_ind1 in coref_chains[subbatch_num].keys() and span_ind2 in coref_chains[subbatch_num].keys():
                                #print(str(span_ind1)+"\t"+str(span_ind2))
                                for coreferent1 in coref_chains[subbatch_num][span_ind1]:
                                    for coreferent2 in coref_chains[subbatch_num][span_ind2]:
                                        if new_rel_labels[subbatch_num, coreferent1, coreferent2] == 0:
                                            new_rel_labels[subbatch_num, coreferent1, coreferent2] = -1
                                            #print("Added1\t"+str(coreferent1)+"\t"+str(coreferent2))
                                        if new_rel_labels[subbatch_num, coreferent2, coreferent1] == 0:
                                            new_rel_labels[subbatch_num, coreferent2, coreferent1] = -1
                                            #print("Added2\t"+str(coreferent2)+"\t"+str(coreferent1))
                                                                                        
                            else:
                                print("Not Found\t"+str(span_ind1)+"\t"+str(span_ind2))
                                print(rel_indices)
                                print(coref_indices)
                                print(coref_labels)
                                exit()'''

            
            #Now for each coreference chain...we add the element
            #print(batch_size)
            #print(coref_indices)
            #print(coref_dicts)
            #exit()
            
            '''indices = np.array(np.arange(len(new_coref_labels)))
            indices2 = indices[:, np.newaxis]                
            iu1 = np.triu_indices(len(indices2))
            new_coref_labels[indices[iu1[0]], indices[iu1[1]]] = -1                
            new_coref_labels[indices, indices] = -1'''                
            
            #for batch_num, coref_dicts1 in enumerate(coref_dicts):
            '''for coref_num in coref_dicts.keys():
                indices = np.array(coref_dicts[coref_num])
                indices2 = indices[:, np.newaxis]
                #print(indices)
                indices1 = indices2.transpose(1, 0)
                new_coref_labels[indices2, indices1] = 1
                iu1 = np.triu_indices(len(indices2))
                #print(iu1)
                #print(indices2)
                #print(indices[iu1[0]])
                #print(indices[iu1[1]])
                new_coref_labels[indices[iu1[0]], indices[iu1[1]]] = -1                
                new_coref_labels[indices, indices] = 0'''
                #maybe remove all (x, x)?

            '''for index, ent1 in enumerate(new_coref_labels):
                if new_coref_labels[index, :index].sum() == 0:
                    #print("Adds to null antecedant")
                    new_coref_labels[index, len(new_coref_labels)] = 1
                    #else:
                    #    print("Did not add")
                    #    print(new_coref_labels[index, :index])
                    #    print(new_coref_labels[index, index:])
                    #    print("=====")
            '''
            #span_mask_doc = span_mask[0].flatten().unsqueeze(-1)
            #mask_entry = span_mask_doc & span_mask_doc.transpose(0, 1)
            #mask_entry = mask_entry.detach().cpu().numpy()
            #print(mask_entry)
            #Add a more efficient replacement
            #new_coref_labels[~mask_entry] = -1
            #print(batch[4].squeeze(0)[:max(batch_size)].size())
            #print(new_coref_labels.shape)
            
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "entity_indices":batch[4].squeeze(0)[:batch_size], "entity_span_mask":batch[5][:, :batch_size].squeeze(0), "entity_labels":batch[6][:, :batch_size].squeeze(0), "rel_labels": torch.tensor(newer_rel_labels, dtype=torch.long).to(args.device), "coref_labels":torch.tensor(new_coref_labels, dtype=torch.long).to(args.device), "training":training.to(args.device)}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss1 = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss2 = outputs[1]
            loss3 = outputs[2]
            loss4 = outputs[3]

            '''if (global_step < args.save_steps*3):
                loss = loss1 + loss2 +loss3
                logger.info("Only Entity Training")
            elif (global_step < args.save_steps * 6):
                #loss = 2*loss1+(loss2+loss3)
                loss = loss1 + loss2 + loss3
                logger.info("Coref Training")
            else:
                loss = loss1 + loss2 + loss3
                logger.info("Relation Training")'''
            #loss1: entity, loss2: relation, loss3: coreference
            #loss = 2*loss1+0.05*(loss2)+(batch_size/14.0)*(loss3)
            #loss = loss1 + loss2 + loss3
            #loss = loss1 + loss2 + loss4
            loss = loss1 + loss2 + loss4
            #loss = loss1
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training and epoch_num > -1
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        #results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        result, _, _, _, type_result, _, _, _, rel_result, _, _, _ = evaluate(args, model, tokenizer, labels, rel_labels, pad_token_label_id, mode="dev", prefix=global_step)
                        '''
                        for key in sorted(results.keys()):
                        writer.write("{} = {}\n".format(key, str(results[key])))
                        
                        for key, value in result.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)'''
                    #tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    #tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    #logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, rel_labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    training = torch.tensor([0])

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    rel_preds = []
    rel_out_labels = []
    rel_out_index = []
    
    entity_preds = []
    out_labels = []
    entity_index  = []

    type_entity_preds = []
    type_out_labels = []
    type_entity_index  = []
                

    total_relations = 0
    
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            span_indices = batch[4].cpu().numpy()
            span_labels = batch[6].cpu().numpy()
                                    
            rel_indices = batch[7].cpu().numpy()
            rel_labels = batch[8].cpu().numpy()
            span_mask = batch[5]


            batch_size = 0
            
            for batch_num, subbatch_span_labels in enumerate(span_labels):
                for subbatch_num, span_label in enumerate(subbatch_span_labels):
                    if span_label.sum() == -1 * len(span_label):
                        break
                    batch_size += 1
                    #print(batch_size)
                    
            #print(batch_size)
            if batch_size == 1:
                batch_size += 1
            #sentence_span_indices = batch[9].cpu().numpy()            
            #new_rel_labels = np.ones((batch[3].size(0), batch[3].size(1)*7, batch[3].size(1)*7))*-1
            new_rel_labels = np.zeros((batch_size, batch[4].size(2), batch[4].size(2)))
            
            '''for batch_num, spans in enumerate(sentence_span_indices):
                for [span_ind1, span_ind2] in spans:
                    sent_indices = np.arange(span_ind1, span_ind2+1)
                    sent_indices = sent_indices[:, np.newaxis]
                    new_rel_labels[batch_num, sent_indices, sent_indices.transpose(1, 0)]  = 0'''
                                                                                                
            '''for batch_num, (rel_label, rel_index) in enumerate(zip(rel_labels, rel_indices)):
                for label, [span_ind1, span_ind2] in zip(rel_label, rel_index):
                    if label > 0:
                        #is an entity!!
                        new_rel_labels[batch_num, span_ind1, span_ind2] = label
                        new_rel_labels[batch_num, span_ind2, span_ind1] = -1                        
                        #print(str([batch_num, span_ind1, span_ind2])+"\t"+str(label))
            #print(new_rel_labels)'''

            for batch_num, (subbatch_rel_label, subbatch_rel_index) in enumerate(zip(rel_labels, rel_indices)):
                for sent_labels, sent_indices in zip(subbatch_rel_label[:batch_size], subbatch_rel_index[:batch_size]):
                    for label, [span_ind1, span_ind2] in zip(sent_labels, sent_indices):
                        #if span_ind1 != -1 and span_ind2 != -1:
                        if label > 0:
                            #is an entity!!
                            new_rel_labels[batch_num, span_ind1, span_ind2] = label
                            #new_rel_labels[batch_num, span_ind1, span_ind2] = -1
                            new_rel_labels[batch_num, span_ind2, span_ind1] = -1

                            #print(str(batch_num)+"\t"+str(span_ind1)+"\t"+str(span_ind2)+"\t"+str(label))                                                                                 

            newer_rel_labels = new_rel_labels
            '''for p, (sliced, sliced_span_mask) in enumerate(zip(new_rel_labels, span_mask[0])):
                sliced_span_mask = sliced_span_mask.unsqueeze(-1)
                mask_entry = sliced_span_mask & sliced_span_mask.transpose(0, 1).unsqueeze(0)
                mask_entry = mask_entry.detach().cpu().numpy()
                sliced[~mask_entry] = -1
                newer_rel_labels[p] = sliced'''

            coref_indices = batch[9].cpu().numpy()
            coref_labels = batch[10].cpu().numpy()

            new_coref_labels = np.zeros((batch_size * batch[4].size(2), batch_size * batch[4].size(2)))
            
            coref_dicts = dict()

            for batch_index, (coref_index1, coref_label1) in enumerate(zip(coref_indices[0], coref_labels[0])):
                for index, label in zip(coref_index1, coref_label1):
                    if label == 0 or label == -1:
                        continue
                    if label in coref_dicts.keys():
                        coref_dicts[label].append(batch_index * batch[4].size(2) + index[0])
                    else:
                        coref_dicts[label] = []
                        coref_dicts[label].append(batch_index * batch[4].size(2) + index[0])

            for coref_num in coref_dicts.keys():
                indices = np.array(coref_dicts[coref_num])
                indices2 = indices[:, np.newaxis]
                #print(indices)
                indices1 = indices2.transpose(1, 0)
                new_coref_labels[indices2, indices1] = 1
                iu1 = np.triu_indices(len(indices2))
                #print(iu1)
                #print(indices2)
                #print(indices[iu1[0]])
                #print(indices[iu1[1]])
                new_coref_labels[indices[iu1[0]], indices[iu1[1]]] = -1
                
                new_coref_labels[indices, indices] = 0
                #maybe remove all (x, x)?
            
            
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "entity_indices":batch[4][:, :batch_size].squeeze(0), "entity_span_mask":batch[5][:, :batch_size].squeeze(0), "entity_labels":batch[6][:, :batch_size].squeeze(0), "rel_labels": torch.tensor(newer_rel_labels, dtype=torch.long).to(args.device), "coref_labels":torch.tensor(new_coref_labels, dtype=torch.long).to(args.device), "training":training.to(args.device)}
            
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss1, tmp_eval_loss2, tmp_eval_loss3, tmp_eval_loss4, logits, rel_logits, coref_logits, type_logits, top_indices, top_mask, flat_top_indices = outputs[:11]
            
            #tmp_eval_loss = tmp_eval_loss1 + tmp_eval_loss2 + tmp_eval_loss3
            tmp_eval_loss = tmp_eval_loss1

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            #print(batch[5](batch[5]>1))
            #print("---")

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        #label_map = {i: label for i, label in enumerate(convert_span_labels(labels))}        

        out_label = inputs["entity_labels"].clamp(0, 1).squeeze(0).detach().cpu().numpy()
        entity_indices1 = inputs["entity_indices"].squeeze(0).detach().cpu().numpy()

        type_out_label = inputs["entity_labels"].squeeze(0).detach().cpu().numpy()
        type_entity_indices1 = inputs["entity_indices"].squeeze(0).detach().cpu().numpy()
                
        
        #print(out_label.shape)
        #print(len(out_label.shape))
        
        if len(out_label.shape) < 2:
            out_label = out_label.unsqueeze(0)

        if len(type_out_label.shape) < 2:
            type_out_label = type_out_label.unsqueeze(0)
                        
        #type_preds = np.argmax((type_logits*top_mask.unsqueeze(-1)).detach().cpu().numpy(), axis=2).tolist()
        type_preds = np.argmax((type_logits).detach().cpu().numpy(), axis=2).tolist() 
        
        preds = np.argmax(logits.detach().cpu().numpy(), axis=2).tolist()

        #print(out_label)
        #print(preds)
        #print("!!!")

        for i, (type_pred, type_label, top_index, top_mask1) in enumerate(zip(type_preds, type_out_label, top_indices, top_mask)):
            pred_local = []
            label_local = []
            entity_index_local = []
            for j, type_label_t in enumerate(type_label):
                if type_label_t == -1:
                    continue
                if j in top_index:
                    index1 = top_index.detach().cpu().numpy().tolist().index(j)
                    if top_mask1[index1] > 0:
                        pred_label_temp = type_pred[index1]
                    else:
                        pred_label_temp = 0
                                            
                else:
                    pred_label_temp = 0
                if pred_label_temp == 0 and type_label_t == 0:
                    continue
                else:
                    pred_local.append(pred_label_temp)
                    label_local.append(type_label_t)
                    entity_index_local.append(type_entity_indices1[i][j])
            type_entity_preds.append(pred_local)
            type_out_labels.append(label_local)
            type_entity_index.append(entity_index_local)

        '''for i, (pred, label) in enumerate(zip(type_preds, type_out_label)):
            pred_local = []
            label_local = []
            entity_index_local = []
            #print(pred)
            #print(label)
            
            for pred_t, label_t, index_t in zip(pred, label, entity_indices1[i]):
                if label_t != -1:
                    pred_local.append(pred_t)
                    label_local.append(label_t)
                    entity_index_local.append(index_t)
                    
            type_entity_preds.append(pred_local)
            type_out_labels.append(label_local)
            type_entity_index.append(entity_index_local)'''
            
        
        for i, (pred, label) in enumerate(zip(preds, out_label)):
            pred_local = []
            label_local = []
            entity_index_local = []
            #print(pred)
            #print(label)
            
            for pred_t, label_t, index_t in zip(pred, label, entity_indices1[i]):
                if label_t != -1:
                    pred_local.append(pred_t)
                    label_local.append(label_t)
                    entity_index_local.append(index_t)
                    
            entity_preds.append(pred_local)
            out_labels.append(label_local)
            entity_index.append(entity_index_local)

            #print("Ent-label\t"+str(len(pred_local)))
            
            #top_indices = top_indices.cpu().numpy()
            #rel_indices = rel_indices.detach().cpu().numpy()
            #rel_labels = rel_labels.detach().cpu().numpy()

        #print("====")

        top_mask_flattened = top_mask.flatten().unsqueeze(-1)
        mask_logit_flattened = top_mask_flattened & top_mask_flattened.transpose(0, 1)

        print(coref_dicts)
        
        coref_preds = np.argmax((coref_logits * mask_logit_flattened.unsqueeze(-1)).detach().cpu().numpy(), axis=2)
        #coref_preds = ((coref_logits[:, :-1] * mask_logit_flattened).detach().cpu().numpy()>0.5).int()
        pred_coref_dicts = dict()
        key_val = 0
        
        chains = []
        
        for i, antecedants in enumerate(coref_preds):
            span1 = flat_top_indices[i].item()
            local_chain = []
            for j, coref_val in enumerate(antecedants):
                if coref_val > 0:
                    #[i, j] is the pair
                    #span1 = flat_top_indices[i].item()
                    span2 = flat_top_indices[j].item()
                    local_chain.append([span1, span2])
                    notAdded = 1
                    for key in pred_coref_dicts:
                        coref_chain = pred_coref_dicts[key]
                        if span1 in coref_chain and span2 in coref_chain:
                            notAdded = 0
                            break
                        elif span1 in coref_chain and span2 not in coref_chain:
                            notAdded = 0
                            pred_coref_dicts[key].append(span2)
                            break
                        elif span1 not in coref_chain and span2 in coref_chain:
                            notAdded = 0
                            pred_coref_dicts[key].append(span1)
                            break
                    if notAdded:
                        pred_coref_dicts[key_val] = []
                        pred_coref_dicts[key_val].append(span1)
                        pred_coref_dicts[key_val].append(span2)
                        
                        key_val += 1
            chains.append(local_chain)
        #print(chains)
        print(pred_coref_dicts)
        print("====")
        
        #relation part commented from here...
        '''for i, (rel_logit_slice, top_index, top_mask1, rel_index, rel_label) in enumerate(zip(rel_logits, top_indices, top_mask, rel_indices[0], rel_labels[0])):
            #print(top_index)
            #print(top_mask1)
            top_mask1 = top_mask1.unsqueeze(-1)
            mask_logit = top_mask1 & top_mask1.transpose(0, 1)
            #print(mask_logit)
            #print(mask_logit.size())
            #print(rel_logit_slice.size())
            rel_preds1 = np.argmax((rel_logit_slice * mask_logit.unsqueeze(-1)).detach().cpu().numpy(), axis=2)
            #print(rel_preds1)
            ind1, ind2 = np.where(rel_preds1 > 0)
            #print(ind1)
            #print(ind2)
            #print(top_index)
            top_index = top_index.cpu().numpy()
            rel_pred_indices_temp = np.stack([top_index[ind1], top_index[ind2]], axis=1).tolist()
            rel_preds_temp_temp = rel_preds1[ind1, ind2]
            rel_preds_prob_temp = rel_logit_slice[ind1, ind2, rel_preds_temp_temp]

            rel_pred_indices = []
            rel_preds_temp = []
            
            #removing duplicates from this rel_pred_indices_list

            #print(rel_index)
            #print(rel_label)
            
            #print(rel_pred_indices_temp)
            #print(rel_preds_temp_temp)

            for i, [pred_ind1, pred_ind2] in enumerate(rel_pred_indices_temp):
                if [pred_ind2, pred_ind1] in rel_pred_indices_temp[i+1:]:
                    prob1 = rel_preds_prob_temp[i]
                    prob2 = rel_preds_prob_temp[rel_pred_indices_temp.index([pred_ind2, pred_ind1])]
                    if prob1 > prob2:
                        rel_pred_indices.append([pred_ind1, pred_ind2])
                        rel_preds_temp.append(rel_preds_temp_temp[i])
                elif [pred_ind2, pred_ind1] not in rel_pred_indices and [pred_ind1, pred_ind2] not in rel_pred_indices:
                    rel_pred_indices.append([pred_ind1, pred_ind2])
                    rel_preds_temp.append(rel_preds_temp_temp[i])

            #print(rel_pred_indices)
            #print(rel_preds_temp)
                    
            rel_label_local = []
            rel_pred_local = []
            rel_index_local = []
            
            for (gold_ind, gold_label) in zip(rel_index, rel_label):
                if gold_label == -1:
                    continue
                #print(gold_ind)
                rel_index_local.append(gold_ind)
                [gold_ind1, gold_ind2] = gold_ind
                if [gold_ind1, gold_ind2] in rel_pred_indices:
                    #both have them
                    rel_label_local.append(gold_label)
                    rel_pred_local.append(rel_preds_temp[rel_pred_indices.index([gold_ind1, gold_ind2])])
                elif [gold_ind2, gold_ind1] in rel_pred_indices:
                    rel_label_local.append(gold_label)
                    rel_pred_local.append(rel_preds_temp[rel_pred_indices.index([gold_ind2, gold_ind1])])
                    
                else:
                    rel_label_local.append(gold_label)
                    rel_pred_local.append(0)

            for (pred_ind, pred_label) in zip(rel_pred_indices, rel_preds_temp):
                [pred_ind1, pred_ind2] = pred_ind
                if [pred_ind1, pred_ind2] not in rel_index and [pred_ind2, pred_ind1] not in rel_index:
                    rel_index_local.append(pred_ind)
                    rel_pred_local.append(pred_label)
                    rel_label_local.append(0)

            rel_preds.append(rel_pred_local)
            rel_out_labels.append(rel_label_local)
            rel_out_index.append(rel_index_local)'''

            #print(rel_pred_local)
            #print(rel_label_local)
            #print(rel_index_local)
            #print("=========")
            
            #print("Rel-label\t"+str(len(rel_pred_local)))
            
    rel_labels1 = get_rel_labels()
        
    #print("Total_relations\t"+str(total_relations))
    rel_label_map = {i: label for i, label in enumerate(rel_labels1)}
            
    rel_out_label_list = [[] for _ in range(len(rel_out_labels))]
    rel_preds_list = [[] for _ in range(len(rel_preds))]
    rel_out_index_list = [[] for _ in range(len(rel_out_index))]
    
    total_relations = 0
    for i in range(len(rel_preds)):
        for ent in rel_preds[i]:
            total_relations += 1
            if ent >0:
                rel_preds_list[i].append("S-"+rel_label_map[ent].replace("-", "_").replace("ORG", "1ORG"))
            #elif ent > 0:
            #    rel_preds_list[i].append(rel_label_map[ent])
            else:
                rel_preds_list[i].append(rel_label_map[0])
                                
    #print("Predicted relations\t"+str(total_relations))

    #print(rel_label_map)
    
    total_relations = 0
    for i in range(len(rel_out_labels)):
        for ent in rel_out_labels[i]:
            if ent > 0:
                #print(ent)
                rel_out_label_list[i].append("S-"+rel_label_map[ent].replace("-", "_").replace("ORG", "1ORG"))
            #elif ent > 0:
            #    rel_out_label_list[i].append(rel_label_map[ent])
            else:
                rel_out_label_list[i].append(rel_label_map[0])
                                
            total_relations += 1
    #print("Gold_relations\t"+str(total_relations))

    '''for i in range(len(rel_preds_list)):
        print(rel_preds_list[i])
        print(rel_out_label_list[i])
        print("----")'''

    
    for i in range(len(rel_out_index)):
        for ent in rel_out_index[i]:
            rel_out_index_list[i].append(ent)
    

    eval_loss = eval_loss / nb_eval_steps
    #preds = np.argmax(preds, axis=2)

    #entity_label_map = {i: label for i, label in enumerate(convert_span_labels(labels))}
    entity_label_map = {0: "O", 1:"MENTION"}
        
    #print(rel_label_map)
    #print(entity_label_map)

    entity_out_label_list = [[] for _ in range(len(out_labels))]
    entity_preds_list = [[] for _ in range(len(out_labels))]
    entity_indices_list = [[] for _ in range(len(out_labels))]

    '''for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])'''

    for i in range(len(entity_preds)):
        for ent in entity_preds[i]:
            if ent > 0:
                entity_preds_list[i].append("S-"+entity_label_map[ent])
            else:
                entity_preds_list[i].append(entity_label_map[ent])

    for i in range(len(out_labels)):
        for ent in out_labels[i]:
            if ent > 0:
                entity_out_label_list[i].append("S-"+entity_label_map[ent])
            else:
                entity_out_label_list[i].append(entity_label_map[ent])

    for i in range(len(out_labels)):
        for index in entity_index[i]:
            entity_indices_list[i].append(index)
                


    type_entity_label_map = {i: label for i, label in enumerate(convert_span_labels(labels))}

    #print(type_entity_label_map)

    type_entity_out_label_list = [[] for _ in range(len(type_out_labels))]
    type_entity_preds_list = [[] for _ in range(len(type_out_labels))]
    type_entity_indices_list = [[] for _ in range(len(type_out_labels))]
    
    for i in range(len(type_entity_preds)):
        for ent in type_entity_preds[i]:
            if ent > 0:
                type_entity_preds_list[i].append("S-"+type_entity_label_map[ent])
            else:
                type_entity_preds_list[i].append(type_entity_label_map[ent])

    #print(type_entity_preds_list)

    for i in range(len(type_out_labels)):
        for ent in type_out_labels[i]:
            if ent > 0:
                type_entity_out_label_list[i].append("S-"+type_entity_label_map[ent])
            else:
                type_entity_out_label_list[i].append(type_entity_label_map[ent])

    #print(type_entity_out_label_list)
    for i in range(len(type_out_labels)):
        for index in type_entity_index[i]:
            type_entity_indices_list[i].append(index)

    #logger.info("out_label_list\t %s", str(out_label_list))
    #logger.info("preds_list\t %s", str(preds_list))
                
    results = {
        "loss": eval_loss,
        "precision": precision_score(entity_out_label_list, entity_preds_list),
        "classification_report": classification_report(entity_out_label_list, entity_preds_list, digits=4),
        "recall": recall_score(entity_out_label_list, entity_preds_list),
        "f1": f1_score(entity_out_label_list, entity_preds_list),
    }

    type_results = {
        "loss": eval_loss,
        "precision": precision_score(type_entity_out_label_list, type_entity_preds_list),
        "classification_report": classification_report(type_entity_out_label_list, type_entity_preds_list, digits=4),
        "recall": recall_score(type_entity_out_label_list, type_entity_preds_list),
        "f1": f1_score(type_entity_out_label_list, type_entity_preds_list),
    }
        
    logger.info("***** Mention Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    logger.info("***** Type Eval results %s *****", prefix)
    for key in sorted(type_results.keys()):
        logger.info("  %s = %s", key, str(type_results[key]))

    
    '''rel_results = {
        "loss": eval_loss,
        "precision": precision_score(rel_out_label_list, rel_preds_list),
        "classification_report": classification_report(rel_out_label_list, rel_preds_list, digits=4),
        "recall": recall_score(rel_out_label_list, rel_preds_list),
        "f1": f1_score(rel_out_label_list, rel_preds_list),
    }'''

    rel_results = type_results

    logger.info("***** REL Eval results %s *****", prefix)
    for key in sorted(rel_results.keys()):
        logger.info("  %s = %s", key, str(rel_results[key]))
    #rel_results = {}

    #exit()

    return results, entity_preds_list, entity_out_label_list, entity_indices_list, type_results, type_entity_preds_list, type_entity_out_label_list, type_entity_indices_list, rel_results, rel_preds_list, rel_out_label_list, rel_out_index_list
#inputs["labels"].detach().cpu().numpy()


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if mode == "train":
        max_seq_length = args.max_train_seq_length
    else:
        max_seq_length = args.max_test_seq_length

    print(max_seq_length)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        doc_list = read_examples_from_file(args, args.data_dir, mode)
        features = convert_examples_to_features(
            mode,
            doc_list,
            labels,
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            ####torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    
    '''for f in features:
        print(len(f.entity_tuples))
        print(f.entity_tuples)
        print(len(f.entity_tuples[0]))
        exit()'''
        
    all_entity_tuples = torch.tensor([f.entity_tuples for f in features], dtype=torch.long)
    all_entity_label_ids = torch.tensor([f.entity_label_ids for f in features], dtype=torch.long)
    all_entity_span_mask = torch.tensor([f.entity_span_mask for f in features], dtype=torch.long)
    
    all_relation_tuples = torch.tensor([f.relation_tuples for f in features], dtype=torch.long)
    all_rel_label_ids = torch.tensor([f.rel_label_ids for f in features], dtype=torch.long)

    all_coref_tuples = torch.tensor([f.coref_tuples for f in features], dtype=torch.long)
    all_coref_label_ids = torch.tensor([f.coref_label_ids for f in features], dtype=torch.long)
    
    #all_span_indices = torch.tensor([f.sentence_span_indices for f in features], dtype=torch.long)

    ######Here convert the relation strings to the torch tuples of relations!!
    ##all_relation_ids = [f.relation_ids for f in features]
    
    #dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_entity_tuples, all_entity_label_ids, all_relation_tuples, all_rel_label_ids)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_entity_tuples, all_entity_span_mask, all_entity_label_ids, all_relation_tuples,  all_rel_label_ids, all_coref_tuples, all_coref_label_ids)
    
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_train_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_test_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--num_widths", default=7, type=int, help="Maximum span width")
    parser.add_argument("--width_hidden_size", default=100, type=int, help="Embeddings for the span width embeddings")
    
    parser.add_argument(
        "--max_steps",
        default=20000,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=2000, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=45, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(torch.cuda.is_available())
        print(device)
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print(device)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    #args.device = "cpu"
    print(args.device)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    #span_labels = get_span_labels(args.labels)
    #num_labels = len(span_labels)
    num_labels = len(labels)
    rel_labels = get_rel_labels()
    num_rel_labels = len(rel_labels)
    
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=len(convert_span_labels(labels)),
        #num_labels,
        #num_rel_labels = num_rel_labels,
        output_hidden_states=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        print(checkpoints)
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        #exit()
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _, _, _, type_result, _, _, _, rel_result, _, _, _ = evaluate(args, model, tokenizer, labels, rel_labels, pad_token_label_id, mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            for key in sorted(rel_result.keys()):
                writer.write("{} = {}\n".format(key, str(rel_result[key])))
                                            
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        #rel_preds_list, rel_out_label_list\, rel_out_label_indices
        
        #for i in range(2):
        result, predictions, original_labels, indices_labels, type_result, type_predictions, type_original_labels, type_indices_labels, rel_result, rel_predictions, rel_original_labels, rel_out_label_indices = evaluate(args, model, tokenizer, labels, rel_labels, pad_token_label_id, mode="test")

        '''for i in range(len(rel_predictions)):
            print(rel_predictions[i])
            print(rel_original_labels[i])
            print(rel_out_label_indices[i])
            print("=====")'''
            
        #more_labels1 = copy.deepcopy(more_labels)
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
            for key in sorted(rel_result.keys()):
                writer.write("{} = {}\n".format(key, str(rel_result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")

        '''for pred, label in zip(predictions, original_labels):
            writer.write(str(pred)+"\n")
            writer.write("~~~~~")
            writer.write(str(label)+"\n")
            writer.write("\n")'''
        
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
                sentences = []
                curr_sent = ""
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if curr_sent != "":
                            sentences.append(curr_sent+"EOS")
                            curr_sent = ""
                    else:
                        curr_sent += line.split()[0]+"\t"+line.split()[1]+"\n"

                print(len(sentences))
                print(len(predictions))
                #exit()
                        
                for pred, label, index, type_pred, type_label, type_index, curr_sent, rel_pred, rel_label, rel_index in zip(predictions, original_labels, indices_labels, type_predictions, type_original_labels, type_indices_labels, sentences, rel_predictions, rel_original_labels, rel_out_label_indices):
                    writer.write(str(curr_sent)+"\n")
                    for pred1, label1, index1 in zip(pred, label, index):
                        if pred1 != "O" or label1 != "O":
                            writer.write(str(index1)+"--")
                            #writer.write("~~~~~")                                
                            writer.write(str(label1)+"--")
                            #writer.write("~~~~~")
                            writer.write(str(pred1)+";;;")
                            #writer.write("\n")
                    writer.write("\n=======\n")
                    for pred1, label1, index1 in zip(type_pred, type_label, type_index):
                        if pred1 != "O" or label1 != "O":
                            writer.write(str(index1)+"--")
                            #writer.write("~~~~~")
                            writer.write(str(label1)+"--")
                            #writer.write("~~~~~")
                            writer.write(str(pred1)+";;;")
                            #writer.write("\n")
                    writer.write("\n=======\n")
                    
                    for rel_pred1, rel_label1, rel_index1 in zip(rel_pred, rel_label, rel_index):
                        #writer.write("==========")
                        writer.write(str(rel_index1)+"--")
                        #writer.write("~~~~~")
                        writer.write(str(rel_label1)+"--")
                        #writer.write("~~~~~")
                        writer.write(str(rel_pred1)+";;;")
                        #writer.write("\n")
                    writer.write("\n\n")
                                                                        
                #with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
                '''example_id = 0
                for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                #writer.write(str(more_labels[example_id])+"\n\n")
                writer.write(line)
                if not predictions[example_id]:
                example_id += 1
                elif predictions[example_id]:
                output_line = line.split()[0] + "\t" + line.split()[1]+ "\t"+original_labels[example_id].pop(0)+"\t"+ predictions[example_id].pop(0) + "\n"
                writer.write(output_line)'''
                #writer.write(str(more_labels1[example_id])+"\n")
                '''else:
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])'''
                #writer.write(str(get_entities(labels, False)) + "\n")


        '''result, predictions, original_labels, more_labels, rel_result, rel_predictions, rel_original_labels, rel_out_label_indices = evaluate(args, model, tokenizer, labels, rel_labels, pad_token_label_id, mode="test2")
        more_labels1 = copy.deepcopy(more_labels)
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test2_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
            for key in sorted(rel_result.keys()):
                writer.write("{} = {}\n".format(key, str(rel_result[key])))
                                                                                                            

        output_test_predictions_file = os.path.join(args.output_dir, "test2_predictions.txt")

        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        #writer.write(str(more_labels[example_id])+"\n\n")
                        writer.write(line)
                        if example_id !=0 and np.mod(example_id, args.eval_batch_size) == 0 and len(rel_out_label_indices) > 0:
                            
                            writer.write(str(more_labels.pop(0))+"\t"+str(original_labels.pop(0))+"\t"+str(predictions.pop(0))+"\n")
                            writer.write(str(rel_out_label_indices.pop(0))+"\t"+str(rel_original_labels.pop(0))+"\t"+str(rel_predictions.pop(0))+"\n")
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split()[0] + "\t" + line.split()[1]+ "\n"
                        #"\t"+original_labels[example_id].pop(0)+"\t"+ predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
            #writer.write(str(more_labels1[example_id])+"\n")
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
                        #writer.write(str(get_entities(labels, False)) + "\n")'''
                
    return results


if __name__ == "__main__":
    main()
