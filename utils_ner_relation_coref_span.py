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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

import math

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, entity_spans, relations, span_corefs):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.entity_spans = entity_spans
        self.relations = relations
        self.span_corefs = span_corefs


class DocumentExample(object):
    """All documents for entity spans, relation and coreference classification."""
    
    #list of all the input sentences === words, labels, entity_spans, relations, coref_values for the entity spans
    def __init__(self, inputSentences):
        self.inputSentences = inputSentences
    

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, entity_tuples, entity_span_mask, entity_label_ids, relation_tuples, rel_label_ids, coref_tuples, coref_label_ids, sentence_span_indices=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.relation_tuples = relation_tuples
        self.rel_label_ids = rel_label_ids
        self.entity_tuples = entity_tuples
        self.entity_span_mask = entity_span_mask
        self.entity_label_ids = entity_label_ids
        self.coref_tuples = coref_tuples
        self.coref_label_ids = coref_label_ids
        #self.sentence_span_indices = sentence_span_indices


def relation_tuples(trimmed_labels, labels, corefs, entity_corefs):

    #print(entity_corefs)
    #exit()
    
    #TODO: Automatically get them 
    entities = ["PER", "ORG", "GPE", "LOC", "FAC", "WEA", "VEH"]
    relation_types = {"PHYS":1, "PER-SOC":2, "ORG-AFF":3, "ART":4, "PART-WHOLE":5, "GEN-AFF":6}            
    
    gold_rel1_start = dict()
    gold_rel1_end = dict()
    gold_rel2_start = dict()
    gold_rel2_end = dict()

    for types in relation_types.keys() :
        gold_rel1_start[types] = dict()
        gold_rel1_end[types] = dict()
        gold_rel2_start[types] = dict()
        gold_rel2_end[types] = dict()
        
    #print(labels)
    #print(corefs)
    for i in range(len(labels)):
        if labels[i] == 'O':
            continue
        elif len(labels[i].split("_")) < 2:
            continue
        else:
            while "B-" in labels[i] or "S-" in labels[i]:
                for entity in entities:
                    if "B-"+entity in labels[i] or "S-"+entity in labels[i]:
                        current_tags = labels[i].split(";")
                        
                        ctag = ""
                        for tag in current_tags:
                            if "B-"+entity in tag or "S-"+entity in tag:
                                ctag = tag
                        relType = ctag.split("_")[1]
                        relId = ctag.split("_")[3]
                        argType = np.mod(int(ctag.split("_")[2]), 2)

                        if argType == 1:
                            gold_rel1_start[relType][relId] = i
                        if argType == 0:
                            gold_rel2_start[relType][relId] = i

                        labels[i] = labels[i].replace(ctag, "")

                        ctag = ctag[1:]
                        k = i
                        while k+1 < len(labels) and ctag in labels[k+1]:
                            k += 1
                            if argType == 1:
                                gold_rel1_end[relType][relId] = k
                            if argType == 0:
                                gold_rel2_end[relType][relId] = k
                            if "E-"+ctag in labels[i]:
                                labels[i] = labels[i].replace("E-"+ctag, "")

                            if "I-"+ctag in labels[i]:
                                labels[i] = labels[i].replace("I-"+ctag, "")
                            
                        if relId not in gold_rel1_end[relType].keys() or relId not in gold_rel2_end[relType].keys():
                            if argType == 1:
                                gold_rel1_end[relType][relId] = k
                            if argType == 0:
                                gold_rel2_end[relType][relId] = k

    '''
    {'PHYS': {'10': 30}, 'PER-SOC': {}, 'ORG-AFF': {'5': 2}, 'ART': {}, 'PART-WHOLE': {}, 'GEN-AFF': {}}
    {'PHYS': {'10': 30}, 'PER-SOC': {}, 'ORG-AFF': {'5': 2}, 'ART': {}, 'PART-WHOLE': {}, 'GEN-AFF': {}}
    {'PHYS': {'10': 32}, 'PER-SOC': {}, 'ORG-AFF': {'5': 5}, 'ART': {}, 'PART-WHOLE': {}, 'GEN-AFF': {}}
    {'PHYS': {'10': 32}, 'PER-SOC': {}, 'ORG-AFF': {'5': 5}, 'ART': {}, 'PART-WHOLE': {}, 'GEN-AFF': {}}'''

    tuples = []
    coref_tuples = []

    num_relations = 0
    
    for relType in gold_rel1_start.keys():
        for relId in gold_rel1_start[relType].keys():
            tuples.append([gold_rel1_start[relType][relId], gold_rel1_end[relType][relId], gold_rel2_start[relType][relId], gold_rel2_end[relType][relId], relType])
            #tuples.append([gold_rel2_start[relType][relId], gold_rel2_end[relType][relId], gold_rel1_start[relType][relId], gold_rel1_end[relType][relId], relType])
            num_relations += 1
            coref_tuples.append([corefs[gold_rel1_start[relType][relId]], corefs[gold_rel2_start[relType][relId]], relType])

    #print(coref_tuples)

    '''for relType in gold_rel1_start.keys():
        for relId in gold_rel1_start[relType].keys():
            for relType2 in gold_rel2_start.keys():
                for relId2 in gold_rel2_start[relType2].keys():
                    
                    coref_arg1 = corefs[gold_rel1_start[relType][relId]]
                    coref_arg2 = corefs[gold_rel2_start[relType2][relId2]]
                    
                    if relType == relType2 and relId == relId2:
                        continue
                    else:
                        found = False
                        for relType_all in gold_rel1_start.keys():
                            if [coref_arg1, coref_arg2, relType_all] in coref_tuples:
                                if ([gold_rel1_start[relType][relId], gold_rel1_end[relType][relId], gold_rel2_start[relType2][relId2], gold_rel2_end[relType2][relId2], relType_all]) not in tuples:
                                    tuples.append([gold_rel1_start[relType][relId], gold_rel1_end[relType][relId], gold_rel2_start[relType2][relId2], gold_rel2_end[relType2][relId2], relType_all])
                                    found=True
                                    #print(str(coref_arg1)+"-"+str(coref_arg2)+"-"+relType_all)
                                    #print("Added")
                                    #exit()
                        if not found:
                            tuples.append([gold_rel1_start[relType][relId], gold_rel1_end[relType][relId], gold_rel2_start[relType2][relId2], gold_rel2_end[relType2][relId2], "None"])

    print(tuples)

    print(labels)'''

    
    '''all_entities = get_entities(list(trimmed_labels))

    for (ent_type1, start1, end1) in all_entities:
        for (ent_type2, start2, end2) in all_entities:
            if (start1 == start2 and end1 == end2):
                continue
            
            found=False
            #for relType in gold_rel1_start.keys():
            for relType in relation_types.keys():
                if [corefs[start1], corefs[start2], relType] in coref_tuples or [corefs[start2], corefs[start1], relType] in coref_tuples:
                    #Add them if already not present
                    if [start1, end1, start2, end2, relType] not in tuples and [start2, end2, start1, end1, relType] not in tuples:
                        tuples.append([start1, end1, start2, end2, relType])
                        found=True

                    if [start1, end1, start2, end2, relType] in tuples or [start2, end2, start1, end1, relType] in tuples:
                        found=True
                        
                    #if [start2, end2, start1, end1, relType] not in tuples:
                    #    tuples.append([start2, end2, start1, end1, relType])
                    #    found=True

            for relType in gold_rel1_start.keys():
                if [start1, end1, start2, end2, relType] in tuples or [start2, end2, start1, end1, relType] in tuples:
                    found=True
                    
            if not found and [start2, end2, start1, end1, "O"] not in tuples:
                tuples.append([start1, end1, start2, end2, "O"])
                #tuples.append([start2, end2, start1, end1, "None"])'''
                                        


    # print(tuples)
    #print("----")

    if len(tuples) == 0:
        tuples = [[0, 0, 0, 0, "O"]]

    #Need to include all the entities without any relations!!
    return tuples, num_relations
    
    
def read_examples_from_file(args, data_dir, mode):
    if mode!= "test2":
        file_path = os.path.join(data_dir, "{}.txt".format(mode))
    else:
        file_path = os.path.join(args.output_dir, "test_predictions.txt")
        #print(file_path)
        
    guid_index = 1
    doc_examples = []
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        pred_labels = []
        original_labels = []
        corefs = []
        relations = []

        num_sent = 0

        total_relations = 0
        for line in f:            
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    #relations, num_relations = relation_tuples(pred_labels, original_labels, corefs)
                    #print(labels)
                    #print(relations)
                    #print(num_relations)
                    #total_relations += num_relations

                    all_entities = get_entities(list(labels))

                    entity_spans = []
                    entity_corefs = []
                    for i in range(len(labels)):
                        for j in range(i, len(labels), 1):
                            if j-i <= args.num_widths:#make this an argument later
                                entity_type = "O"
                                coref_type = 0
                                for (ent_type, start, end) in all_entities:
                                    if i==start and j==end:
                                        entity_type = ent_type
                                        coref_type = corefs[i]
                                        entity_corefs.append([i, j, int(corefs[i])])
                                '''if mode == "train":
                                    if entity_type!= "O":
                                        entity_spans.append([i, j, entity_type])
                                    else:
                                        if j-i == 1:
                                            entity_spans.append([i, j, entity_type])
                                elif mode != "train":'''
                                entity_spans.append([i, j, entity_type])
                                #entity_corefs.append([i, j, coref_type])
                    relations, num_relations = relation_tuples(pred_labels, original_labels, corefs, entity_corefs)
                    total_relations += num_relations                    
                    
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels, entity_spans=entity_spans, relations=relations, span_corefs=entity_corefs))
                    #if mode == "train" and num_sent > 500:
                    #if num_sent > 100:
                    #    break
                    num_sent += 1
                    #relation_tuples(labels, original_labels, corefs)
                    guid_index += 1
                    words = []
                    labels = []
                    pred_labels = []
                    original_labels = []
                    corefs = []
                    relations = []
                if line.startswith("-DOCSTART-"):
                    if len(examples) > 0:
                        #print(examples[-1].span_corefs)
                        #print(examples[-1].relations)
                        doc_examples.append(DocumentExample(examples))
                        #print(doc_examples[-1].inputSentences)
                        count_corefs = dict()
                        count_relations = dict()
                        for input1 in doc_examples[-1].inputSentences:
                            #print(input1.span_corefs)
                            #print(input1.relations)
                            coref_map = dict()
                            for coref1 in input1.span_corefs:
                                [start1, start2, number] = coref1
                                if number in count_corefs.keys():
                                    count_corefs[number] += 1
                                else:
                                    count_corefs[number] = 1
                                coref_map[str(start1)+"-"+str(start2)] = number
                            for rel in input1.relations:
                                [start1, end1, start2, end2, REL_TYPE] = rel
                                if REL_TYPE != "O" and end2-start2 <= args.num_widths and end1-start1 <= args.num_widths:
                                    number1 = coref_map[str(start1)+"-"+str(end1)]
                                    number2 = coref_map[str(start2)+"-"+str(end2)]
                                    if number1 in count_relations.keys():
                                        count_relations[number1] += 1
                                    else:
                                        count_relations[number1] = 1
                                    if number2 in count_relations.keys():
                                        count_relations[number2] +=1
                                    else:
                                        count_relations[number2] = 1
                                                                                                        
                                
                        #print(count_corefs)
                        #print(count_relations)
                        #print("=======")
                    examples = []
            else:
                #splits = line.split(" ")
                
                splits = line.split("\t")
                words.append(splits[0])
                corefs.append(int(splits[3].strip()))
                
                if len(splits) > 1:
                    #labels.append(splits[-1].replace("\n", ""))
                    #if len(splits[1].split("_")) > 1:
                    #labels.append(splits[1].split("_")[0]+"_"+splits[1].split("_")[1])
                    labels.append(splits[1].split("_")[0])                        
                    
                    if mode!= "test2": 
                        pred_labels.append(splits[1].split("_")[0])
                    else:
                        pred_labels.append(splits[3].strip().split("_")[0])
                    original_labels.append(splits[1])
                    #else:
                    #    labels.append("O")
                    #print(labels)
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
                    pred_labels.append("O")
                    original_labels.append("O")
        
        #creating labels for relation extraction -- find all spans and then for each pair!! start1, end1, start2, end2, label
        if words:
            relations, num_relations = relation_tuples(pred_labels, original_labels, corefs)
            total_relations += num_relations
            logger.info("%s ", words)
            logger.info("%s ", labels)
            logger.info("====")
            #exit()

            all_entities = get_entities(list(labels))
            
            entity_spans = []
            entity_corefs = []
            for i in range(len(labels)):
                for j in range(i, len(labels), 1):
                    if j-i <= args.num_widths:#make this an argument later
                        entity_type = "O"
                        for (ent_type, start, end) in all_entities:
                            if i==start and j == end:
                                entity_type = ent_type
                                entity_corefs.append([i, j, int(corefs[i])])
                                                                        
                        entity_spans.append([i, j, entity_type])
                                
            
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels, entity_spans = entity_spans, relations=relations, span_corefs=entity_corefs))
        if len(examples) > 0:
            doc_examples.append(DocumentExample(examples))
            
        print("Total relations")
        print(mode)
        print(total_relations)
        
        
    return doc_examples



def convert_examples_to_features(
    mode,
    documents,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    max_rel_length = 100,
):
    """ Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
    - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
    - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    max_sent_length = 128
    max_sent_number = 8
    max_coref_length = 100
    
    label_map = {label: i for i, label in enumerate(label_list)}

    rel_label_map = {label:i for i, label in enumerate(get_rel_labels())}

    features = []

    print(len(documents))
    
    for (doc_index, document) in enumerate(documents):
        examples = document.inputSentences

        #print(len(examples))
        
        tokens = []
        
        sub_documents = []

        start_index = 0
        end_index = 0
        last2 = 0
        last1 = 0
        for (ex_index, example) in enumerate(examples):
            '''if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))'''
            
            sent_tokens = []
            
            for i, (word, label) in enumerate(zip(example.words, example.labels)):
                word_tokens = tokenizer.tokenize(word)
                sent_tokens.extend(word_tokens)

            special_tokens_count = 3 if sep_token_extra else 2
            
            '''print("Last2\t"+str(last2))
            print("Last1\t"+str(last1))
            print("Current\t"+str(ex_index)+"\t"+str(len(sent_tokens)))
            print(str([start_index, end_index]))
            print("----")'''
                                                            

            if len(tokens) + len(sent_tokens) < max_seq_length - special_tokens_count and (end_index - start_index) < max_sent_number:
                tokens.extend(sent_tokens)
                end_index += 1
                last2 = last1
                last1 = len(sent_tokens)
                
            else:
                #print(len(tokens))
                #print(len(tokens)+len(sent_tokens))
                #add this to the feature and then continue again from 2 previous sentences
                sub_documents.append([start_index, end_index])
                ###start_index = end_index - 2 if end_index-2 > 0 else end_index
                start_index = end_index
                end_index += 1
                #need length of the last two sentences!!
                ###tokens = tokens[-(last2+last1):]
                tokens = []
                tokens.extend(sent_tokens)
        #if (len(tokens) > (last2+last1)):
        sub_documents.append([start_index, end_index])
                
        #subdocuments!!!
        for [start_index, end_index] in sub_documents:
            tokens = []
            label_ids = []
            new_index_map = dict()
            new_index=0
            
            for k, example in enumerate(examples[start_index: end_index]):
                for i, (word, label) in enumerate(zip(example.words, example.labels)):
                    word_tokens = tokenizer.tokenize(word)
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                    new_index_map[str(k)+"-"+str(i)] = new_index
                    new_index += len(word_tokens)
            special_tokens_count = 3 if sep_token_extra else 2

            #Should not be the case now!! So check
            #print("Inside\n")
            #print(str([start_index, end_index]))
            if len(tokens) > max_seq_length - special_tokens_count:
                #print(len(tokens))
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
                print("Still exceeded max_seq_length")
                exit()

            tokens += [sep_token]
            label_ids += [pad_token_label_id]
                
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
                    
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                    
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                
                segment_ids = [cls_token_segment_id] + segment_ids
                
                for old_index in new_index_map.keys():
                    new_index_map[old_index] += 1
                    
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                for old_index in new_index_map.keys():
                    new_index_map[old_index] += padding_length

            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                    
            span_label_map = {label: i for i, label in enumerate(convert_span_labels(label_list))}
            entity_tuples = []
            entity_label_ids = []
            entity_span_mask = []
            #entity_span_mask = [1 if mask_padding_with_zero else 0] * len(example.entity_spans)

            relation_tuples = []
            rel_label_ids = []

            coref_tuples = []
            coref_label_ids = []

            #keep account of the beginning and end span for each sentence!!
            sentence_span_indices = []

            #print(start_index)
            #print(end_index)
            
            for k, example in enumerate(examples[start_index: end_index]):
                start_index1 = len(entity_tuples)
                
                sent_entity_tuples = []
                sent_entity_label_ids = []
                sent_entity_span_mask = []

                
                for entity in example.entity_spans:
                    if new_index_map[str(k)+"-"+str(entity[0])] < max_seq_length and new_index_map[str(k)+"-"+str(entity[1])] < max_seq_length:
                        sent_entity_tuples.extend([[new_index_map[str(k)+"-"+str(entity[0])], new_index_map[str(k)+"-"+str(entity[1])]]])
                        sent_entity_label_ids.extend([span_label_map[entity[2]]])
                        sent_entity_span_mask.extend([1 if mask_padding_with_zero else 0])
                end_index1 = len(entity_tuples)-1

                if len(sent_entity_tuples) > max_sent_length * (7):
                    sent_entity_tuples = sent_entity_tuples[: max_sent_length * 7]
                    sent_entity_label_ids = sent_entity_label_ids[: max_sent_length * 7]
                    sent_entity_span_mask = sent_entity_span_mask[: max_sent_length * 7]
                sent_entity_span_padding_length = max_sent_length * (7) - len(sent_entity_tuples)
                sent_entity_tuples += [[0, 0]] * sent_entity_span_padding_length
                sent_entity_span_mask += [0 if mask_padding_with_zero else 1] * sent_entity_span_padding_length
                #print(sent_entity_tuples)
                
                #entity_label_ids += [pad_token_label_id] * entity_span_padding_length
                sent_entity_label_ids += [-1] * sent_entity_span_padding_length
                    
                sentence_span_indices.append([start_index1, end_index1])
                entity_tuples.append(sent_entity_tuples)
                entity_label_ids.append(sent_entity_label_ids)
                entity_span_mask.append(sent_entity_span_mask)

                sent_relation_tuples = []
                sent_rel_label_ids = []
                
                for relation in example.relations:
                #relation_tuples.extend([[new_index_map[relation[0]], new_index_map[relation[1]], new_index_map[relation[2]], new_index_map[relation[3]]]])
                    #rel_label_ids.extend([rel_label_map[relation[4]]])
                    #print(entity_tuples)
                    if [new_index_map[str(k)+"-"+str(relation[0])], new_index_map[str(k)+"-"+str(relation[1])]] in sent_entity_tuples and [new_index_map[str(k)+"-"+str(relation[2])], new_index_map[str(k)+"-"+str(relation[3])]] in sent_entity_tuples:
                        sent_relation_tuples.append([sent_entity_tuples.index([new_index_map[str(k)+"-"+str(relation[0])], new_index_map[str(k)+"-"+str(relation[1])]]), sent_entity_tuples.index([new_index_map[str(k)+"-"+str(relation[2])], new_index_map[str(k)+"-"+str(relation[3])]])])
                        sent_rel_label_ids.append(rel_label_map[relation[4]])
                        #rel_label_ids[entity_tuples.index([new_index_map[relation[0]], new_index_map[relation[1]]]), entity_tuples.index([new_index_map[relation[2]], new_index_map[relation[3]]])] = rel_label_map[relation[4]]
                    else:
                        print(mode)
                        print(str([new_index_map[str(k)+"-"+str(relation[2])], new_index_map[str(k)+"-"+str(relation[3])]]))
                        print(str([new_index_map[str(k)+"-"+str(relation[0])], new_index_map[str(k)+"-"+str(relation[1])]]))

                sent_rel_padding_length = max_rel_length - len(sent_relation_tuples)
                sent_relation_tuples += [[0, 0]] * sent_rel_padding_length
                sent_rel_label_ids += [-1] * sent_rel_padding_length

                relation_tuples.append(sent_relation_tuples)
                rel_label_ids.append(sent_rel_label_ids)

                #all_pairs
                sent_coref_tuples = []
                sent_coref_label_ids = []
                
                for coref in example.span_corefs:
                    if coref[2] > 0:
                        if [new_index_map[str(k)+"-"+str(coref[0])], new_index_map[str(k)+"-"+str(coref[1])]] in sent_entity_tuples:
                            sent_coref_tuples.append([sent_entity_tuples.index([new_index_map[str(k)+"-"+str(coref[0])], new_index_map[str(k)+"-"+str(coref[1])]])])
                            sent_coref_label_ids.append(coref[2])
                        else:
                            print(str([new_index_map[str(k)+"-"+str(coref[0])], new_index_map[str(k)+"-"+str(coref[1])]]))
                        
                sent_coref_padding_length = max_coref_length - len(sent_coref_tuples)
                sent_coref_tuples += [[0]] * sent_coref_padding_length
                sent_coref_label_ids += [-1] * sent_coref_padding_length
                
                #print(sent_coref_tuples)
                
                coref_tuples.append(sent_coref_tuples)
                coref_label_ids.append(sent_coref_label_ids)

            #print(entity_tuples)
            if len(entity_tuples) > max_sent_number :
                entity_tuples = entity_tuples[: max_sent_number]
                entity_label_ids = entity_label_ids[: max_sent_number]
                entity_span_mask = entity_span_mask[: max_sent_number]
            entity_span_padding_length = max_sent_number - len(entity_tuples)
            #entity_tuples += [[[[0, 0]] * max_sent_length * (7)] * entity_span_padding_length]
            entity_tuples += [[[0, 0]]*max_sent_length * 7] * entity_span_padding_length
            entity_span_mask += [[0 if mask_padding_with_zero else 1] * max_sent_length * 7] * entity_span_padding_length
            #entity_label_ids += [pad_token_label_id] * entity_span_padding_length
            entity_label_ids += [[-1] * max_sent_length * 7] * entity_span_padding_length

            #print(entity_tuples)
            #exit()
            
            #print(len(entity_tuples))
                                            
            rel_padding_length = max_sent_number - len(relation_tuples)
            relation_tuples += [[[0, 0]] * max_rel_length] * rel_padding_length
            rel_label_ids += [[-1] * max_rel_length] * rel_padding_length

            coref_padding_length = max_sent_number - len(coref_tuples)
            coref_tuples += [[[0]] * max_coref_length] * coref_padding_length
            coref_label_ids += [[-1] * max_coref_length] * coref_padding_length
            
            #span_index_padding_length = 50 - len(sentence_span_indices)
            #sentence_span_indices += [[-1, -1]] * span_index_padding_length

            '''if start_index < 5:
                logger.info("*** Example ***")
                #logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))'''

            features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, entity_tuples=entity_tuples, entity_span_mask=entity_span_mask, entity_label_ids=entity_label_ids, relation_tuples=relation_tuples, rel_label_ids=rel_label_ids, coref_tuples=coref_tuples, coref_label_ids=coref_label_ids)
            )
                
    '''for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []

        new_index_map = dict()

        new_index = 0
        for i, (word, label) in enumerate(zip(example.words, example.labels)):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            #mapping old_index to new_index
            new_index_map[i] = new_index
            new_index += len(word_tokens)

        relation_tuples = []
        rel_label_ids = []
        for relation in example.relations:
            relation_tuples.extend([new_index_map[relation[0]], new_index_map[relation[1]]      

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            for old_index in new_index_map.keys():
                if new_index_map[old_index] >= (max_seq_length - special_tokens_count):
                    new_index_map.remove(old_index)
            
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            
            segment_ids = [cls_token_segment_id] + segment_ids

            for old_index in new_index_map.keys():
                new_index_map[old_index] += 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids

            for old_index in new_index_map.keys():
                new_index_map[old_index] += padding_length
            
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length



        span_label_map = {label: i for i, label in enumerate(convert_span_labels(label_list))}
            
        entity_tuples = []
        entity_label_ids = []
        entity_span_mask = []
        #entity_span_mask = [1 if mask_padding_with_zero else 0] * len(example.entity_spans)
        
        for entity in example.entity_spans:
            if new_index_map[entity[0]] < max_seq_length and new_index_map[entity[1]] < max_seq_length:
                entity_tuples.extend([[new_index_map[entity[0]], new_index_map[entity[1]]]])
                entity_label_ids.extend([span_label_map[entity[2]]])
                entity_span_mask.extend([1 if mask_padding_with_zero else 0])'''
                
    return features


def convert_span_labels(labels):
    new_list = []
    for label in labels:
        if label.split("-")[-1] not in new_list:
            new_list.append(label.split("-")[-1])
    new_list.remove("O")
    new_list = ["O"] + new_list
    #if "PAD" not in new_list:
    #    new_list = ["PAD"] + new_list
    return new_list


def get_span_labels(path):
    if path:
        labels = []
        with open(path, "r") as f:
            #labels = f.read().splitlines()
            for tag in f.readlines():                                                                                                              
                if tag.strip().split("-")[-1] not in labels:                                                                                         
                    labels.append(tag.strip().split("-")[-1])
        if "O" not in labels:
            labels = ["O"] + labels
        #if "PAD" not in labels:
        #    labels = ["PAD"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
                                                                                        

def get_labels(path):
    if path:
        labels = []
        with open(path, "r") as f:
            labels = f.read().splitlines()
            '''for tag in f.readlines():
                if tag.strip().split("-")[-1] not in labels:
                    labels.append(tag.strip().split("-")[-1])'''
        if "O" not in labels:
            labels = ["O"] + labels
        #if "PAD" not in labels:
        #    labels = ["PAD"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def get_rel_labels():
    return ["O", "PHYS", "PER-SOC", "ORG-AFF", "ART", "PART-WHOLE", "GEN-AFF"]


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
    seq (list): sequence of labels.
    Returns:
    list: list of (chunk_type, chunk_start, chunk_end).
    Example:
    >>> from seqeval.metrics.sequence_labeling import get_entities
    >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
    >>> get_entities(seq)
    [('PER', 0, 1), ('LOC', 3, 3)]
    """
    
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
        
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if chunk == '':
            chunk = 'O'
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]
            
        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_
            
    return chunks



def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
    prev_tag: previous chunk tag.
    tag: current chunk tag.
    prev_type: previous type.
    type_: current type.
    Returns:
    chunk_end: boolean.
    """
    chunk_end = False
    
    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True
    
    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True
    
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_ and type_ != "_PAD_" and prev_type != "_PAD_":
        chunk_end = True
        
    return chunk_end



def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False
    
    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True
    
    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True
    
    if tag != 'O' and tag != '.' and prev_type != type_ and type_ != "_PAD_" and prev_type != "_PAD_":
        chunk_start = True
        
    return chunk_start




#methods adapted form allennlp

def get_device_of(tensor: torch.Tensor) -> int:
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()
                            

def get_range_vector(size: int, device: int) -> torch.Tensor:
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)

    

def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)
        
    offset_indices = indices + offsets
    
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target: torch.Tensor, indices: torch.LongTensor, flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
    if flattened_indices is None:
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))
    #print(flattened_indices)
    
    flattened_target = target.view(-1, target.size(-1))
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]

    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets
                                                                    


def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    span_starts, span_ends = spans.split(1, dim=-1)
    
    span_widths = span_ends - span_starts
    max_batch_span_width = span_widths.max().item() + 1
    
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(1, 1, -1)
    
    span_mask = (max_span_range_indices <= span_widths).float()
    raw_span_indices = span_ends - max_span_range_indices
    
    span_mask = span_mask * (raw_span_indices >= 0).float()
    span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()
    
    span_embeddings =  batched_index_select(target, span_indices)
    
    return span_embeddings, span_mask


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    return tensor.masked_fill((1 - mask).to(dtype=torch.bool), replace_with)



def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).long()



def get_pruned_gold_relations(relation_labels, top_span_indices, top_span_masks):
    """
    Loop over each slice and get the labels for the spans from that slice.
    All labels are offset by 1 so that the "null" label gets class zero. This is the desired
    behavior for the softmax. Labels corresponding to masked relations keep the label -1, which
    the softmax loss ignores.
    """

    relations = []

    for sliced, ixs, top_span_mask in zip(relation_labels, top_span_indices, top_span_masks.bool()):
        entry = sliced[ixs][:, ixs].unsqueeze(0)
        top_span_mask = top_span_mask.unsqueeze(-1)
        mask_entry = top_span_mask & top_span_mask.transpose(0, 1).unsqueeze(0)
        #exit()
        #entry[mask_entry] += 1
        entry[~mask_entry] = -1
        relations.append(entry)

    return torch.cat(relations, dim=0)


def get_pruned_gold_mentions(entity_labels, top_span_indices, top_span_masks):
    entities = []

    for sliced, ixs, top_span_mask in zip(entity_labels, top_span_indices, top_span_masks):
        entry = sliced[ixs]
        entry[~top_span_mask] = -1
        entities.append(entry)

    return torch.cat(entities, dim=0)


def get_pruned_gold_coref_relations(coref_labels, top_span_indices, top_span_mask):
    #print(coref_labels.size())
    #print(top_span_indices)
    top_span_indices = top_span_indices
    entry = coref_labels[top_span_indices][:, top_span_indices].unsqueeze(0)
    top_span_mask = top_span_mask.bool().unsqueeze(-1)
    mask_entry = top_span_mask & top_span_mask.transpose(0, 1).unsqueeze(0)
    #print(entry.size())
    #print(mask_entry.size())
    #print("=====")

    entry[~mask_entry] = -1

    return entry


def bucket_values(
        distances: torch.Tensor, num_identity_buckets: int = 10, num_total_buckets: int = 30
) -> torch.Tensor:
    """
    Places the given values (designed for distances) into `num_total_buckets`semi-logscale
    buckets, with `num_identity_buckets` of these capturing single values.
    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    # Parameters
    distances : `torch.Tensor`, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: int, optional (default = 4).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : int, (default = 10)
        The total number of buckets to bucket values into.
    # Returns
    `torch.Tensor`
        A tensor of the same shape as the input, containing the indices of the buckets
        the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (
        num_identity_buckets - 1
    )
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


def batched_span_select2(target, span_starts, span_ends):
    #print(span_ends.size())
    #print(span_starts.size())
    span_widths = (span_ends - span_starts).clamp(min=0)
    
    max_batch_span_width = span_widths.max().item() + 1

    #print(max_batch_span_width)
    max_batch_span_width = min(max_batch_span_width, 30)
    #print(max_batch_span_width)
    #print("======")
    
    #print(span_widths.size())
    #num_batches, num_candidates, num_candidates
    #print(max_batch_span_width)
    
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, 1, -1)
    #print(max_span_range_indices.size())
    span_mask = max_span_range_indices <= span_widths.unsqueeze(-1)
    #print(span_mask.size())
    raw_span_indices = span_ends.unsqueeze(-1) - max_span_range_indices
    #print(raw_span_indices.size())
    span_mask = span_mask & (raw_span_indices >= 0)
    #print(span_mask.size())
    
    span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()
    
    span_embeddings = batched_index_select(target, span_indices.view(target.size(0), -1)).view(span_mask.size(0), span_mask.size(1), 1, span_mask.size(3), -1)
    #print(target.size())
    #print(span_embeddings.size())
    
    return span_embeddings, span_mask
    

#read_examples_from_file("./data/", "test")

'''seq = ["B-PER", "_PAD_", "_PAD_", "E-PER", "S-GPE"]
print(seq)
print(get_entities(seq))

seq = ["B-PER", "E-PER", "_PAD_", "S-GPE"]
print(seq)
print(get_entities(seq))'''
