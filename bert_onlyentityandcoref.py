import numpy as np
import torch
import torch.nn as nn
#from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel, BertModel
from utils_ner_relation_coref_span import get_rel_labels, batched_span_select, batched_index_select, replace_masked_values, get_mask_from_sequence_lengths, flatten_and_batch_shift_indices, get_pruned_gold_relations, get_pruned_gold_coref_relations, get_range_vector, bucket_values, get_pruned_gold_mentions, batched_span_select2

from typing import Optional

#import torch.nn.functional.gelu as gelu

'''ACT2FN = {"relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}

try:
    ACT2FN["gelu"] = torch.nn.functional.gelu
except AttributeError:
    ACT2FN["gelu"] = gelu'''


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        #self.num_rel_labels = config.num_rel_labels
        self.num_rel_labels = len(get_rel_labels())
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
                
        #self.mention_dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.entity_dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.coref_dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.rel_dropout = nn.Dropout(config.hidden_dropout_prob)
                        
        #change this to be the parameters
        #self.width_embeddings = nn.Embedding(50, config.hidden_size)
        #self.position_embeddings = nn.Embedding(100, config.hidden_size)
        self.width_embeddings = nn.Embedding(50, 200)
        self.position_embeddings = nn.Embedding(100, config.hidden_size)
                        
        #self.classifier = nn.Linear(3*config.hidden_size, config.num_labels)
        self.mention_weights = nn.Linear(2*config.hidden_size+200, config.hidden_size)
        #self.mention_gelu = gelu()

        self.mention_type_weights = nn.Linear(2*config.hidden_size+200, config.hidden_size)
        #self.mention_type_gelu = gelu()                

        self.entity_weights = nn.Linear(3*config.hidden_size, config.hidden_size)
        
        self.mention_classifier = nn.Linear(config.hidden_size, 2)
        self.mention_type_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.gelu = nn.GELU()
        
        self.source_weights = nn.Linear(8*config.hidden_size, config.hidden_size)
        #self.target_weights = nn.Linear(2*config.hidden_size, config.hidden_size)
        #self.rel_gelu = gelu()
        self.rel_classifier = nn.Linear(config.hidden_size, self.num_rel_labels)
        self.freq_reps = nn.Embedding(20, config.hidden_size)

        #self.coref_gelu = gelu()
        self.coref_weights = nn.Linear(7*config.hidden_size+200*3, config.hidden_size)
        self.coref_classifier = nn.Linear(config.hidden_size, 2)
        #self.coref_position_embeddings = nn.Embedding(100, config.hidden_size)
        
        self.init_weights()
        
    #[DOCS]
    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)            
        
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            entity_indices=None,
            entity_span_mask=None,
            entity_labels=None,
            #rel_indices=None,
            rel_labels=None,
            sent_indices=None,
            coref_labels=None,
            training=None,
    ):
        """n the configuration (:class:`~transformers.BertConfig`) and inputs:
        sequence_length, sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
        Examples::
        from transformers import BertTokenizer, BertForTokenClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
        """
        '''entity_indices = entity_indices.squeeze(0)
        entity_labels = entity_labels.squeeze(0)
        entity_span_mask = entity_span_mask.squeeze(0)
        rel_labels = rel_labels.squeeze(0)
        coref_labels = coref_labels.squeeze(0)'''
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        #####sequence_output = self.dropout(sequence_output)

        #Sequence classification
        #logits = self.classifier(sequence_output)

        #change here for the span representations!!!
        
        #span_embeddings, span_mask = batched_span_select(sequence_output, entity_indices)

        span_starts, span_ends = [index.squeeze(-1) for index in entity_indices.split(1, dim=-1)]

        if entity_span_mask is not None:
            span_starts = span_starts * entity_span_mask
            span_ends = span_ends * entity_span_mask

        #span_starts = span_starts.view(sequence_output.size(0), -1)
        #span_ends = span_ends.view(sequence_output.size(0), -1)
        '''print("Part1---")
        print(sequence_output.size())
        print(entity_indices.size())
        print(span_starts.size())
        print(span_ends.size())'''
            
        start_embeddings = batched_index_select(sequence_output, span_starts.view(sequence_output.size(0), -1)).view(span_starts.size(0), span_starts.size(1), -1)
        end_embeddings = batched_index_select(sequence_output, span_ends.view(sequence_output.size(0), -1)).view(span_starts.size(0), span_starts.size(1), -1)

        span_widths = span_ends - span_starts

        #print(start_embeddings.size())
        #print(end_embeddings.size())

        torch.set_printoptions(profile="full")        
        
        #print(span_widths)
        
        ##width_embeddings = self.dropout(self.width_embeddings(span_widths).to(start_embeddings.device))
        width_embeddings = self.width_embeddings(span_widths).to(start_embeddings.device)
        #print(combined_tensors.size())
        #print("Part2----")
        #print(start_embeddings.size())
        #print(end_embeddings.size())
        #print(width_embeddings.size())
        
        combined_tensors = torch.cat([start_embeddings, end_embeddings], dim=-1)
        #print(combined_tensors.size())
        
        combined_tensors = torch.cat([combined_tensors, width_embeddings], dim=-1)

        if entity_span_mask is not None:
            combined_tensors = combined_tensors * entity_span_mask.unsqueeze(-1).float()

        #sometimes uses the width of the embeddings as an additional feature
        #use that as well!!

        mention_features = self.gelu(self.mention_weights(combined_tensors))

        #mention_features = self.mention_dropout(mention_features)
        
        #Entity score
        logits = self.mention_classifier(mention_features)
        
        softmax_logits = nn.functional.softmax(logits.view(-1, logits.size(2)), dim=1)
        #print(softmax_logits)
        
        softmax_logits = softmax_logits.view(logits.size(0), logits.size(1), logits.size(2))

        softmax_logits = softmax_logits * entity_span_mask.unsqueeze(-1).float()

        #print(entity_labels.size())
        #print(softmax_logits.size())
        #print(entity_labels == 1)
        if training[0] == 1:
            softmax_logits[entity_labels == 1] = 1.0
        
        #print(softmax_logits)
        
        ####At this point, only keep the real entity spans

        #relation classification

        #hardcode the number of spans to keep? Let's keep 30 for now!!
        num_items = combined_tensors.size(1)

        num_items_to_keep = 10
        #num_items_to_keep = max(torch.sum(softmax_logits[:, :, 1:] > 0.5).item(), 1)

        #print(num_items_to_keep)
        
        batch_size = entity_span_mask.size()[0]
        
        num_items_to_keep = num_items_to_keep * torch.ones([batch_size], dtype=torch.long, device=entity_span_mask.device)
        num_good_items_to_keep = num_items_to_keep * torch.ones([batch_size], dtype=torch.long, device=entity_span_mask.device)

        #print(softmax_logits[:, :, 1:].sum(dim=-1))

        num_items_to_keep = torch.sum((((softmax_logits[:, :, 1:].sum(dim=-1) > 0.50))), dim=1).squeeze()
        #print(num_items_to_keep)
        
        num_items_to_keep = torch.max(num_good_items_to_keep, num_items_to_keep)
        ###num_items_to_keep = torch.min(num_good_items_to_keep, num_items_to_keep)
        
        #print(num_items_to_keep)
        #num_items_to_keep = num_good_items_to_keep

        num_items_to_keep = num_items_to_keep.clamp(1, 23)
        #print(num_items_to_keep)
        
        #max_items_to_keep = max(num_items_to_keep.max().item(), 1)
        max_items_to_keep = max(num_items_to_keep.max().item(), 1)
        
        #scores, _ = logits.max(dim=-1)
        #scores = scores.unsqueeze(-1)

        #print(max_items_to_keep)

        new_scores, _ = logits[:, :, 1:].max(dim=-1)
        new_scores = new_scores.unsqueeze(-1)
        
        new_logits = replace_masked_values(new_scores, entity_span_mask.unsqueeze(-1), -1e3)

        #new_logits = torch.where(entity_labels > 0, torch.zeros_like(entity_labels, dtype=torch.float), -1e20 * torch.ones_like(entity_labels, dtype=torch.float))

        #removing the 'O' spans! But at the time of training these are present ~~~~
 
        _, top_indices = new_logits.topk(max_items_to_keep, 1)
        
        #print(sent_indices)
        
        #_, top_indices1 = new_logits1.topk(max_items_to_keep1, 1)

        #(500*7 = 3500 === now we want to make sure that same proportion are selected from (7*25) ~ 20 buckets)
        #print(top_indices)
        ##exit()
        
        top_indices_mask = get_mask_from_sequence_lengths(num_items_to_keep, max_items_to_keep)

        top_indices_mask = top_indices_mask.bool()
        
        # Shape: (batch_size, max_num_items_to_keep)

        #print("Before---")
        #print(top_indices.size())
        top_indices = top_indices.squeeze(-1)
        #print(top_indices.size())
        
        fill_value, _ = top_indices.max(dim=1)
        fill_value = fill_value.unsqueeze(-1)
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = torch.where(top_indices_mask, top_indices, fill_value)
        top_indices, _ = torch.sort(top_indices, 1)

        #print(top_indices)

        flat_top_indices = flatten_and_batch_shift_indices(top_indices, num_items)

        #print(flat_top_indices)
        
        sequence_mask = batched_index_select(entity_span_mask.unsqueeze(-1), top_indices, flat_top_indices)
        #exit()
        
        sequence_mask = sequence_mask.squeeze(-1).bool()
        top_mask = top_indices_mask & sequence_mask
        top_mask = top_mask.long()

        #print(top_indices)
        #print(top_mask)

        #coref_positions = (top_mask.cumsum(0) * top_mask).flatten()
        combined_tensors = batched_index_select(combined_tensors, top_indices, flat_top_indices)
        #Entity typing part

        mention_type_features = self.gelu(self.mention_type_weights(combined_tensors))
        #mention_type_features = self.entity_dropout(mention_type_features)
        
        #Entity score
        type_logits = self.mention_type_classifier(mention_type_features)
        #print(entity_labels.size())
        #print(top_indices.size())
        #print(top_mask.size())
        entity_type_labels = get_pruned_gold_mentions(entity_labels, top_indices, top_mask)
        #entity_type_labels = entity_labels
        

        #combined_tensors = batched_index_select(combined_tensors, top_indices, flat_top_indices)
                
        #batch_size, 20, hidden_dim_size
        num_coref_candidates = combined_tensors.size(0)*combined_tensors.size(1)

        #print(combined_tensors.size())
        
        
        #span_starts, span_ends = [index.squeeze(-1) for index in entity_indices.split(1, dim=-1)]
        #print(span_starts.size())
        #print(top_indices)
        #print(span_starts)
        span_starts = batched_index_select(span_starts.unsqueeze(-1), top_indices, flat_top_indices)
        #span_starts_flattened = flatten_and_batch_shift_indices(span_starts.squeeze(-1), num_items)
        span_starts_flattened = span_starts.squeeze(-1).flatten()
 
        #print(span_starts_flattened)
        #print(span_starts_flattened.size())

        span_ends = batched_index_select(span_ends.unsqueeze(-1), top_indices, flat_top_indices)
        span_ends_flattened = span_ends.squeeze(-1).flatten()
        #print(span_ends_flattened)
        
        #coref_positions = span_starts_flattened.unsqueeze(1)

        #print(coref_positions)
        
        #coref_positions = torch.arange(num_coref_candidates)
        #coref_positions = coref_positions.unsqueeze(0)
        #coref_positions = span_ends.unsqueeze(0)
        #coref_positions2 = coref_positions.transpose(1, 0)
        
        coref_position_2d = (span_starts_flattened.unsqueeze(0) - span_ends_flattened.unsqueeze(1)).to(combined_tensors.device)
        #print(coref_position_2d)
        #exit()
        coref_position_2d = torch.max(coref_position_2d, coref_position_2d.transpose(1, 0))
        coref_position_2d[coref_position_2d < 0] = 0

        coref_position_2d = bucket_values(coref_position_2d)
        
        #print(coref_position_2d)
        #exit()
        
        #coref_position_2d = coref_position_2d.repeat(num_coref_candidates, 1).to(combined_tensors.device)
        
        batch_coref_position_embeddings = self.position_embeddings(coref_position_2d).to(combined_tensors.device)
        ##batch_coref_position_embeddings = self.dropout(self.position_embeddings(coref_position_2d).to(combined_tensors.device))
        num_coref_candidates = combined_tensors.size(0)*combined_tensors.size(1)
                
        corefembeddings_1 = combined_tensors.view(-1, combined_tensors.size(2)).unsqueeze(1)
        corefembeddings_1_tiled = corefembeddings_1.repeat(1, num_coref_candidates, 1 )

        corefembeddings_2 = combined_tensors.view(-1, combined_tensors.size(2)).unsqueeze(0)
        corefembeddings_2_tiled = corefembeddings_2.repeat(num_coref_candidates, 1, 1)
                        
        corefsimilarity_embeddings = corefembeddings_1 * corefembeddings_2
        
        #corefpair_embeddings_list = [corefembeddings_1_tiled, corefembeddings_2_tiled, corefsimilarity_embeddings, batch_coref_position_embeddings, batch_position_embeddings]
        corefpair_embeddings_list = [corefembeddings_1_tiled, corefembeddings_2_tiled, corefsimilarity_embeddings, batch_coref_position_embeddings]
        #corefpair_embeddings_list = [corefembeddings_1_tiled, corefembeddings_2_tiled, corefsimilarity_embeddings]
        
        corefpair_embeddings = torch.cat(corefpair_embeddings_list, dim=2)

        #print(corefpair_embeddings.size())

        #print(flat_top_indices.size())
        #print(top_mask.size())
        #print("~~~~~")
        
        new_coref_labels = get_pruned_gold_coref_relations(coref_labels[:, :-1], flat_top_indices, top_mask.flatten())
        #print(new_coref_labels.size())
        
        #print(coref_labels)
        #exit()
        #print(new_coref_labels.size())

        #new_coref_labels = torch.cat([new_coref_labels.squeeze(0), coref_labels[flat_top_indices, -1].unsqueeze(-1)], dim=1)

        #new_coref_labels = new_coref_labels.clamp(0, 1)
        #print(new_coref_labels)
        
        #entity_reps = (combined_tensors.view(-1, combined_tensors.size()[2]).unsqueeze(0) * new_coref_labels.unsqueeze(-1)).sum(dim=1).squeeze().view(combined_tensors.size()[0], combined_tensors.size()[1], -1)
        #entity_reps = entity_reps/10.0

        
        #print((coref_labels>0).sum(dim=1))
        ##entity_reps = self.freq_reps((coref_labels>0).sum(dim=1))
        ##entity_reps = entity_reps.view(combined_tensors.size()[0], combined_tensors.size()[1], -1)
        #print(entity_reps.size())
        #exit()
        #print(combined_tensors.size())
        #exit()

        #print("========")
        
        coref_projected = self.gelu(self.coref_weights(corefpair_embeddings))

        #coref_projected = self.coref_dropout(coref_projected)
        
        coref_scores = self.coref_classifier(coref_projected)
        coref_logits = coref_scores
        #print(coref_logits.size())
        '''coref_scores = torch.cat([coref_scores, torch.zeros([coref_scores.size(0), 1, coref_scores.size(2)], dtype=torch.float, device=coref_scores.device)], dim=1)
        coref_mask = top_mask.flatten().unsqueeze(1) & top_mask.flatten().unsqueeze(1).transpose(0, 1)
        coref_mask = torch.cat([coref_mask, torch.ones([coref_scores.size(0), 1], dtype=torch.long, device=coref_scores.device)], dim=1)

        coref_mask[new_coref_labels == -1] = 0

        new_coref_scores = coref_scores * coref_mask.unsqueeze(-1)
        new_coref_scores[new_coref_scores == 0] = -10
        new_coref_scores[:, -1, :] = 0.
        
        print(coref_scores.size())
        print(coref_mask.size())

        #How to keep the 10 nearest?

        coref_logits = nn.functional.softmax(new_coref_scores.view(-1, new_coref_scores.size(1)), dim=1)
        print(coref_logits.size())
        print(new_coref_labels.size())'''
        #mult = coref_logits.view(-1, 1) * new_coref_labels.view(-1).unsqueeze(-1)
        #print(mult.size())
        
        #coref_logits = coref_scores

        #relation part commented from here...
        #print(combined_tensors.size())
        '''num_candidates = combined_tensors.size(1)

        #print("~~~~~~")
        
        #print(span_starts.size())
        #print(span_ends.size())
        
        positions = span_starts

        context_positions11 = (span_starts-1).clamp(min=0)
        
        positions2 = span_ends.transpose(2, 1)
        
        context_positions12 = (span_ends.transpose(2, 1)+1)

        #print("~~~~")
        #print(span_starts.size())
        #print(span_ends.size())
                        
        
        context_positions21 = (span_ends.transpose(2, 1)+1)
        context_positions22 = (span_starts-1).clamp(min=0)

        #print("=====")
        #print(context_positions11.size())
        #print(context_positions12.size())
        
        #1st one:
        partOne_output, partOne_mask = batched_span_select2(sequence_output, context_positions12, context_positions11)
        #print(partOne_output.size())
        
        #partOne_embeddings = partOne_output[:, :, :, :-1]
        #print(partOne_embeddings.size())
        #print(partOne_mask.size())

        partOne_max_pooled_features = partOne_output * partOne_mask.unsqueeze(-1)
        partOne_max_pooled_features = partOne_max_pooled_features.max(dim=3)[0]
        #print(partOne_max_pooled_features.size())

        #print("====")
        #print(context_positions22.size())
        #print(context_positions12.size())
                        
        partTwo_output, partTwo_mask = batched_span_select2(sequence_output, context_positions21, context_positions22)
        partTwo_max_pooled_features = partTwo_output * partTwo_mask.unsqueeze(-1)
        partTwo_max_pooled_features = partTwo_max_pooled_features.max(dim=3)[0]    
        
        #exit()
        
        context_embed = ((context_positions11 - context_positions12).clamp(0, 1)).unsqueeze(-1) * partOne_max_pooled_features + ((context_positions22 - context_positions21).clamp(0, 1)).unsqueeze(-1) * partTwo_max_pooled_features

        #context_embed = partOne_max_pooled_features

        #print(context_embed.size())
        #exit()
        
        #print(sequence_output[:, context_positions11.flatten()].size())
        #print(sequence_output[:, context_positions11.flatten()].view(span_starts.size(0), span_starts.size(1), span_starts.size(2), -1).size())
        #print(context_positions11.size())
        #print(context_positions12.size())
        #print(((context_positions11 - context_positions12).clamp(0, 1)).unsqueeze(-1).size())
        
        #context_embed = torch.cat([sequence_output[:, context_positions11.flatten()].view(span_starts.size(0), span_starts.size(1), span_starts.size(2), -1), sequence_output[:, context_positions12.flatten()].view(positions2.size(0), positions.size(1), positions.size(2), -1)], dim=3) * ((context_positions11 - context_positions12).clamp(0, 1)).unsqueeze(-1) + torch.cat([sequence_output[:, context_positions22.flatten()].view(positions2.size(0), positions.size(1), positions.size(2), -1), sequence_output[:, context_positions21.flatten()].view(span_starts.size(0), span_starts.size(1), span_starts.size(2), -1)], dim=3) * ((context_positions11 - context_positions12).clamp(0, 1)).unsqueeze(-1)

        #context_embed = context_embed.view((combined_tensors.size(0), combined_tensors.size(1), combined_tensors.size(1), -1))

        #print(context_embed.size())
        
        #positions = torch.arange(num_candidates).unsqueeze(1)
        #positions = (top_mask.cumsum(1)*top_mask).unsqueeze(1)
        #positions2 = positions.transpose(1, 0)
        #print(positions)
        #print(positions2)
        
        position_2d = (positions - positions2).to(combined_tensors.device)
        #print(position_2d)
        position_2d = torch.max(position_2d, position_2d.transpose(2, 1))
        #print(position_2d)
        position_2d[position_2d < 0] = 0

        position_2d = bucket_values(position_2d)
        
        #position_2d = torch.remainder(position_2d, 100)

        #exit()
        #position_2d = position_2d.repeat(combined_tensors.size(0), 1, 1).to(combined_tensors.device)
        batch_position_embeddings = self.dropout(self.position_embeddings(position_2d).to(combined_tensors.device))

        #combined_tensors = torch.cat((combined_tensors, entity_reps), 2)

        #print(combined_tensors.size())
        
        embeddings_1_expanded = combined_tensors.unsqueeze(2)
        embeddings_1_tiled = embeddings_1_expanded.repeat(1, 1, num_candidates, 1)
        #span_1_mask = entity_span_mask.unsqueeze(2)
        
        embeddings_2_expanded = combined_tensors.unsqueeze(1)
        embeddings_2_tiled = embeddings_2_expanded.repeat(1, num_candidates, 1, 1)
        #span_2_mask = entity_span_mask.unsqueeze(1)
        
        #####similarity_embeddings = embeddings_1_expanded * embeddings_2_expanded
        #pair_spans = span_1_mask * span_2_mask

        #pair_embeddings_list = [embeddings_1_tiled, embeddings_2_tiled, similarity_embeddings, batch_position_embeddings]
        pair_embeddings_list = [embeddings_1_tiled, embeddings_2_tiled, batch_position_embeddings, context_embed]
        pair_embeddings = torch.cat(pair_embeddings_list, dim=3)

        #print(pair_embeddings.size())

        rel_labels = get_pruned_gold_relations(rel_labels, top_indices, top_mask)

        relations_projected = self.gelu(self.source_weights(pair_embeddings))

        #relations_projected = self.rel_dropout(relations_projected)
                
        relation_scores = self.rel_classifier(relations_projected)

        rel_logits = relation_scores'''

        #print("=======")
        torch.set_printoptions(profile="full")
        #print(rel_labels)
        #exit()
                
        #print(rel_logits)
        #print(rel_labels)
        #print(rel_labels.size())
        
        '''indices = np.nonzero(rel_labels)
        
        #print(indices.size()[0])
        indices0 = indices[np.arange(indices.size()[0]), 0]
        indices1 = indices[np.arange(indices.size()[0]), 1]
        
        part = (rel_indices[indices0, indices1, :])
        rel_labels = rel_labels[indices0, indices1]
        
        hidden_states = sequence_output
        features0 = hidden_states[indices0, part[:, 0], :]
        features1 = hidden_states[indices0, part[:, 1], :]
        features2 = hidden_states[indices0, part[:, 2], :]
        features3 = hidden_states[indices0, part[:, 3], :]
        
        src_feature = torch.cat((features0, features1), 1)
        trg_feature = torch.cat((features2, features3), 1)
                                                                                
        rel_feat = self.rel_relu(self.source_weights(src_feature) + self.target_weights(trg_feature))

        rel_logits = self.rel_classifier(rel_feat)

        rel_feat_reversed = self.rel_relu(self.source_weights(trg_feature) + self.target_weights(src_feature))
        rel_logits_reversed = self.rel_classifier(rel_feat_reversed)'''

        #logits = type_logits
        #coref_logits = type_logits
        #top_indices = None
        #top_mask = None
        #flat_top_indices = None
        
        #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        #outputs = (logits, rel_logits, coref_logits, top_indices, top_mask, flat_top_indices) + outputs[2:]
        outputs = (logits, coref_logits, coref_logits, type_logits, top_indices, top_mask, flat_top_indices) + outputs[2:]
        
        if labels is not None and rel_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                '''active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = entity_labels.view(-1)[active_loss]
                loss1 = loss_fct(active_logits, active_labels)'''
                loss1 = loss_fct(logits.view(-1, 2), entity_labels.clamp(0, 1).view(-1))
                
                #if rel_labels.size()[0] > 0:
                loss2 = loss_fct(coref_logits.view(-1, 2), new_coref_labels.view(-1))
                #loss3 = loss_fct(coref_logits.view(-1, 1), coref_labels.view(-1))
                #loss_part = (coref_logits.view(-1, 1) * new_coref_labels.view(-1)).sum(dim=1)
                #loss_part = (coref_logits.view(-1, 1) * new_coref_labels.view(-1).unsqueeze(-1)).sum(dim=1)
                #loss2 = torch.log(loss_part + 1e-5).mean()
                loss3 = loss2
                loss4 = loss_fct(type_logits.view(-1, self.num_labels), entity_type_labels.view(-1))
                #loss2 = loss3 = loss4
                #else:
                #loss2 = loss1
                ###loss3 = loss1
                
            else:
                loss1 = loss_fct(logits.view(-1, 2), entity_labels.clamp(0, 1).view(-1))
                loss4 =loss_fct(type_logits.view(-1, self.num_labels), entity_type_labels.view(-1))
                
                if rel_labels.size()[0] > 0:
                    loss2 = loss_fct(rel_logits.view(-1, self.num_rel_labels), rel_labels.view(-1))
                    loss3 = loss_fct(coref_logits.view(-1, 2), coref_labels.view(-1))
                else:
                    loss2 = loss1
                    loss3 = loss1
            outputs = (loss1, loss2, loss3, loss4) + outputs
            #outputs = (loss1, loss1, loss1) + outputs
            
        return outputs  # (loss), scores, rel_scores, (hidden_states), (attentions)
