#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False, ignore_scoring_margin=False,conve_drop0=False, conve_drop1=False,conve_drop2=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        if ignore_scoring_margin:
            self.gamma = nn.Parameter(
                torch.Tensor([0.0]), 
                requires_grad=False
            )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'RESCAL':
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        else:
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['RESCAL', 'TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE','ConvE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')


        if model_name == 'ConvE':
            # key: embedding dimension, value: hidden size of fc layer
            # notes that 9728 is special for hidden_dim=200
            fc_project_dict = {100:3648, 200:9728, 500:27968, 1000:58368, 2000:119168}
            self.inp_drop = nn.Dropout(conve_drop0)
            self.hidden_drop      = nn.Dropout(conve_drop1)
            self.feature_map_drop = nn.Dropout2d(conve_drop2)
            self.emb_dim1         = 20
            self.emb_dim2         = self.hidden_dim // self.emb_dim1
            self.conv1            = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=False)
            self.bn0              = nn.BatchNorm2d(1)
            self.bn1              = nn.BatchNorm2d(32)
            self.bn2              = nn.BatchNorm1d(self.hidden_dim)
            self.fc               = nn.Linear(fc_project_dict[self.hidden_dim], self.hidden_dim)
            self.bias             = nn.Parameter(torch.zeros(self.nentity))


    def forward(self, sample,mode='single',ss_torch_index=None):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single' or mode == 'single_inverse':
            batch_size, negative_sample_size = sample.size(0), 1
            if self.model_name == 'ConvE' and mode == 'single_inverse':
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:, 0]
                ).unsqueeze(1)

            else:
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:, 0]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1))
            if ss_torch_index == None:
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
                head = head.view(batch_size, negative_sample_size, -1)
            else:
                head = head.unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1))
            if ss_torch_index == None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                tail = tail.view(batch_size, negative_sample_size, -1)
            else:
                tail = tail.unsqueeze(1)
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'RESCAL': self.RESCAL,
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'ConvE': self.ConvE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail,ss_torch_index, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail,ss_torch_index, mode):
        if mode == 'head-batch':
            if ss_torch_index != None:
                score = head + (relation - tail)[ss_torch_index]
            else:
                score = head + (relation - tail)
        elif mode == 'tail-batch':
            if ss_torch_index != None:
                score = (head + relation)[ss_torch_index] - tail
            else:
                score = (head + relation) - tail
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def RESCAL(self, head, relation, tail, mode):
        score = (head @ relation.squeeze(1)) * tail
        score = self.gamma.item() + score.sum(dim = 2)
        return score

    def DistMult(self, head, relation, tail,ss_torch_index, mode):
        if mode == 'head-batch':
            if ss_torch_index != None:
                score = head * (relation * tail)[ss_torch_index]
            else:
                score = head * (relation * tail)
        elif mode == 'tail-batch':
            if ss_torch_index != None:
                score = (head * relation)[ss_torch_index] * tail
            else:
                score = (head * relation) * tail
        else:
            score = (head * relation) * tail
        score = self.gamma.item() + score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail,ss_torch_index, mode):

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)



        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            if ss_torch_index != None:
                score = re_head * re_score[ss_torch_index] + im_head * im_score[ss_torch_index]
            else:
                score = re_head * re_score + im_head * im_score
        elif mode == 'tail-batch':
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            if ss_torch_index != None:
                score = re_score[ss_torch_index] * re_tail + im_score[ss_torch_index] * im_tail
            else:
                score = re_score * re_tail + im_score * im_tail
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail
        score = self.gamma.item() + score.sum(dim = 2)
        return score

    def ConvE(self, head, relation, tail,ss_torch_index, mode):
        '''
            f(h,r,t) = < conv(h,r), t >
        '''

        if mode == 'head-batch':
            head_temp = tail.view(-1, 1, self.emb_dim1, self.emb_dim2)
        else:
            head_temp = head.view(-1, 1, self.emb_dim1, self.emb_dim2)
        relation = relation.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([head_temp, relation], 2)
        # stacked_inputs = self.bn0(stacked_inputs)

        # φ(h,r) -> x
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(stacked_inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)  # [B, dim]

        if mode == 'single' or mode == 'single_inverse':
            self.regularizeOnPositiveSamples([head_temp, relation, tail], x)
        if mode == 'single':
            score = torch.bmm(x.unsqueeze(1), tail.transpose(1, 2)).squeeze(1)
        elif ss_torch_index != None :
            if mode == 'head-batch':
                score = torch.bmm(x[ss_torch_index].unsqueeze(1), head.transpose(1, 2)).squeeze(1)
            else:
                score = torch.bmm(x[ss_torch_index].unsqueeze(1), tail.transpose(1, 2)).squeeze(1)
        else:
            if mode == 'head-batch':
                score = torch.bmm(x.unsqueeze(1), head.transpose(1, 2)).squeeze(1)
            else:
                score = torch.bmm(x.unsqueeze(1), tail.transpose(1, 2)).squeeze(1)

        return score
    def regularizeOnPositiveSamples(self, embeddings, queries):
        '''
        available regularizer:
            FRO / NUC / DURA / None
        inputs:
            embeddings: heads, relations, tails
            queries:    combination of heads and relations
        '''
        self.regu = 0
        self.regu_weight = 0.00979
        self.regularizer = 'DURA'
        [heads, relations, tails] = embeddings

        if self.regularizer == 'FRO':
            # squared L2 norm
            self.regu += heads.norm(p=2) ** 2 / heads.shape[0]
            self.regu += tails.norm(p=2) ** 2 / tails.shape[0]
            self.regu += relations.norm(p=2) ** 2 / relations.shape[0]

        elif self.regularizer == 'NUC':
            # nuclear 3-norm
            self.regu += heads.norm(p=3) ** 3 / heads.shape[0]
            self.regu += tails.norm(p=3) ** 3 / tails.shape[0]
            self.regu += relations.norm(p=3) ** 3 / relations.shape[0]

        elif self.regularizer == 'DURA':
            # duality-induced regularizer for tensor decomposition models
            # regu = L2(φ(h,r)) + L2(t)
            self.regu += queries.norm(p=2) ** 2 / queries.shape[0]
            self.regu += tails.norm(p=2) ** 2 / tails.shape[0]

        else:
            # None
            pass

        self.regu *= self.regu_weight
        return
    def RotatE(self, head, relation, tail,ss_torch_index, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)


        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail

            if ss_torch_index != None:
                re_score = re_score[ss_torch_index] - re_head
                im_score = im_score[ss_torch_index] - im_head
            else:
                re_score = re_score - re_head
                im_score = im_score - im_head
        elif mode == 'tail-batch':
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            if ss_torch_index != None:
                re_score = re_score[ss_torch_index] - re_tail
                im_score = im_score[ss_torch_index] - im_tail
            else:
                re_score = re_score - re_tail
                im_score = im_score - im_tail
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score


    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, hr_freq, tr_freq, mode = next(train_iterator)



        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            hr_freq = hr_freq.cuda()
            tr_freq = tr_freq.cuda()


        if mode == 'tail-batch':
            aa = torch.reciprocal(torch.pow(hr_freq, args.square))
        else:
            aa = torch.reciprocal(torch.pow(tr_freq, args.square))
        aa = aa/aa.sum()

        negative_sample_size = negative_sample.shape[1]
        N_sample = int(negative_sample_size/6)

        ss_torch_out = torch.multinomial(aa,N_sample * aa.shape[0],replacement=True)

        args.umax += torch.unique(ss_torch_out,return_counts=True)[1].max().item()
        if len(torch.unique(ss_torch_out,return_counts=True)[0]) < len(aa):
            args.umin += 0
            args.lzero += len(aa) - len(torch.unique(ss_torch_out, return_counts=True)[0])

        else:
            args.umin += torch.unique(ss_torch_out,return_counts=True)[1].min().item()

        ss_torch_index = torch.sort(ss_torch_out)[0]

        ss_torch_rand = torch.randint(negative_sample_size,ss_torch_index.shape)

        negative_sample = negative_sample[ss_torch_index,ss_torch_rand]


        negative_score = model((positive_sample, negative_sample), mode, ss_torch_index)

        if mode == 'tail-batch':
            positive_score = model(positive_sample)
        else:
            positive_score = model(positive_sample, 'single_inverse')
        
        batch_size = positive_score.shape[0]

        if args.freq_based_subsampling:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                        * F.logsigmoid(-negative_score)).sum(dim = 1)
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(-negative_score).sum(dim = 1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
            if mode == 'head-batch':
                subsampling_weight = tr_freq
            if mode == 'tail-batch':
                subsampling_weight = hr_freq
            conditional_weight = (hr_freq + tr_freq).cuda()
            conditional_weight = torch.sqrt(1 / conditional_weight)
            subsampling_weight = torch.sqrt(1 / subsampling_weight)
            subsampling_weight = subsampling_weight.cuda()
            positive_sample_loss = - (conditional_weight * positive_score).sum() / conditional_weight.sum()
            negative_sample_loss = - N_sample * (subsampling_weight[ss_torch_index] * negative_score).sum() / subsampling_weight[ss_torch_index].sum()
            # negative_sample_loss = - negative_score.sum() / batch_size

        elif args.uniq_based_subsampling:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                        * F.logsigmoid(-negative_score)).sum(dim = 1)
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(-negative_score).sum(dim = 1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
            if mode == 'head-batch':
                subsampling_weight = tr_freq
            if mode == 'tail-batch':
                subsampling_weight = hr_freq
            subsampling_weight = torch.sqrt(1 / subsampling_weight)
            subsampling_weight = subsampling_weight.cuda()
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - N_sample * (subsampling_weight[ss_torch_index] * negative_score).sum() / subsampling_weight[ss_torch_index].sum()
        elif args.default_subsampling:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                        * F.logsigmoid(-negative_score)).sum(dim = 1)
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(-negative_score).sum(dim = 1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
            subsampling_weight = hr_freq + tr_freq
            subsampling_weight = torch.sqrt(1 / subsampling_weight)
            subsampling_weight = subsampling_weight.cuda()
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        else:
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                        * F.logsigmoid(-negative_score)).sum(dim = 1)
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(-negative_score).sum(dim = 1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
            positive_sample_loss = - (positive_score).sum() / batch_size
            negative_sample_loss = - (negative_score).sum() / batch_size

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'max_sample': args.umax/(1+args.step),
            'min_sample': args.umin/(1+args.step),
            'zero_sample': args.lzero/(1+args.step),
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
