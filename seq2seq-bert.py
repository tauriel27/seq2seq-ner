import pdb

import os
import math
import tqdm
import shutil
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from transformers import BertModel, RobertaModel

from tensorboardX import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SEED = 27
PAD_INDEX = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, embedding_weight=None, bert=None):
        super().__init__()

        self.bert = bert

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0, _weight=embedding_weight)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src len, batch size]
        # src_len = [src len]

        # pdb.set_trace()

        if self.bert:
            tmp_src = src.permute(1, 0)
            embedded = self.bert(tmp_src)[0]
            embedded = embedded.permute(1, 0, 2)
        else:
            embedded = self.dropout(self.embedding(src))

        # embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, fp16=True):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        self.fp16 = fp16

    def forward(self, hidden, encoder_outputs, mask):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(
                torch.cat(
                    (hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention = [batch size, src len]

        if self.fp16:
            attention = attention.masked_fill(mask == 0, np.float16('-inf'))
        else:
            attention = attention.masked_fill(mask == 0, np.float32('-inf'))

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
            self,
            output_dim,
            emb_dim,
            enc_hid_dim,
            dec_hid_dim,
            dropout,
            attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + emb_dim,
            output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        #         print(embedded.size(), weighted.size())
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0):
        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of
        # the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(
            trg_len,
            batch_size,
            trg_vocab_size).to(
            self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed
        # through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        trg = trg[1:, :]
        # 可能会对性能造成影响
        #         input = torch.tensor([output_vocab.word2index['SOS']] * batch_size).to(self.device)

        mask = self.create_mask(src)

        # mask = [batch size, src len]

        for t in range(0, trg_len - 1):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(
                input, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        # if name == 'encoder.embedding.weight':
        #     print('encoder embedding init is skipped!')
        #     continue
        
        if 'bert' in name:
            continue

        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(args, model, data, input_vocab, output_vocab, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    total_loss = 0
    total_steps = 0
    val_iter = data_iter(
        args,
        data,
        input_vocab,
        output_vocab,
        data_type='val')
    with torch.no_grad():
        for i, batch in tqdm.tqdm(
                enumerate(val_iter), desc='Evaluating', total=math.ceil(data['val_num']/args.batch_size)):
            batch = tuple(t.to(device) for t in batch)
            src, src_len, trg = batch

            # turn off teacher forcing
            output = model(src, src_len, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]

            output = output[:-1].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()
            total_steps += 1

    return total_loss / total_steps


def get_input_ids(args, tokens, input_vocab):
    tokens = [i for i in tokens]
    seq_len = min(args.max_len, len(tokens))
    input_unk_token_id = input_vocab.word2index['UNK']

    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]

    input_ids = [input_vocab.word2index.get(
        i, input_unk_token_id) for i in tokens]
    return input_ids, seq_len


def parse_entities(sents, y_pred):
    """严格按照B...E
    """
    entities = []
    # assert len(sents) == len(y_pred)

    for chars, tags in zip(sents, y_pred):
        tmp = ''
        begin_flag = False

        # assert len(chars) == len(tags)
        for char, tag in zip(chars, tags):
            if tag[0] == 'B':
                tmp += char
                begin_flag = True
            elif tag[0] == 'M' and begin_flag:
                tmp += char
            elif tag[0] == 'E' and begin_flag:
                tmp += char
                entities.append(tmp)
                tmp = ''
                begin_flag = False
            elif tag[0] == 'O' and begin_flag:
                tmp = ''
                begin_flag = False

    print('%d entities.' % len(entities))
    return entities


def parse(args, tokens, input_vocab, output_vocab, model, device):
    model.eval()

    src_ids, src_len = get_input_ids(args, tokens, input_vocab)

    # batch_size last
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)

    src_len = torch.LongTensor([src_len]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_ids = [output_vocab.word2index['SOS']]

    attentions = torch.zeros(args.max_len, 1, len(src_ids)).to(device)

    for i in range(args.max_len):

        trg_tensor = torch.LongTensor([trg_ids[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(
                trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        pred_token = output.argmax(1).item()

        trg_ids.append(pred_token)

        if len(trg_ids) == src_len + 1:
            break

    trg_tokens = [output_vocab.index2word[i] for i in trg_ids[1:]]

    return trg_tokens, attentions[:len(trg_tokens) - 1]


def data_iter(args, data, input_vocab, output_vocab, data_type='train'):
    if data_type == 'train':
        tmp_input = data['input_train'][:]
        tmp_output = data['output_train'][:]
    elif data_type == 'val':
        tmp_input = data['input_val'][:]
        tmp_output = data['output_val'][:]
    elif data_type == 'test':
        tmp_input = data['input_test'][:]
        tmp_output = data['output_test'][:]

    tmp_lens = list(map(lambda x: min(args.max_len, len(x)), tmp_input))

    lens_series = pd.Series(tmp_lens)
    sorted_idx = lens_series.sort_values(ascending=False).index

    tmp_input = np.array(tmp_input)
    tmp_output = np.array(tmp_output)
    tmp_lens = np.array(tmp_lens)

    sorted_input = tmp_input[sorted_idx]
    sorted_output = tmp_output[sorted_idx]
    sorted_lens = tmp_lens[sorted_idx]

    inputs, outputs = [], []
    for _len, _input, _output in zip(sorted_lens, sorted_input, sorted_output):
        inputs.append(_input[:_len])
        outputs.append(_output[:_len])

    src = []
    src_len = []
    trg = []

    input_unk_token_id = input_vocab.word2index['UNK']
    if output_vocab:
        output_unk_token_id = output_vocab.word2index['UNK']

    for seq_len, eit, dot in zip(sorted_lens, inputs, outputs):

        dot = ['SOS'] + dot

        encoder_input_ids = torch.tensor(
            [input_vocab.word2index.get(i, input_unk_token_id) for i in eit])
        decoder_output_ids = torch.tensor(
            [output_vocab.word2index.get(i, output_unk_token_id) for i in dot])

        src.append(encoder_input_ids)
        src_len.append(seq_len)
        trg.append(decoder_output_ids)

        if len(src) == args.batch_size:
            src = torch.nn.utils.rnn.pad_sequence(
                src, padding_value=input_vocab.word2index['PAD'])
            trg = torch.nn.utils.rnn.pad_sequence(
                trg, padding_value=output_vocab.word2index['PAD'])
            yield src, torch.tensor(src_len), trg
            src, src_len, trg = [], [], []
    if len(src) > 0:
        src = torch.nn.utils.rnn.pad_sequence(
            src, padding_value=input_vocab.word2index['PAD'])
        trg = torch.nn.utils.rnn.pad_sequence(
            trg, padding_value=output_vocab.word2index['PAD'])
        yield src, torch.tensor(src_len), trg


def parse_iter(args, encoder_input_tokens, input_vocab):
    src = []
    src_len = []

    input_unk_token_id = input_vocab.word2index['UNK']

    print(f'{len(encoder_input_tokens)} examples')

    for eit in encoder_input_tokens:
        seq_len = min(args.max_len, len(eit))

        if len(eit) > args.max_len:
            eit = eit[:args.max_len]

        encoder_input_ids = torch.LongTensor(
            [input_vocab.word2index.get(i, input_unk_token_id) for i in eit])

        src.append(encoder_input_ids)
        src_len.append(seq_len)

        if len(src) == args.batch_size:
            src = torch.nn.utils.rnn.pad_sequence(
                src, padding_value=input_vocab.word2index['PAD'])
            yield src, torch.LongTensor(src_len)
            src, src_len = [], []

    if len(src) > 0:
        src = torch.nn.utils.rnn.pad_sequence(
            src, padding_value=input_vocab.word2index['PAD'])
        yield src, torch.LongTensor(src_len)


def parse_batch(args, tokens, model, input_vocab, output_vocab, device):
    model.eval()

    iteration = parse_iter(args, tokens, input_vocab)

    output_unk_id = output_vocab.word2index['UNK']

    output_tokens = []
    for batch in tqdm.tqdm(iteration, desc='Predicting'):
        src_ids, src_lens = batch
        src_ids = src_ids.to(device)
        src_lens = src_lens.to(device)

        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src_ids, src_lens)

        mask = model.create_mask(src_ids)

        # max_len + SOS
        example_num = src_ids.size(1)

        trg_ids = torch.empty(
            args.max_len + 1,
            example_num,
            dtype=torch.long).to(device)
        init_ids = torch.LongTensor(
            [output_vocab.word2index['SOS']] * example_num).to(device)
        trg_ids[0, ...] = init_ids

        for i in range(args.max_len):
            decoder_src = trg_ids[i, ...]

            with torch.no_grad():
                output, hidden, attention = model.decoder(
                    decoder_src, hidden, encoder_outputs, mask)

            pred_tokens = output.argmax(1)

            trg_ids[i + 1, ...] = pred_tokens

        trg_ids = trg_ids.detach().cpu().numpy()

        for i in range(example_num):
            seq_len = src_lens[i]
            pred_ids = trg_ids[..., i][1:seq_len + 1]
            pred_token = [
                output_vocab.index2word.get(
                    i, output_unk_id) for i in pred_ids]
            output_tokens.append(pred_token)

    return output_tokens


def random_evaluation(args, data, input_vocab, output_vocab, model, device):
    x = random.sample(data['input_test'], 1)[0]
    pred_tags, _ = parse(args, x, input_vocab, output_vocab, model, device)
    print('*' * 50)
    print(''.join(x))
    print(pred_tags)
    print(parse_entities([x], [pred_tags]))
    print('*' * 50)


def train(args, data, model, input_vocab, output_vocab, device):

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    tb_writer = SummaryWriter(args.output_dir)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer)

    criterion = nn.CrossEntropyLoss(
        ignore_index=output_vocab.word2index['PAD'])

    model.train()

    total_steps = 0

    best_loss = float('inf')

    for e in tqdm.trange(args.epochs):
        iterator = data_iter(
            args,
            data,
            input_vocab,
            output_vocab,
            data_type='train')
        for i, batch in tqdm.tqdm(
                enumerate(iterator), desc='Training', total=math.ceil(data['train_num']/args.batch_size)):

            batch = tuple(t.to(device) for t in batch)
            src, src_len, trg = batch

            output = model(src, src_len, trg)

            output_dim = output.shape[-1]

            # output[-1, ...] is 'EOS'
            output = output[:-1].view(-1, output_dim)
            # trg[0, ...] is 'SOS'
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)


            if i % args.gradient_accumulation_steps == 0:
                tb_writer.add_scalar('train loss', loss.item(), total_steps)
                optimizer.step()
                optimizer.zero_grad()


                if total_steps % args.validate_steps == 0:
                    valid_loss = evaluate(
                        args,
                        model=model,
                        data=data,
                        input_vocab=input_vocab,
                        output_vocab=output_vocab,
                        device=device)

                    tb_writer.add_scalar(
                        'val loss',
                        valid_loss,
                        total_steps /
                        args.validate_steps)

                    print(
                        f'train loss: {loss.item()}, val loss: {valid_loss}, steps: {total_steps}')

                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        print('Saving best model !')
                        torch.save(
                            model.state_dict(),
                            args.output_dir + '/model.pt')

                        random_evaluation(args, data, input_vocab, output_vocab, model, device)

                    model.train()
                total_steps += 1


def load_data(args, input_vocab, output_vocab):

    with open(args.bert_model + '/vocab.txt') as f:
        for line in f:
            line = line.rstrip()
            input_vocab.addWord(line)

    input_seq_train, output_seq_train = [], []
    _input_seq, _output_seq = [], []
    with open(args.data_dir + '/train.txt') as f:
        for line in f:
            if line == '\n':
                input_seq_train.append(_input_seq)
                output_seq_train.append(_output_seq)
                _input_seq, _output_seq = [], []
                continue
            items = line.rstrip().split()
            if len(items) == 2:
                char, tag = items
                # input_vocab.addWord(char)
                output_vocab.addWord(tag)
                _input_seq.append(char)
                _output_seq.append(tag)
    print(f'train examples: {len(input_seq_train)}')

    input_seq_val, output_seq_val = [], []
    _input_seq, _output_seq = [], []
    with open(args.data_dir + '/dev.txt') as f:
        for line in f:
            if line == '\n':
                input_seq_val.append(_input_seq)
                output_seq_val.append(_output_seq)
                _input_seq, _output_seq = [], []
                continue
            items = line.rstrip().split()
            if len(items) == 2:
                char, tag = items
                # input_vocab.addWord(char)
                output_vocab.addWord(tag)
                _input_seq.append(char)
                _output_seq.append(tag)
    print(f'val examples: {len(input_seq_val)}')

    input_seq_test, output_seq_test = [], []
    _input_seq, _output_seq = [], []
    with open(args.data_dir + '/test.txt') as f:
        for line in f:
            if line == '\n':
                input_seq_test.append(_input_seq)
                output_seq_test.append(_output_seq)
                _input_seq, _output_seq = [], []
                continue
            items = line.rstrip().split()
            if len(items) == 2:
                char, tag = items
                #             input_vocab.addWord(char)
                #             output_vocab.addWord(tag)
                _input_seq.append(char)
                _output_seq.append(tag)
    print(f'test examples: {len(input_seq_test)}')

    data = {
        'input_train': input_seq_train,
        'output_train': output_seq_train,
        'train_num': len(input_seq_train),
        'input_val': input_seq_val,
        'output_val': output_seq_val,
        'val_num': len(input_seq_val),
        'input_test': input_seq_test,
        'output_test': output_seq_test,
        'test_num': len(input_seq_test)
    }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--bert_model', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--clip', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--validate_steps', type=int, default=10)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    args = parser.parse_args()

    input_vocab = Vocab(name='input')
    output_vocab = Vocab(name='output')

    data = load_data(
        args,
        input_vocab,
        output_vocab)

    input_dim = input_vocab.n_words
    output_dim = output_vocab.n_words

    print(f'input vocab words: {input_dim}')
    print(f'output vocab words: {output_dim}')

    emb_dim = 128
    encoder_hidden = 256
    decoder_hidden = 256
    dropout = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use bert as feature extractor
    bert = RobertaModel.from_pretrained(args.bert_model)

    encoder = Encoder(
        input_dim=21128,
        emb_dim=768,
        enc_hid_dim=encoder_hidden,
        dec_hid_dim=decoder_hidden,
        dropout=dropout,
        bert=bert,
        # use pretrained embedding
        # embedding_weight=bert.state_dict()['embeddings.word_embeddings.weight'],
    )
    attention = Attention(
        enc_hid_dim=encoder_hidden,
        dec_hid_dim=decoder_hidden,
        fp16=args.fp16)

    decoder = Decoder(
        output_dim=output_dim,
        emb_dim=emb_dim,
        enc_hid_dim=decoder_hidden,
        dec_hid_dim=decoder_hidden,
        dropout=dropout,
        attention=attention)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=input_vocab.word2index['PAD'],
        device=device)

    for name, param in model.named_parameters():
        print(name, param.size())

    model.to(device)

    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    train(
        args=args,
        data=data,
        model=model,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        device=device)


if __name__ == '__main__':
    main()

