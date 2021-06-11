# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from .modules import Embedding, Linear, Conv1d, ConvTranspose1d
from .modules import HighwayConv1d, get_mask_from_lengths, conv_block, residual_block, BatchConv2D_5D
from .modules import position_encoding_init
from .deepvoice3 import AttentionLayer
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Encoder(nn.Module):
    def __init__(self, n_vocab, embed_dim, channels, kernel_size=3,
                 n_speakers=1, speaker_embed_dim=16, embedding_weight_std=0.01,
                 padding_idx=None, dropout=0.1):
        super(Encoder, self).__init__()
        self.dropout = dropout

        # Text input embeddings
        self.embed_tokens = Embedding(
            n_vocab, embed_dim, padding_idx, embedding_weight_std)

        E = embed_dim # 128
        D = channels # 256
        self.convnet = nn.Sequential(
            Conv1d(E, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            nn.ReLU(inplace=True),
            Conv1d(2 * D, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),

            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=3, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=9, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=27, std_mul=1.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=3, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=9, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=27, std_mul=1.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=1, padding=0,
                          dilation=1, std_mul=1.0, dropout=dropout),
        )

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        # embed text_sequences
        # (B, T, E)
        x = self.embed_tokens(text_sequences)
        # print("######## Anchit print x embed tokens ########")
        # print(x.shape)
        # print(x.transpose(1, 2).shape)
        x = self.convnet(x.transpose(1, 2)).transpose(1, 2)
        # print("######## Anchit print x convnet ########")
        # print(x.shape)
        # (B, T, D) and (B, T, D)
        keys, values = x.split(x.size(-1) // 2, dim=-1)
        # print(keys.shape, values.shape)
        return keys, values

class Face_Encoder(nn.Module):
    def __init__(self, video, length, embed_dim=512, channels=256, kernel_size=3,
                 n_speakers=1, speaker_embed_dim=16, embedding_weight_std=0.01,
                 padding_idx=None, dropout=0.1):
        super(Face_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=(1,2),padding=1)
#         self.conv2 = BatchConv2D_5D(6, 32, 3, stride=(1,2),padding=1, groups=4)
        
        
        E = embed_dim # 512
        D = channels # 256
        self.convnet = nn.Sequential(
            Conv1d(E, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            nn.ReLU(inplace=True),
            Conv1d(2 * D, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),

            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=3, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=9, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=27, std_mul=1.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=3, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=9, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=27, std_mul=1.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=1, padding=0,
                          dilation=1, std_mul=1.0, dropout=dropout),
        )
        
    def forward(self, video, lengths, batch_size=4):

#         # (B, T, E)
#         input_face = video.transpose(2,4).transpose(3,4)
        new_input_face = None
        if lengths is not None:
            for idx, vl in enumerate(lengths):
                face = video[idx][:vl]   # face = no. frames x 48 x 96 x 3
                if new_input_face is None:
                    new_input_face = face
                else:
                    new_input_face = torch.cat((new_input_face,face),0) # sum_of_all_frames_in_batch x 48 x 96 x 3
        else:
            # during inference
            new_input_face = video
#  save to check
        # new_input_face = new_input_face.to(device)
        new_input_face = new_input_face.transpose(1,3).transpose(2,3) # sum_of_all_frames_in _batch x 3 x 48 x 96
        # print("new_input_face", new_input_face.shape)
        
        out = self.conv1(new_input_face)
#         print(out.shape)
        x1_face = conv_block(out,64,kernel_size=5,strides=2)
        x1_face = residual_block(x1_face,64)
        x1_face = residual_block(x1_face,64)
        x2_face = conv_block(x1_face,128,kernel_size=3,strides=2)
        x2_face = residual_block(x2_face,128)
        x2_face = residual_block(x2_face,128)
        x2_face = residual_block(x2_face,128)
        x3_face = conv_block(x2_face,256,kernel_size=3,strides=2)
        x3_face = residual_block(x3_face,256)
        x3_face = residual_block(x3_face,256)
        x4_face = conv_block(x3_face,512,kernel_size=3,strides=2)
        x5_face = conv_block(x4_face,512,kernel_size=3,strides=1,do_pad=False)
        x6_face = conv_block(x5_face,512,kernel_size=1,strides=1)
        y = torch.squeeze(x6_face)
        y = y.to(device)  # all_frames_batch x 512
#         print('naya face encoder hai')
#         print(y.shape)
        
#         x = y.unsqueeze(0) # 1 x all_feams x 512
#          unstack and padd then highway thing
        if lengths is not None:
            max_vid_len = max(lengths)
            final = torch.zeros(batch_size,max_vid_len,512)
            prev = 0
            nxt = lengths[0]
            final[0][:nxt] = y[:nxt][:]
            for idx in range(1, len(lengths)):
                prv = nxt
                nxt = prv + lengths[idx]
                final[idx][:lengths[idx]] = y[prv:nxt][:]
        else:
            # during inference
            final = y.unsqueeze(0)
        final=final.to(device)
        # print(final)
        t = self.convnet(final.transpose(1, 2)).transpose(1, 2) # 1x all_frames x 512
        # print('satcskahgbjvckjdcvkhj')
        # print(final.shape)
        # print(t.shape)
        ## shape of t [1, sum of frames, 512]
        # t = torch.squeeze(t) ## now [sum of frames, 512]
        # print(x.shape)
	    ## using video lengths , dicard the rest rows and generate keys and values
        # print('final')
        # print(t.shape)
#         max_vid_len = max(lengths)
        # print(lengths)
#         final = torch.zeros(batch_size,max_vid_len,512)
        # for idx,vl in enumerate(lengths):
        #     final[idx][:vl] = t[idx][:vl]
#         prv = 0
#         nxt = lengths[0]
        # print(prv, "----->", nxt)
#         final[0][:nxt] = t[prv:nxt]
#         for idx in range(1, len(lengths)):
#         	prv = nxt
#         	nxt = prv + lengths[idx]
        	# print(prv, "----->",nxt)
#         	final[idx][:lengths[idx]] = t[prv:nxt]
        
#         for idx, vl in enumerate(lengths):
#             final[idx]
        # print(final.shape)
        t = t.to(device)
        keys, values = t.split(t.size(-1) // 2, dim=-1)
        # print(keys.shape, values.shape)
        return keys, values

class Face_Decoder(nn.Module):
    def __init__(self, embed_dim, in_dim=80, r=5, channels=256, kernel_size=3,
                 n_speakers=1, speaker_embed_dim=16,
                 max_positions=512, padding_idx=None,
                 dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 window_ahead=3,
                 window_backward=1,
                 key_projection=False,
                 value_projection=False,
                 ):
        super(Face_Decoder, self).__init__()
        def forward(self, encoder_out, face_encoder_out, inputs=None,
                text_positions=None, frame_positions=None,
                speaker_embed=None, lengths=None, video_lengths=None):

#         if inputs is None:
#             assert text_positions is not None
#             self.start_fresh_sequence()
#             outputs = self.incremental_forward(encoder_out, text_positions)
            return 0
#         print('here')
class Decoder(nn.Module):
    def __init__(self, embed_dim, in_dim=80, r=5, channels=256, kernel_size=3,
                 n_speakers=1, speaker_embed_dim=16,
                 max_positions=512, padding_idx=None,
                 dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 window_ahead=3,
                 window_backward=1,
                 key_projection=False,
                 value_projection=False,
                 ):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r

        D = channels
        F = in_dim * r  # should be r = 1 to replicate
        self.audio_encoder_modules = nn.ModuleList([
            Conv1d(F, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            nn.ReLU(inplace=True),
            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.ReLU(inplace=True),
            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),

            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=1, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=3, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=9, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=27, causal=True, std_mul=1.0, dropout=dropout),

            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=1, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=3, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=9, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=27, causal=True, std_mul=1.0, dropout=dropout),

            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=3, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=3, causal=True, std_mul=1.0, dropout=dropout),
        ])

        self.attention = AttentionLayer(D, D, dropout=dropout,
                                        window_ahead=window_ahead,
                                        window_backward=window_backward,
                                        key_projection=key_projection,
                                        value_projection=value_projection)

        self.audio_decoder_modules = nn.ModuleList([
            # Conv1d(2 * D, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            Conv1d(3 * D, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),

            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=1, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=3, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=9, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=27, causal=True, std_mul=1.0, dropout=dropout),

            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=1, causal=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=kernel_size, padding=None,
                          dilation=1, causal=True, std_mul=1.0, dropout=dropout),

            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            nn.ReLU(inplace=True),
            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.ReLU(inplace=True),
            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.ReLU(inplace=True),
        ])
        self.last_conv = Conv1d(D, F, kernel_size=1, padding=0, dilation=1, std_mul=2.0)

        # Done prediction
        self.fc = Linear(F, 1)

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = Embedding(
            max_positions, D, padding_idx)
        self.embed_query_positions.weight.data = position_encoding_init(
            max_positions, D, position_rate=query_position_rate, sinusoidal=True)
        self.embed_keys_positions = Embedding(
            max_positions, D, padding_idx)
        self.embed_keys_positions.weight.data = position_encoding_init(
            max_positions, D, position_rate=key_position_rate, sinusoidal=True)

        # options
        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        self.force_monotonic_attention = force_monotonic_attention

    def forward(self, encoder_out, face_encoder_out, inputs=None,
                text_positions=None, frame_positions=None, vid_positions=None,
                speaker_embed=None, lengths=None, video_lengths=None):

        if inputs is None:
            assert text_positions is not None
            self.start_fresh_sequence()
            outputs = self.incremental_forward(encoder_out, text_positions, face_encoder_out, vid_positions)
            return outputs
#         print('here')
        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r

        keys, values = encoder_out
        keys_f, values_f = face_encoder_out
#         print('here1')
#         print("keys", keys.shape, "values", values.shape)
        if self.use_memory_mask and lengths is not None:
            mask = get_mask_from_lengths(keys, lengths)
        else:
            mask = None

        if self.use_memory_mask and video_lengths is not None:
            mask_f = get_mask_from_lengths(keys_f, video_lengths)
        else:
            mask_f = None

#         print('here2')
        # position encodings
        if text_positions is not None:
            text_pos_embed = self.embed_keys_positions(text_positions)
            keys = keys + text_pos_embed
            ## VIDEO ##
            vid_pos_embed = self.embed_keys_positions(vid_positions)
            keys_f = keys_f + vid_pos_embed
        if frame_positions is not None:
            frame_pos_embed = self.embed_query_positions(frame_positions)
        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()
#         print("hereAnchit")
#         print("keys", keys.shape)
        # (B, T, C)
        x = inputs
#         print("x=inputs", x.shape)
        # (B, C, T)
        x = x.transpose(1, 2)
#         print("x.transpose", x.shape)
        # Apply audio encoder
        for f in self.audio_encoder_modules:
            x = f(x)
        Q = x
        # print("Q=x", Q.shape)
        # Attention modules assume query as (B, T, C)
        x = x.transpose(1, 2)
#         print("x.transpose", x.shape)
        x = x if frame_positions is None else x + frame_pos_embed
#         print('jfhdhgd')
#         print(x.shape)
        R, alignments = self.attention(x, (keys, values), mask=mask)
        # print("R", R.shape)
        R = R.transpose(1, 2)

        ##### VIDEO #######
        keys_f = keys_f.transpose(1, 2).contiguous()
        R_f, alignments_f = self.attention(x, (keys_f, values_f), mask=mask_f)
        R_f = R_f.transpose(1, 2)
        # (B, C*2, T)
        # Rd = torch.cat((R, Q), dim=1)
        Rd = torch.cat((R, Q, R_f), dim=1)

        x = Rd
#         print(x.shape)
        # Apply audio decoder
        for f in self.audio_decoder_modules:
            x = f(x)
        decoder_states = x.transpose(1, 2).contiguous()
        x = self.last_conv(x)

        # (B, T, C)
        x = x.transpose(1, 2)

        # Mel
        outputs = torch.sigmoid(x)

        # Done prediction
        done = torch.sigmoid(self.fc(x))

        # Adding extra dim for convenient
        alignments = alignments.unsqueeze(0)
        alignments_f = alignments_f.unsqueeze(0)
        
        return outputs, alignments, done, decoder_states, alignments_f

    def incremental_forward(self, encoder_out, text_positions, encoder_face, video_positions,
                            initial_input=None, test_inputs=None):
        keys, values = encoder_out
        ## ADDED ##
        keys_f, values_f = encoder_face
        B_f = keys_f.size(0)
        B = keys.size(0)

        # position encodings
        if text_positions is not None:
            text_pos_embed = self.embed_keys_positions(text_positions)
            keys = keys + text_pos_embed
            ## ADDED ##
            vid_pos_embed = self.embed_keys_positions(video_positions)
            keys_f = keys_f + vid_pos_embed

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()
        ## ADDED ##
        keys_f = keys_f.transpose(1, 2).contiguous()
        decoder_states = []
        outputs = []
        alignments = []
        alignments_f = []
        dones = []
        # intially set to zeros
        last_attended = 0 if self.force_monotonic_attention else None
        ## ADDED ##
        last_attended_f = 0 if self.force_monotonic_attention else None

        t = 0
        if initial_input is None:
            initial_input = keys.data.new(B, 1, self.in_dim * self.r).zero_()
        current_input = initial_input
        while True:
            # frame pos start with 1.
            frame_pos = keys.data.new(B, 1).fill_(t + 1).long()
            frame_pos_embed = self.embed_query_positions(frame_pos)

            ## ADDED ##
            frame_pos_f = keys_f.data.new(B_f, 1).fill_(t + 1).long()
            frame_pos_embed_f = self.embed_query_positions(frame_pos_f)

            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]

            # (B, 1, C)
            x = current_input

            for f in self.audio_encoder_modules:
                try:
                    x = f.incremental_forward(x)
                except AttributeError as e:
                    x = f(x)
            Q = x

            R, alignment = self.attention(
                x + frame_pos_embed, (keys, values), last_attended=last_attended)
            
            ## ADDED ##
            # print(x.shape, keys.shape, values.shape)
            # print(x.shape, keys_f.shape, values_f.shape)
            R_f, alignment_f = self.attention(
                x + frame_pos_embed_f, (keys_f, values_f), last_attended=last_attended_f)

            if self.force_monotonic_attention:
                last_attended = alignment.max(-1)[1].view(-1).data[0]
            ## ADDED ##
            if self.force_monotonic_attention:
                last_attended_f = alignment_f.max(-1)[1].view(-1).data[0]

            ## ADDED ##
            Rd = torch.cat((R, Q, R_f), dim=-1)
            x = Rd
            for f in self.audio_decoder_modules:
                try:
                    x = f.incremental_forward(x)
                except AttributeError as e:
                    x = f(x)
            decoder_state = x
            x = self.last_conv.incremental_forward(x)

            # Ooutput & done flag predictions
            output = torch.sigmoid(x)
            done = torch.sigmoid(self.fc(x))

            decoder_states += [decoder_state]
            outputs += [output]
            alignments += [alignment]
            alignments_f += [alignment_f] ## ADDED ##
            dones += [done]

            t += 1
            # print(t, self.max_decoder_steps)
            if test_inputs is None:
                if (done > 0.5).all() and t > self.min_decoder_steps:
                    break
                # elif t > self.max_decoder_steps:
                #     break
                elif t > 200:
                    break

        # Remove 1-element time axis
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        alignments_f = list(map(lambda x: x.squeeze(1), alignments_f)) ## ADDED ##
        decoder_states = list(map(lambda x: x.squeeze(1), decoder_states))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 1)
        alignments_f = torch.stack(alignments_f).transpose(0, 1) ## ADDED ##
        decoder_states = torch.stack(decoder_states).transpose(0, 1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments, dones, decoder_states, alignments_f ## ADDED ##

    def start_fresh_sequence(self):
        _clear_modules(self.audio_encoder_modules)
        _clear_modules(self.audio_decoder_modules)
        self.last_conv.clear_buffer()


def _clear_modules(modules):
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError as e:
            pass


class Converter(nn.Module):
    def __init__(self, in_dim, out_dim, channels=512,  kernel_size=3, dropout=0.1):
        super(Converter, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim

        F = in_dim
        Fd = out_dim
        C = channels
        self.convnet = nn.Sequential(
            Conv1d(F, C, kernel_size=1, padding=0, dilation=1, std_mul=1.0),

            HighwayConv1d(C, C, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(C, C, kernel_size=kernel_size, padding=None,
                          dilation=3, std_mul=1.0, dropout=dropout),

            ConvTranspose1d(C, C, kernel_size=2, padding=0, stride=2, std_mul=1.0),
            HighwayConv1d(C, C, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(C, C, kernel_size=kernel_size, padding=None,
                          dilation=3, std_mul=1.0, dropout=dropout),
            ConvTranspose1d(C, C, kernel_size=2, padding=0, stride=2, std_mul=1.0),
            HighwayConv1d(C, C, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(C, C, kernel_size=kernel_size, padding=None,
                          dilation=3, std_mul=1.0, dropout=dropout),

            Conv1d(C, 2 * C, kernel_size=1, padding=0, dilation=1, std_mul=1.0),

            HighwayConv1d(2 * C, 2 * C, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),
            HighwayConv1d(2 * C, 2 * C, kernel_size=kernel_size, padding=None,
                          dilation=1, std_mul=1.0, dropout=dropout),

            Conv1d(2 * C, Fd, kernel_size=1, padding=0, dilation=1, std_mul=1.0),

            Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            nn.ReLU(inplace=True),
            Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.ReLU(inplace=True),

            Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.Sigmoid(),
        )

    def forward(self, x, speaker_embed=None):
        return self.convnet(x.transpose(1, 2)).transpose(1, 2)
