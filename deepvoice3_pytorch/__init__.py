# coding: utf-8

# from .version import __version__

import torch
from torch import nn

from .modules import Embedding


class MultiSpeakerTTSModel(nn.Module):
    """Attention seq2seq model + post processing network
    """

    def __init__(self, seq2seq, postnet,
                 mel_dim=80, linear_dim=513,
                 n_speakers=1, speaker_embed_dim=16, padding_idx=None,
                 trainable_positional_encodings=False,
                 use_decoder_state_for_postnet_input=False,
                 speaker_embedding_weight_std=0.01,
                 freeze_embedding=False):
        super(MultiSpeakerTTSModel, self).__init__()
        self.seq2seq = seq2seq
        self.postnet = postnet  # referred as "Converter" in DeepVoice3
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.trainable_positional_encodings = trainable_positional_encodings
        self.use_decoder_state_for_postnet_input = use_decoder_state_for_postnet_input
        self.freeze_embedding = freeze_embedding

        # Speaker embedding
        if n_speakers > 1:
            self.embed_speakers = Embedding(
                n_speakers, speaker_embed_dim, padding_idx=None,
                std=speaker_embedding_weight_std)
        self.n_speakers = n_speakers
        self.speaker_embed_dim = speaker_embed_dim

    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)

    def get_trainable_parameters(self):
        freezed_param_ids = set()

        encoder, decoder = self.seq2seq.encoder, self.seq2seq.decoder

        # Avoid updating the position encoding
        if not self.trainable_positional_encodings:
            pe_query_param_ids = set(map(id, decoder.embed_query_positions.parameters()))
            pe_keys_param_ids = set(map(id, decoder.embed_keys_positions.parameters()))
            freezed_param_ids |= (pe_query_param_ids | pe_keys_param_ids)
        # Avoid updating the text embedding
        if self.freeze_embedding:
            embed_param_ids = set(map(id, encoder.embed_tokens.parameters()))
            freezed_param_ids |= embed_param_ids

        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, text_sequences, mel_targets=None, speaker_ids=None,
                text_positions=None, frame_positions=None, vid_positions=None, input_lengths=None, faces=None, video_lengths=None, batch_size=4):
        B = text_sequences.size(0)

        if speaker_ids is not None:
            assert self.n_speakers > 1
            speaker_embed = self.embed_speakers(speaker_ids)
        else:
            speaker_embed = None

        # Apply seq2seq
        # (B, T//r, mel_dim*r)
        mel_outputs, alignments, done, decoder_states, alignments_f = self.seq2seq(
            text_sequences, mel_targets, speaker_embed,
            text_positions, frame_positions, vid_positions, input_lengths, video_lengths, faces, batch_size)

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # Prepare postnet inputs
        if self.use_decoder_state_for_postnet_input:
            postnet_inputs = decoder_states.view(B, mel_outputs.size(1), -1)
        else:
            postnet_inputs = mel_outputs

        # (B, T, linear_dim)
        # Convert coarse mel-spectrogram (or decoder hidden states) to
        # high resolution spectrogram
        linear_outputs = self.postnet(postnet_inputs, speaker_embed)
        assert linear_outputs.size(-1) == self.linear_dim

        return mel_outputs, linear_outputs, alignments, done, alignments_f


class AttentionSeq2Seq(nn.Module):
    """Encoder + Decoder with attention
    """

    def __init__(self, encoder, decoder, face_encoder, face_decoder):
        super(AttentionSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.face_encoder = face_encoder
        self.face_decoder = face_decoder
        if isinstance(self.decoder.attention, nn.ModuleList):
            self.encoder.num_attention_layers = sum(
                [layer is not None for layer in decoder.attention])

    def forward(self, text_sequences, mel_targets=None, speaker_embed=None,
                text_positions=None, frame_positions=None, vid_positions=None, input_lengths=None,video_lengths=None, faces=None, batch_size=4):
        # (B, T, text_embed_dim)
        encoder_outputs = self.encoder(
            text_sequences, lengths=input_lengths, speaker_embed=speaker_embed)

        face_encoder_outputs =  self.face_encoder( faces, video_lengths, batch_size)
#         print(face_encoder_outputs.shape)
        # Mel: (B, T//r, mel_dim*r)
        # Alignments: (N, B, T_target, T_input)
        # Done: (B, T//r, 1)

        
        # face_decoder_outputs = self.face_decoder(
        #     face_encoder_outputs, mel_targets,
        #     text_positions=None, frame_positions=None,
        #     speaker_embed=speaker_embed, lengths=video_lengths)
        # print('shapess')
        # a,b = face_decoder_outputs
        # print(a.shape)
        # print(b.shape)
        
        mel_outputs, alignments, done, decoder_states, alignments_f = self.decoder(
            encoder_outputs, face_encoder_outputs, mel_targets,
            text_positions=text_positions, frame_positions=frame_positions, vid_positions=vid_positions,
            speaker_embed=speaker_embed, lengths=input_lengths, video_lengths=video_lengths)
        # print(alignments.shape)
        # print(alignments_f.shape)
        # print('lklklkl')
        return mel_outputs, alignments, done, decoder_states, alignments_f
