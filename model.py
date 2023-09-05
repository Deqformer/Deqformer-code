from transformers import BertModel
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch


class DNAModel(BertModel):
    def __init__(self, config):
        super(DNAModel, self).__init__(config)
        self.config = config
        self.left = BertModel(self.config)
        self.right = BertModel(self.config)
        self.pre_classifier = nn.Linear(config.dim , config.dim )
        self.classifier = nn.Linear(config.dim, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self,
        input_ids=None,
        rev_input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        output1 = self.left(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output2 = self.right(
            input_ids=rev_input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state1 = output1[0]  # (bs, seq_len, dim)
        pooled_output1 = hidden_state1[:, 0]  # (bs, dim)

        hidden_state2 = output2[0]  # (bs, seq_len, dim)
        pooled_output2 = hidden_state2[:, 0]  # (bs, dim)

        #pooled_output = torch.cat((pooled_output1, pooled_output2), 1)
        pooled_output = torch.add(pooled_output1, pooled_output2)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )