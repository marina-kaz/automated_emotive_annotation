from enum import Enum
from typing import Callable


from transformers import (BertForSequenceClassification,
                          RobertaForSequenceClassification,
                          DistilBertForSequenceClassification)


class ModelTypes(Enum):
    COINTEGRATED_RUBERT_TINY = 'cointegrated_rubert-tiny'
    SBERBANK_RU_ROBERTA_LARGE = 'sberbank-ai_ruRoberta-large'
    DPAVLOV_DISTILRUBERT_CASED_CONV = 'DeepPavlov_distilrubert-tiny-cased-conversational'


MODEL_TYPE_MAPPING = {
    'Distil RuBert Cased Conversational от DeepPavlov': ModelTypes.DPAVLOV_DISTILRUBERT_CASED_CONV,
    'Cointegrated RuBert': ModelTypes.COINTEGRATED_RUBERT_TINY
}


def initialize_model(model_type: ModelTypes):
    if model_type in [ModelTypes.COINTEGRATED_RUBERT_TINY]:
        return BertForSequenceClassification.from_pretrained(model_type.value)
    elif model_type in [ModelTypes.DPAVLOV_DISTILRUBERT_CASED_CONV]:
        return DistilBertForSequenceClassification.from_pretrained(model_type.value)
    elif model_type in [ModelTypes.SBERBANK_RU_ROBERTA_LARGE]:
        return RobertaForSequenceClassification.from_pretrained(model_type.value)


def embeddings_mapper(model_type: ModelTypes) -> Callable:
    if model_type in [ModelTypes.COINTEGRATED_RUBERT_TINY]:
        return lambda model: model.bert.embeddings.word_embeddings
    elif model_type in [ModelTypes.SBERBANK_RU_ROBERTA_LARGE]:
        return lambda model: model.roberta.embeddings.word_embeddings
    elif model_type in [ModelTypes.DPAVLOV_DISTILRUBERT_CASED_CONV]:
        return lambda model: model.distilbert.embeddings.word_embeddings


def get_wordpiece_symbol(model_type: ModelTypes) -> str:
    if model_type in [ModelTypes.COINTEGRATED_RUBERT_TINY,
                      ModelTypes.DPAVLOV_DISTILRUBERT_CASED_CONV]:
        return '##'
