from typing import Dict

from commaqa.inference.dataset_readers import DatasetReader, MultiParaRCReader
from commaqa.inference.participant_qa import (
    LLMQAParticipantModel, LLMQADecompParticipantModel, LLMTitleQAParticipantModel
)
from commaqa.inference.participant_execution_routed import RoutedExecutionParticipant
from commaqa.inference.odqa import AnswerExtractor, CopyQuestionParticipant, RetrieverParticipant


MODEL_NAME_CLASS = {
    "answer_extractor": AnswerExtractor,
    "retriever": RetrieverParticipant,
    "copy_question": CopyQuestionParticipant,
    "execute_router": RoutedExecutionParticipant,
    "llmqa": LLMQAParticipantModel,
    "llmqadecomp": LLMQADecompParticipantModel,
    "llmtitleqa": LLMTitleQAParticipantModel,
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "multi_para_rc": MultiParaRCReader,
}

PREDICTION_TYPES = {"answer", "titles", "pids"}
