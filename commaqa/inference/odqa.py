import re
import json
import time
from typing import List
from functools import lru_cache
import random

import requests
from rapidfuzz import fuzz

from commaqa.inference.data_instances import QuestionAnsweringStep, QuestionGenerationStep, Task
from commaqa.inference.model_search import ParticipantModel
from commaqa.inference.dataset_readers import get_pid_for_title_paragraph_text


random.seed(100)  # Don't change.


@lru_cache(maxsize=None)
def get_spacy_object():
    import spacy

    return spacy.load("en_core_web_sm")


def is_reasoning_sentence(sentence: str) -> bool:
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True

    regex = re.compile("(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
    match = bool(re.match(regex, sentence))
    if match:
        return True

    return False


def remove_reasoning_sentences(sentences: List[str]) -> List[str]:
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]


def safe_post_request(url, params):
    for _ in range(10):
        try:
            return requests.post(url, json=params)
        except:
            print("Post request didn't succeed. Will wait 20s and retry.")
            time.sleep(20)
    raise Exception("Post request couldn't succeed after several attempts.")


def remove_wh_words(text: str) -> str:
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    text = " ".join(words)
    return text


def get_real_pid_for_title_paragraph_text(
    source_corpus_name: str, retriever_host: str, retriever_port: str, title, paragraph_text
) -> str:
    query_text = " ".join(paragraph_text.split(" ")[:30])
    params = {
        "retrieval_method": "retrieve_from_elasticsearch",
        "allowed_titles": [title],
        "query_text": query_text,
        "max_hits_count": 20,
        "corpus_name": source_corpus_name,
        "document_type": "paragraph_text",
    }

    url = retriever_host.rstrip("/") + ":" + str(retriever_port) + "/retrieve"
    result = safe_post_request(url, params)

    result = result.json()
    retrieval = result["retrieval"]

    if not retrieval:
        print("WARNING: Not para with the same title retrieved.")
        return ""

    def para_similarity_func(retrieval_):
        return (
            float(retrieval_["title"].lower() == title.lower())
            + get_token_similarity(retrieval_["paragraph_text"], paragraph_text) / 100
        )

    retrieval = sorted(retrieval, key=para_similarity_func, reverse=True)[0]

    retrieved_title = retrieval["title"]
    retrieved_para = retrieval.get("paragraph_text", "")  # backoff for natcq
    retrieved_id = retrieval["id"]  # has to be there.
    assert retrieved_id

    if retrieved_title != title:
        print("WARNING: Para with the same title couldn't be identified.")
        retrieved_id = ""
    if retrieved_para != paragraph_text:
        print("WARNING: Para with the same paragraph_text couldn't be identified.")
        retrieved_id = ""

    return retrieved_id


def is_para_closely_matching(
    existing_titles: List[str],
    existing_paras: List[str],
    new_title: str,
    new_para: str,
    match_threshold: float = 90,
) -> bool:
    if new_title in existing_titles and new_para in existing_paras:
        return True

    assert match_threshold > 1.0, "The threshold is 0-100 scaled."

    assert len(existing_titles) == len(existing_paras)
    for existing_title, existing_para in zip(existing_titles, existing_paras):
        condition_1 = fuzz.ratio(existing_title, new_title) >= match_threshold
        condition_2 = fuzz.ratio(existing_para, new_para) >= match_threshold
        if condition_1 and condition_2:
            return True
    return False


def para_to_text(title: str, para: str, max_num_words: int) -> int:
    # Note: the split and join must happen before the attaching title+para.
    # also don't split() because that disrupts the new lines.
    para = " ".join(para.split(" ")[:max_num_words])
    para = (
        para.strip()
        if para.strip().startswith("Wikipedia Title: ")
        else "Wikipedia Title: " + title + "\n" + para.strip()
    )
    return para


def assert_unique_titles_paras(titles: List[str], paras: List[str]) -> bool:
    titles_paras = [(title, para) for title, para in zip(titles, paras)]
    assert len(titles_paras) == len(set(titles_paras))


def get_token_similarity(str_1: str, str_2: str) -> float:
    return fuzz.token_sort_ratio(str_1.lower(), str_2.lower())


class AnswerExtractor(ParticipantModel):
    def __init__(
        self,
        regex,
        next_model="[EOQ]",
        match_all_on_failure=False,
        query_source="last_question",
        remove_last_fullstop=False,
    ):
        self.regex = re.compile(regex)
        self.next_model = next_model
        self.num_calls = 0
        self.match_all_on_failure = match_all_on_failure
        self.query_source = query_source
        self.remove_last_fullstop = remove_last_fullstop
        assert query_source in (
            "last_question",
            "last_answer",
        ), f"query_source must be either last_question or last_answer. Found {query_source}."

    def return_model_calls(self):
        return {"extract": self.num_calls}

    def query(self, state, debug=False):
        self.num_calls += 1

        new_state = state.copy()

        if self.query_source == "last_answer":
            query = new_state.data.get_last_answer()
        else:
            query = new_state.data.get_last_question()

        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]

        m = self.regex.match(query)
        if self.match_all_on_failure and not self.regex.match(query):
            m = re.compile(r"(.*)").match(query)

        if m:
            answer = m.group(1)

            if self.remove_last_fullstop and answer.endswith("."):
                answer = answer[:-1]

            if debug:
                print("EXT: " + answer)

            try:  # Hacky. Fix later. This is to handle '[\\"1,450 miles\\"]' to '["1,450 miles"]'
                json.loads(answer)
            except:
                try:
                    answer = json.dumps(json.loads(answer.encode("utf-8").decode("unicode_escape")))
                except:
                    pass

            new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
            new_state.last_output = answer
            new_state.next = self.next_model
            return new_state
        else:
            print("Answer Extractor did not find a match for input regex in {}".format(query))
            return []


class CopyQuestionParticipant(ParticipantModel):
    """
    Generates question by copying the question field from the data json.
    """

    def __init__(
        self,
        next_model=None,
        end_state="[EOQ]",
        eoq_after_n_calls=1,
    ):
        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0
        self.eoq_after_n_calls = eoq_after_n_calls

    def return_model_calls(self):
        return {"copy_question": self.num_calls}

    def query(self, state, debug=False):
        if (self.num_calls + 1) % (self.eoq_after_n_calls + 1) == 0:
            output = self.end_state
        else:
            output = state.data["question"].strip()

        self.num_calls += 1

        new_state = state.copy()

        new_state.data.add_qgen(QuestionGenerationStep(question=output, score=0, participant=state.next))

        if output == self.end_state:
            new_state.next = self.end_state
        else:
            new_state.data.add_task(Task(task_question=None, task_participant=new_state.next))
            new_state.next = self.next_model

        return [new_state]


class RetrieverParticipant(ParticipantModel):
    def __init__(
        self,
        retrieval_type,
        retriever_host,
        retriever_port,
        retrieval_count,
        source_corpus_name,
        query_source,
        document_type,
        cumulate_titles_paras_in_prefix=None,
        global_max_num_paras=100,
        next_model=None,
        end_state="[EOQ]",
    ):

        assert retrieval_type in (
            "bm25",
            "dpr",
            "blink_bm25",
            "bm25_and_blink_bm25",
            "bm25_and_dpr",
        ), f"retrieval_type {retrieval_type} not among the valid choices."

        assert query_source in (
            "original_question",  # state["data"]["question"]
            "last_answer",  # state.data.get_last_answer()
            "last_question",  # state.data.get_last_question()
        ), f"query_source {query_source} not among the valid choices."

        assert document_type in ("title", "paragraph_text", "title__paragraph_text")

        assert retrieval_count is not None, "retrieval_count is needed."
        assert source_corpus_name is not None, "source_corpus_name is needed."

        self.retrieval_type = retrieval_type
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port
        self.retrieval_count = retrieval_count
        self.document_type = document_type
        self.query_source = query_source
        self.source_corpus_name = source_corpus_name
        self.cumulate_titles_paras_in_prefix = cumulate_titles_paras_in_prefix
        self.global_max_num_paras = global_max_num_paras

        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0

        self.retrieval_failures_so_far = 0
        self.retrieval_failures_max = 9

    def return_model_calls(self):
        return {"paragraph_retrieve_and_reset": self.num_calls}

    def query(self, state, debug=False):
        if self.query_source == "original_question":
            input_query = state.data["question"]

        elif self.query_source == "last_answer":
            input_query = state.data.get_last_answer()

        elif self.query_source == "last_question":
            input_query = state.data.get_last_question()

        selected_titles = []
        selected_paras = []

        if self.retrieval_type == "bm25":
            retrieval_types = ["bm25"]
            retrieval_methods = ["retrieve_from_elasticsearch"]
        elif self.retrieval_type == "blink_bm25":
            retrieval_types = ["blink_bm25"]
            retrieval_methods = ["retrieve_from_blink_and_elasticsearch"]
        elif self.retrieval_type == "bm25_and_blink_bm25":
            retrieval_types = ["bm25", "blink_bm25"]
            retrieval_methods = ["retrieve_from_elasticsearch", "retrieve_from_blink_and_elasticsearch"]
        elif self.retrieval_type == "dpr":
            retrieval_types = ["dpr"]
            retrieval_methods = ["retrieve_from_dpr"]
        elif self.retrieval_type == "bm25_and_dpr":
            retrieval_types = ["bm25", "dpr"]
            retrieval_methods = ["retrieve_from_elasticsearch", "retrieve_from_dpr"]
        else:
            raise Exception(f"Unknown retrieval_type {self.retrieval_type}")

        for retrieval_type, retrieval_method in zip(retrieval_types, retrieval_methods):
            params = {
                "retrieval_method": retrieval_method,
                "query_text": input_query,
                "max_hits_count": self.retrieval_count,
                "corpus_name": self.source_corpus_name,
            }

            document_types = self.document_type.split("__")
            for document_type in document_types:
                if retrieval_type == "bm25":
                    params["document_type"] = document_type
                else:
                    params.pop("document_type", None)

                url = self.retriever_host.rstrip("/") + ":" + str(self.retriever_port) + "/retrieve"
                result = safe_post_request(url, params)

                if result.ok:
                    result = result.json()
                    retrieval = result["retrieval"]

                    for retrieval_item in retrieval:
                        if retrieval_item["corpus_name"] != self.source_corpus_name:
                            raise Exception(
                                f"The retrieved corpus name {retrieval_item['corpus_name']} "
                                f"doesn't match {self.source_corpus_name}."
                            )

                        if (  # This was changed post-hoc to include both conditions.
                            retrieval_item["title"] in selected_titles
                            and retrieval_item["paragraph_text"] in selected_paras
                        ):
                            continue

                        if len(selected_paras) >= self.global_max_num_paras:
                            continue

                        selected_titles.append(retrieval_item["title"])
                        selected_paras.append(retrieval_item["paragraph_text"])

                else:
                    self.retrieval_failures_so_far += 1
                    if self.retrieval_failures_so_far > self.retrieval_failures_max:
                        raise Exception(
                            f"Retrieval failure exceeded max allowed times ({self.retrieval_failures_so_far} > {self.retrieval_failures_max})"
                        )
                    print(
                        f"WARNING: Retrieval of titles did not succeed {self.retrieval_failures_so_far} times. Skipping it."
                    )

        self.num_calls += 1
        answer = json.dumps(selected_titles)

        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state

        if self.cumulate_titles_paras_in_prefix:
            _selected_titles = new_state.data.get(self.cumulate_titles_paras_in_prefix + "titles", []) + selected_titles
            _selected_paras = new_state.data.get(self.cumulate_titles_paras_in_prefix + "paras", []) + selected_paras
            new_state.data[self.cumulate_titles_paras_in_prefix + "titles"] = _selected_titles + selected_titles
            new_state.data[self.cumulate_titles_paras_in_prefix + "paras"] = _selected_paras + selected_paras

        return new_state
