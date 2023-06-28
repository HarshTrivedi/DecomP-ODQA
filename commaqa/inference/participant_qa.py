import json
from datetime import datetime
import random
import re

from dateutil.parser import parse

from commaqa.execution.llm_qa_model import LLMQAModel
from commaqa.inference.data_instances import QuestionAnsweringStep
from commaqa.inference.model_search import ParticipantModel
from commaqa.inference.participant_qgen import QuestionGenParticipant
from commaqa.inference.odqa import para_to_text, get_spacy_object


random.seed(100)  # Don't change


def extract_key_information(state, info_type: str) -> str:
    if info_type is None:
        return ""

    elif info_type == "cot":
        generated_sentences = state.data["generated_sentences"]
        generated_sentences = [
            sentence.strip() for sentence in generated_sentences if "answer is".lower() not in sentence.lower()
        ]
        key_info_text = " ".join(generated_sentences).strip()
        key_info_text = "\n\nKey Information: " + key_info_text
        return key_info_text

    elif info_type == "subqas":
        raise Exception("Make sure this is really what you want. In subqa format, the last step is still cot.")
        generated_sentences = state.data["generated_sentences"]
        generated_sentences = [
            sentence.strip() for sentence in generated_sentences if "answer is".lower() not in sentence.lower()
        ]
        key_info_text = "\n".join([e.strip() for e in generated_sentences])
        key_info_text = "\n\nKey Information:\n" + key_info_text
        return key_info_text

    else:
        raise Exception(f"Unknown info_type {info_type}")


class LLMQAParticipantModel(ParticipantModel):
    def __init__(
        self,
        next_model=None,
        end_state="[EOQ]",
        extractor_regex=None,
        extractor_remove_last_fullstop=False,
        allow_empty_answers=True,
        max_para_num_words=350,
        shuffle_paras=False,
        answer_is_numbered_list=False,
        store_sents_in_generated_sentences=False,
        question_prefix="",
        key_info_type=None,
        **kwargs,
    ):
        self.answer_is_numbered_list = answer_is_numbered_list

        if answer_is_numbered_list:
            kwargs["stop"] = ["\n\n"]

        self.key_info_type = key_info_type
        self.qa_model = LLMQAModel(**kwargs)
        self.next_model = next_model
        self.end_state = end_state
        self.extractor_regex = None
        if extractor_regex is not None:
            self.extractor_regex = re.compile(extractor_regex)
        self.extractor_remove_last_fullstop = extractor_remove_last_fullstop
        self.num_calls = 0
        self.max_para_num_words = max_para_num_words
        self.allow_empty_answers = allow_empty_answers
        self.shuffle_paras = shuffle_paras
        self.question_prefix = question_prefix  # Don't strip this. It may have \n
        self.store_sents_in_generated_sentences = store_sents_in_generated_sentences

    def return_model_calls(self):
        return {"llm_qa": self.num_calls}

    def update_state(self, answer, state):
        if not self.allow_empty_answers and answer == "":
            print("WARNING: Generate empty answer.")
            return []
        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=json.dumps(answer), score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state
        return new_state

    def query(self, state, debug=False):
        question = state.data.get_last_question()

        if self.question_prefix:
            assert self.question_prefix.endswith("\n") or self.question_prefix.endswith(" ")
            question = self.question_prefix + question

        titles, paras = state.data["titles"], state.data["paras"]
        zipped_titles_paras = list(zip(titles, paras))
        if self.shuffle_paras:
            random.shuffle(zipped_titles_paras)

        if "paras" in state.data:
            context = [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]
        else:
            context = ""

        self.num_calls += 1
        context_suffix = extract_key_information(state, self.key_info_type)
        answer, facts_used = self.qa_model.ask_question(
            input_question=question, context=context, context_suffix=context_suffix
        )

        if self.extractor_regex and isinstance(answer, str) and self.extractor_regex.match(answer):
            answer = self.extractor_regex.match(answer).group(1)
            if self.extractor_remove_last_fullstop and answer.endswith("."):
                answer = answer[:-1]

        if self.answer_is_numbered_list:
            answer = [re.sub(r"^\d+\.", "", e.strip()).strip() for e in str(answer).split("\n")]
            answer = [e for e in answer if e]
            answer = list(dict.fromkeys(answer).keys())

        if self.store_sents_in_generated_sentences:
            spacy_object = get_spacy_object()
            state.data["generated_sentences"] = [sent.text_with_ws for sent in spacy_object(answer).sents]

        return self.update_state(answer=answer, state=state)


class LLMTitleQAParticipantModel(ParticipantModel):
    def __init__(
        self,
        retriever_host,
        retriever_port,
        retrieval_count,
        source_corpus_name,
        title_question_extractor_regex=".*Titles: (.*?). Question: (.*)",
        answer_extractor_regex=None,
        remove_last_fullstop=False,
        cumulate_titles_paras_in_prefix=None,
        max_para_num_words=350,
        shuffle_paras=False,
        return_both=False,
        next_model=None,
        end_state="[EOQ]",
        **kwargs,
    ):
        self.title_question_extractor_regex = re.compile(title_question_extractor_regex)
        self.answer_extractor_regex = answer_extractor_regex
        if answer_extractor_regex is not None:
            self.answer_extractor_regex = re.compile(answer_extractor_regex)
        self.remove_last_fullstop = remove_last_fullstop
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port
        self.retrieval_count = retrieval_count
        self.source_corpus_name = source_corpus_name
        self.cumulate_titles_paras_in_prefix = cumulate_titles_paras_in_prefix
        self.max_para_num_words = max_para_num_words
        self.shuffle_paras = shuffle_paras
        self.qa_model = LLMQAModel(**kwargs)
        self.next_model = next_model
        self.return_both = return_both
        self.end_state = end_state
        self.num_calls = 0

    def return_model_calls(self):
        return {"llm_title_qa": self.num_calls}

    def update_state(self, answer, state):
        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=json.dumps(answer), score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state
        return new_state

    def query(self, state, debug=False):
        question = state.data.get_last_question()

        match = self.title_question_extractor_regex.match(question)
        if match is None:
            print(f"WARNING: title_question_extractor_regex didn't find any match for: {question}")
            return []

        try:
            titles_str = match.group(1)
            if '"titles":' in titles_str:
                titles_str_parts = [e.strip() for e in titles_str.split("+")]
                titles_parts = [json.loads(_str) for _str in titles_str_parts]
                if all(isinstance(titles_part, list) for titles_part in titles_parts):
                    titles = sum([titles_part for titles_part in titles_parts], [])
                elif all(isinstance(titles_part, dict) for titles_part in titles_parts):
                    titles = sum([titles_part["titles"] for titles_part in titles_parts], [])
                else:
                    raise Exception()
            else:
                titles = json.loads(titles_str)
        except:
            print(f"WARNING: The title couldn't be captured from the question: {question}")
            return []

        try:
            question = match.group(2)
        except:
            print(f"WARNING: The question couldn't be captured from the question: {question}")
            return []

        if not isinstance(titles, list):
            print(f"WARNING: The titles_str {titles_str} couldn't be parsed as a list.")
            return []

        selected_titles = []
        selected_paragraph_texts = []

        for title in titles:
            # Redo the retrieval and get the paragraph.

            params = {
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": title,
                "max_hits_count": self.retrieval_count,
                "document_type": "title",
                "corpus_name": self.source_corpus_name,
            }
            url = self.retriever_host.rstrip("/") + ":" + str(self.retriever_port) + "/retrieve"
            result = safe_post_request(url, params)

            if not result.ok:
                time.sleep(5)
                result = safe_post_request(url, params)

            if not result.ok:
                raise Exception("Unexpected requests error.")

            result = result.json()
            for retrieval in result["retrieval"]:
                if (
                    retrieval["title"].lower() != title.lower()
                    and SequenceMatcher(None, retrieval["title"].lower(), title.lower()).ratio() < 90
                ):
                    print(
                        "WARNING: Title of the requested and returned paragraph don't match well: "
                        + retrieval["title"]
                        + " AND "
                        + title
                    )

                if retrieval["corpus_name"] != self.source_corpus_name:
                    raise Exception(
                        f"The retrieved corpus name {retrieval['corpus_name']} "
                        f"doesn't match {self.source_corpus_name}."
                    )

                paragraph_text = retrieval["paragraph_text"]
                retrieved_title = retrieval["title"]
                if retrieved_title in selected_titles and paragraph_text in selected_paragraph_texts:
                    continue

                selected_titles.append(retrieved_title)
                selected_paragraph_texts.append(paragraph_text)

        zipped_titles_paras = list(zip(selected_titles, selected_paragraph_texts))
        if self.shuffle_paras:
            random.shuffle(zipped_titles_paras)

        context = [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]

        self.num_calls += 1
        answer, _ = self.qa_model.ask_question(input_question=question, context=context)

        if self.answer_extractor_regex and isinstance(answer, str) and self.answer_extractor_regex.match(answer):
            answer = self.answer_extractor_regex.match(answer).group(1)
            if self.remove_last_fullstop and answer.endswith("."):
                answer = answer[:-1]

        if self.return_both:
            output = json.dumps({"titles": titles, "answer": answer})
        else:
            output = json.dumps(answer)

        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=output, score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state

        if self.cumulate_titles_paras_in_prefix:
            _selected_titles = new_state.data.get(self.cumulate_titles_paras_in_prefix + "titles", []) + selected_titles
            _selected_paras = (
                new_state.data.get(self.cumulate_titles_paras_in_prefix + "paras", []) + selected_paragraph_texts
            )
            new_state.data[self.cumulate_titles_paras_in_prefix + "titles"] = _selected_titles + selected_titles
            new_state.data[self.cumulate_titles_paras_in_prefix + "paras"] = _selected_paras + selected_paragraph_texts

        return new_state


class LLMQADecompParticipantModel(QuestionGenParticipant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self, state, debug=False):
        # Is this being called to generate a question?
        if len(state.data.get_current_inference_seq()) == 0 or isinstance(
            state.data.get_last_step(), QuestionAnsweringStep
        ):
            # if there is no previous question or the last step was a QA Step
            new_states = super().query(state=state, debug=debug)
        else:
            # or answer a question
            new_state = state.copy()
            question = state.data.get_last_question()
            # take the last question and a decomposition level
            new_state.data.add_subdecomp(
                StructuredDataInstance(input_data={"qid": state.data["qid"], "query": question, "question": question})
            )
            # then generate the decomposition
            new_states = super().query(state=new_state, debug=debug)
        if not isinstance(new_states, list):
            new_states = [new_states]

        for new_state in new_states:
            # if [EOQ] was generated, i.e. the module is done answering this question
            if new_state.next == self.end_state and not new_state.data.at_root_level():
                last_answer = new_state.data.get_last_answer()
                new_state.data.popup_decomp_level()
                new_state.data.add_answer(QuestionAnsweringStep(answer=last_answer, score=0, participant=state.next))
        return new_states


def date_difference(date1: str, date2: str, units: str = "years"):
    default_date = datetime(3000, 1, 1)
    try:
        date1_datetime = parse(date1, default=default_date)
        date2_datetime = parse(date2, default=default_date)
    except Exception:
        # couldn't parse date
        return None
    # if one doesn't have month set, not usable
    if date1_datetime.year == default_date.year and date1_datetime.month == default_date.month:
        return None
    if date2_datetime.year == default_date.year and date2_datetime.month == default_date.month:
        return None

    if date1_datetime.year == default_date.year and date2_datetime.year != default_date.year:
        # one date is relative and other is not
        date1_datetime = date1_datetime.replace(year=date2_datetime.year)
    elif date2_datetime.year == default_date.year and date1_datetime.year != default_date.year:
        # one date is relative and other is not
        date2_datetime = date2_datetime.replace(year=date1_datetime.year)

    if units == "days":
        return (date1_datetime - date2_datetime).days
    if units == "months":
        return (date1_datetime.year - date2_datetime.year) * 12 + (date1_datetime.month - date2_datetime.month)
    if units == "years":
        # human annotations are often on just the year value
        return date1_datetime.year - date2_datetime.year
    print("Unknown unit:" + units)
    return None


def sort_without_duplicates(arr):
    last_val = None
    output_arr = []
    for key, val in sorted(arr, key=lambda x: x[1]):
        if val == last_val:
            continue
        else:
            output_arr.append((key, val))
            last_val = val
    return output_arr


def sorted_key(arr):
    return [x[0] for x in sort_without_duplicates(arr)]


def sorted_value(arr):
    return [x[1] for x in sort_without_duplicates(arr)]
