from typing import List, Dict, Tuple, Union, Any
from functools import lru_cache
import math
import json
import random
import copy
import re


random.seed(13370)  # Don't change.


def safe_sample(items: List[Any], count: int) -> List[Any]:
    count = min(count, len(items))
    return random.sample(items, count) if count > 0 else []


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


@lru_cache(maxsize=None)
def get_spacy_object():
    import spacy

    return spacy.load("en_core_web_sm")


def clip_paragraph_text(paragraph_text: str, max_tokens: int = 250) -> str:
    spacy_object = get_spacy_object()
    paragraph_object = spacy_object(paragraph_text)
    paragraph_sents = paragraph_object.sents

    clipped_paragraph_tokens = 0
    clipped_paragraph_text = ""
    for sent in paragraph_sents:
        if clipped_paragraph_tokens + len(sent) >= max_tokens:
            break
        clipped_paragraph_text += sent.text_with_ws
        clipped_paragraph_tokens += len(sent)
    return clipped_paragraph_text


def clip_paragraphs(paragraphs: List[Dict], max_tokens: int = 250):
    paragraphs = copy.deepcopy(paragraphs)
    for paragraph in paragraphs:
        if paragraph["is_supporting"] or paragraph["is_pinned"]:
            continue

        paragraph_text = paragraph["paragraph_text"]
        clipped_paragraph_text = clip_paragraph_text(paragraph_text, max_tokens)
        paragraph["paragraph_text"] = clipped_paragraph_text

    return paragraphs


class PromptGenerator:
    def __init__(
        self,
        input_file_path: str,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
    ):
        self._instances = read_jsonl(input_file_path)
        self._demonstration_delimiter = demonstration_delimiter
        self._one_demonstration_per_instance = one_demonstration_per_instance

    def _generate(self, instance: Dict) -> str:
        raise NotImplementedError

    def generate(self) -> str:
        def instance_to_header(instance):
            return "# METADATA: " + json.dumps({"qid": instance["question_id"]})

        all_demonstrations_with_headers = []
        for instance in self._instances:
            local_demonstrations_with_headers = [
                "\n".join([instance_to_header(instance), demonstration]).strip()
                for demonstration in self._generate(instance)
            ]
            if len(local_demonstrations_with_headers) > 1 and self._one_demonstration_per_instance:
                all_demonstrations_with_headers.append(random.choice(local_demonstrations_with_headers))
            else:
                all_demonstrations_with_headers += local_demonstrations_with_headers

        generated_output = self._demonstration_delimiter.join(all_demonstrations_with_headers)
        return generated_output


class QuestionDecompositionPromptGenerator(PromptGenerator):
    def _generate(self, instance: Dict) -> List[Dict]:
        main_question_text = instance["question_text"]
        main_answer_text = instance["answers_objects"][0]["spans"][0]
        reasoning_steps = instance["reasoning_steps"]

        relevant_steps_info = []
        all_gold_titles = []
        for reasoning_step in reasoning_steps:
            sub_question_answer = reasoning_step["sub_question_answer"]
            paragraphs = reasoning_step["paragraphs"]

            assert len(paragraphs) == 1
            paragraph = paragraphs[0]

            if sub_question_answer is None:
                assert paragraph["title"] is None

            if paragraph["title"] is None and sub_question_answer is not None:
                print("WARNING: The following reasoning step has sub_question_answer but no title.")
                print(reasoning_step)

            if sub_question_answer is None or paragraph["title"] is None:
                continue

            assert len(sub_question_answer) == 2

            gold_title = paragraph["title"]
            relevant_steps_info.append(
                {
                    "gold_title": gold_title,
                    "question": sub_question_answer[0],
                    "answer": sub_question_answer[1],
                }
            )
            all_gold_titles.append(gold_title)

        # Gather Distractor Paragraphs
        distractor_paragraphs = [
            paragraph
            for paragraph in instance["contexts"]
            if not paragraph["is_supporting"] and paragraph["title"] not in all_gold_titles
        ]
        distractor_titles = [para["title"] for para in distractor_paragraphs]

        # Put the prompt together
        prompt_main_question_text = f"QC: {main_question_text}"
        prompt_step_texts = []
        all_local_titles = []
        num_original_distractors = len(distractor_titles)
        for relevant_step_info in relevant_steps_info:
            gold_title = relevant_step_info["gold_title"]
            question = relevant_step_info["question"]
            answer = relevant_step_info["answer"]
            local_distractor_titles = []
            max_count = math.ceil(num_original_distractors / len(relevant_steps_info))
            for _ in range(random.choice(range(1, max_count))):
                local_distractor_titles.append(distractor_titles.pop())
            prompt_step_text = f"QS: (select) [retrieve_odqa] {question}\n"
            local_titles = local_distractor_titles + [gold_title]
            random.shuffle(local_titles)
            if isinstance(answer, str):
                answer = [answer]
            prompt_step_text += "A: " + json.dumps({"titles": local_titles, "answer": answer}, ensure_ascii=False)
            prompt_step_texts.append(prompt_step_text)
            all_local_titles += local_titles

        prompt_steps_text = "\n".join(prompt_step_texts)
        all_local_titles_str = json.dumps(all_local_titles, ensure_ascii=False)
        prompt_final_step_text = (
            f"QS: (select) [multihop_titleqa] Titles: {all_local_titles_str}. Question: {main_question_text}\n"
        )
        prompt_final_step_text += f"A: {json.dumps([main_answer_text])}\n"
        prompt_final_step_text += "QS: [EOQ]"

        pinned_paragraphs = instance.get("pinned_contexts", [])
        context_text = "\n\n".join(
            [
                "Wikipedia Title: "
                + paragraph["title"]
                + "\n"
                + paragraph["paragraph_text"].strip().replace("\n", " ").strip()
                for paragraph in pinned_paragraphs
            ]
        ).strip()

        demonstration = "\n".join(
            [context_text, "", prompt_main_question_text, prompt_steps_text, prompt_final_step_text]
        ).strip()

        return [demonstration]


class RetrieveODQAPromptGenerator(PromptGenerator):
    def _generate(self, instance: Dict) -> List[Dict]:
        reasoning_steps = instance["reasoning_steps"]

        relevant_steps_info = []
        all_gold_titles = []
        for reasoning_step in reasoning_steps:
            sub_question_answer = reasoning_step["sub_question_answer"]
            paragraphs = reasoning_step["paragraphs"]

            assert len(paragraphs) == 1
            paragraph = paragraphs[0]

            if sub_question_answer is None:
                assert paragraph["title"] is None

            if paragraph["title"] is None and sub_question_answer is not None:
                print("WARNING: The following reasoning step has sub_question_answer but no title.")
                print(reasoning_step)

            if sub_question_answer is None or paragraph["title"] is None:
                continue

            assert len(sub_question_answer) == 2

            gold_title = paragraph["title"]
            relevant_steps_info.append(
                {
                    "gold_title": gold_title,
                    "question": sub_question_answer[0],
                    "answer": sub_question_answer[1],
                }
            )
            all_gold_titles.append(gold_title)

        # Gather Distractor Paragraphs
        distractor_paragraphs = [
            paragraph
            for paragraph in instance["contexts"]
            if not paragraph["is_supporting"] and paragraph["title"] not in all_gold_titles
        ]
        distractor_titles = [para["title"] for para in distractor_paragraphs]

        # Put the prompt together
        num_original_distractors = len(distractor_titles)

        demonstrations = []
        for relevant_step_info in relevant_steps_info:
            gold_title = relevant_step_info["gold_title"]
            question = relevant_step_info["question"]
            answer = relevant_step_info["answer"]
            local_distractor_titles = []
            max_count = math.ceil(num_original_distractors / len(relevant_steps_info))
            for _ in range(random.choice(range(1, max_count))):
                local_distractor_titles.append(distractor_titles.pop())
            prompt_step_text = f"QS: (select) [retrieve_odqa] {question}\n"
            local_titles = local_distractor_titles + [gold_title]
            random.shuffle(local_titles)
            if isinstance(answer, str):
                answer = [answer]
            prompt_step_text += "A: " + json.dumps({"titles": local_titles, "answer": answer}, ensure_ascii=False)

            local_titles_str = json.dumps(local_titles, ensure_ascii=False)
            final_answer_str = json.dumps({"titles": local_titles, "answer": answer}, ensure_ascii=False)

            demonstration = ""
            demonstration += f"QC: {question}\n"
            demonstration += f"QS: (select) [retrieve] {question}\n"
            demonstration += f"A: {local_titles_str}\n"
            demonstration += f"QS: (select) [singlehop_titleqa] Titles: {local_titles_str}. Question: {question}\n"
            demonstration += f"A: {final_answer_str}\n"
            demonstration += "QS: [EOQ]"
            demonstration = demonstration.strip()
            demonstrations.append(demonstration)

        pinned_paragraphs = instance.get("pinned_contexts", [])
        context_text = "\n\n".join(
            [
                "Wikipedia Title: "
                + paragraph["title"]
                + "\n"
                + paragraph["paragraph_text"].strip().replace("\n", " ").strip()
                for paragraph in pinned_paragraphs
            ]
        ).strip()

        demonstrations = ["\n\n".join([context_text, demonstration]).strip() for demonstration in demonstrations]

        return demonstrations


class SinglehopTitleqaPromptGenerator(PromptGenerator):
    def __init__(
        self,
        input_file_path: str,
        distractor_count: Union[int, Tuple[int, int]] = 0,
        max_paragraph_tokens: int = 250,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
    ):
        assert isinstance(distractor_count, int)
        assert isinstance(max_paragraph_tokens, int)
        self._distractor_count = distractor_count
        self._max_paragraph_tokens = max_paragraph_tokens
        super().__init__(input_file_path, demonstration_delimiter, one_demonstration_per_instance)

    def _generate(self, instance: Dict) -> List[Dict]:
        reasoning_steps = instance["reasoning_steps"]

        # Gather Gold Paragraphs and Gold CoT
        gold_paragraphs = instance.get("pinned_contexts", [])
        taken_gold_title_paras = []
        for paragraph in gold_paragraphs:
            paragraph["is_pinned"] = True
            paragraph["is_supporting"] = True
            title_para = (paragraph["title"], paragraph["paragraph_text"])
            taken_gold_title_paras.append(title_para)

        sub_question_answers = []
        for reasoning_step in reasoning_steps:
            sub_question_answer = reasoning_step["sub_question_answer"]
            paragraphs = reasoning_step["paragraphs"]

            assert len(paragraphs) == 1
            paragraph = paragraphs[0]

            if sub_question_answer is None:
                assert paragraph["title"] is None

            if paragraph["title"] is None and sub_question_answer is not None:
                print("WARNING: The following reasoning step has sub_question_answer but no title.")
                print(reasoning_step)

            if sub_question_answer is None or paragraph["title"] is None:
                continue

            assert len(sub_question_answer) == 2

            sub_question_answer = reasoning_step["sub_question_answer"]
            sub_question_answers.append((sub_question_answer[0], sub_question_answer[1]))

            title_para = (paragraph["title"], paragraph["paragraph_text"])

            if title_para in taken_gold_title_paras:
                continue

            paragraph["is_supporting"] = True
            paragraph["is_pinned"] = False

            gold_paragraphs.append(paragraph)
            taken_gold_title_paras.append(title_para)

        # Gather Distractor Paragraphs
        distractor_paragraphs = []

        if isinstance(self._distractor_count, int):
            distractor_count = self._distractor_count
        else:
            distractor_count = random.randint(*self._distractor_count)
        candidate_distractor_paragraphs = [
            paragraph for paragraph in instance["contexts"] if not paragraph["is_supporting"]
        ]
        candidate_distractor_paragraphs = [
            paragraph
            for paragraph in candidate_distractor_paragraphs
            if (paragraph["title"], paragraph["paragraph_text"]) not in taken_gold_title_paras
        ]
        for paragraph in candidate_distractor_paragraphs:
            assert not paragraph.get("is_supporting", False)
            paragraph["is_supporting"] = False
            paragraph["is_pinned"] = False
        distractor_paragraphs = safe_sample(candidate_distractor_paragraphs, distractor_count)

        # Put all paragraphs together in context
        all_paragraphs = gold_paragraphs + distractor_paragraphs
        all_paragraphs = clip_paragraphs(all_paragraphs, max_tokens=self._max_paragraph_tokens)

        random.shuffle(all_paragraphs)
        all_paragraphs = sorted(all_paragraphs, key=lambda e: int(e["is_pinned"]), reverse=True)

        context_text = "\n\n".join(
            [
                "Wikipedia Title: "
                + paragraph["title"]
                + "\n"
                + paragraph["paragraph_text"].strip().replace("\n", " ").strip()
                for paragraph in all_paragraphs
            ]
        ).strip()

        question_answer_texts = []
        for question, answer in sub_question_answers:
            if isinstance(answer, str):
                answer = [answer]
            question_answer_text = f"Q: {question}\n"
            question_answer_text += f"A: {json.dumps(answer, ensure_ascii=False)}"
            question_answer_texts.append(question_answer_text)
        question_answer_text = "\n\n".join(question_answer_texts)

        demonstration = "\n".join([context_text, "", question_answer_text]).strip()

        return [demonstration]


class MultihopTitleqaPromptGenerator(PromptGenerator):
    def __init__(
        self,
        input_file_path: str,
        qa_type: str,
        distractor_count: Union[int, Tuple[int, int]] = 0,
        max_paragraph_tokens: int = 250,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
    ):
        assert isinstance(distractor_count, int)
        assert isinstance(max_paragraph_tokens, int)
        self._qa_type = qa_type
        self._distractor_count = distractor_count
        self._max_paragraph_tokens = max_paragraph_tokens
        super().__init__(input_file_path, demonstration_delimiter, one_demonstration_per_instance)

    def _generate(self, instance: Dict) -> List[Dict]:
        cot_sents = []
        reasoning_steps = instance["reasoning_steps"]

        # Gather Gold Paragraphs and Gold CoT
        gold_paragraphs = instance.get("pinned_contexts", [])
        taken_gold_title_paras = []
        for paragraph in gold_paragraphs:
            paragraph["is_pinned"] = True
            paragraph["is_supporting"] = True
            title_para = (paragraph["title"], paragraph["paragraph_text"])
            taken_gold_title_paras.append(title_para)

        for reasoning_step in reasoning_steps:
            cot_sents.append(reasoning_step["cot_sent"])

            sub_question_answer = reasoning_step["sub_question_answer"]
            paragraphs = reasoning_step["paragraphs"]

            assert len(paragraphs) == 1
            paragraph = paragraphs[0]

            if sub_question_answer is None:
                assert paragraph["title"] is None

            if paragraph["title"] is None and sub_question_answer is not None:
                print("WARNING: The following reasoning step has sub_question_answer but no title.")
                print(reasoning_step)

            if sub_question_answer is None or paragraph["title"] is None:
                continue

            assert len(sub_question_answer) == 2

            title_para = (paragraph["title"], paragraph["paragraph_text"])

            if title_para in taken_gold_title_paras:
                continue

            paragraph["is_supporting"] = True
            paragraph["is_pinned"] = False

            gold_paragraphs.append(paragraph)
            taken_gold_title_paras.append(title_para)

        # Gather Distractor Paragraphs
        distractor_paragraphs = []

        if isinstance(self._distractor_count, int):
            distractor_count = self._distractor_count
        else:
            distractor_count = random.randint(*self._distractor_count)
        candidate_distractor_paragraphs = [
            paragraph for paragraph in instance["contexts"] if not paragraph["is_supporting"]
        ]
        candidate_distractor_paragraphs = [
            paragraph
            for paragraph in candidate_distractor_paragraphs
            if (paragraph["title"], paragraph["paragraph_text"]) not in taken_gold_title_paras
        ]
        for paragraph in candidate_distractor_paragraphs:
            assert not paragraph.get("is_supporting", False)
            paragraph["is_supporting"] = False
            paragraph["is_pinned"] = False
        distractor_paragraphs = safe_sample(candidate_distractor_paragraphs, distractor_count)

        # Put all paragraphs together in context
        all_paragraphs = gold_paragraphs + distractor_paragraphs
        all_paragraphs = clip_paragraphs(all_paragraphs, max_tokens=self._max_paragraph_tokens)

        random.shuffle(all_paragraphs)
        all_paragraphs = sorted(all_paragraphs, key=lambda e: int(e["is_pinned"]), reverse=True)

        context_text = "\n\n".join(
            [
                "Wikipedia Title: "
                + paragraph["title"]
                + "\n"
                + paragraph["paragraph_text"].strip().replace("\n", " ").strip()
                for paragraph in all_paragraphs
            ]
        ).strip()

        full_question_text = instance["question_text"]
        assert len(instance["answers_objects"]) == 1
        assert len(instance["answers_objects"][0]["spans"]) == 1
        full_answer_text = instance["answers_objects"][0]["spans"][0]

        if self._qa_type == "direct":
            answer_or_cot_text = json.dumps([full_answer_text], ensure_ascii=False)
        elif self._qa_type == "cot":
            answer_or_cot_text = re.sub(r" +", " ", " ".join(cot_sents))
        else:
            raise Exception(f"Encountered unknown choice of qa_type {self._qa_type}.")

        question_answer_text = f"Q: {full_question_text}\n"
        question_answer_text += f"A: {answer_or_cot_text}"

        demonstration = "\n".join([context_text, "", question_answer_text]).strip()

        return [demonstration]
