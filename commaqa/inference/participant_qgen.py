import logging
import random

from commaqa.inference.prompt_reader import read_prompt
from commaqa.inference.data_instances import QuestionGenerationStep, Task
from commaqa.inference.model_search import ParticipantModel
from commaqa.inference.utils import get_sequence_representation
from commaqa.models.generator import LMGenerator
from commaqa.models.gpt3generator import GPT3Generator
from commaqa.models.llm_client_generator import LLMClientGenerator
from commaqa.inference.odqa import para_to_text

logger = logging.getLogger(__name__)

random.seed(100)  # Don't change.


class QuestionGenParticipant(ParticipantModel):
    def __init__(
        self,
        scale_by_step=1,
        add_eos=False,
        prompt_file="",
        prompt_reader_args=None,
        next_model="execute",
        end_state="[EOQ]",
        use_special_number_format=False,
        add_context=False,
        max_steps=10,
        max_para_num_words=350,
        gen_model="lm",
        shuffle_paras=False,
        **kwargs
    ):
        self.scale_by_step = scale_by_step
        self.add_eos = add_eos
        if prompt_file:
            prompt_reader_args = prompt_reader_args or {}
            prompt_reader_args["file_path"] = prompt_file
            self.prompt = read_prompt(**prompt_reader_args)
        else:
            self.prompt = None
        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0
        self.use_special_number_format = use_special_number_format
        self.add_context = add_context
        self.max_steps = max_steps
        self.gen_model = gen_model
        self.shuffle_paras = shuffle_paras
        self.max_para_num_words = max_para_num_words
        if gen_model == "lm":
            self.generator = LMGenerator(**kwargs)
        elif gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
        elif gen_model == "llm_api":
            self.generator = LLMClientGenerator(**kwargs)
        else:
            raise ValueError("Unknown gen_model: " + gen_model)

    def return_model_calls(self):
        return {self.gen_model + "gen": self.num_calls}

    def query(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param data: should have a dictionary as input containing
          mutable data
        :type data: dict
        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        :raises: ValueError
        """
        ## first checks state of `json_input` to figure out how to format things
        ## the first question
        data = state.data
        question_seq = data.get_current_qseq()
        answer_seq = data.get_current_aseq()
        # avoid long chains since this is costly and most likely an error
        if len(question_seq) >= self.max_steps:
            new_state = state.copy()
            output = self.end_state
            new_state.next = self.end_state
            new_state.data.add_qgen(QuestionGenerationStep(question=output, score=0, participant=state.next))
            return new_state

        if self.use_special_number_format:
            gen_seq = "QC: " + data.get_current_inference_data()["query"]
            for qidx, (ques, ans) in enumerate(zip(question_seq, answer_seq)):
                gen_seq += "\nQ{}: {}\n#{}: {}".format(qidx + 1, ques, qidx + 1, ans)
            gen_seq += "\nQ{}:".format(len(question_seq) + 1)
        else:
            gen_seq = get_sequence_representation(
                origq=data.get_current_inference_data()["query"],
                question_seq=question_seq,
                answer_seq=answer_seq,
                compq_marker="QC: ",
                simpq_marker="\nQS: ",
                answer_marker="\nA: ",
                interq_marker="\nQS: ",
            )

        if not state.data["paras"] and self.add_context:
            print("WARNING: Found no paragraphs in the state but add_context is True.")

        if state.data["paras"] and not self.add_context:
            print("WARNING: Found paragraphs in the state but add_context is False.")

        if self.add_context and "paras" in state.data and state.data["paras"]:
            zipped_titles_paras = list(zip(state.data["titles"], state.data["paras"]))
            if self.shuffle_paras:
                random.shuffle(zipped_titles_paras)
            paras = [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]
            gen_seq = "\n\n".join(paras) + "\n\n" + gen_seq

        if self.prompt:
            gen_seq = self.prompt + "\n\n\n" + gen_seq.strip()

        ## eventual output
        new_states = []
        ## go through generated questions
        output_seq_scores = self.generator.generate_text_sequence(gen_seq)
        self.num_calls += 1
        observed_outputs = set()
        for output_seq, score in output_seq_scores:
            if debug:
                print("--> " + output_seq + " : " + str(score))
            output = output_seq.strip()
            # catch potentially spurious duplicates
            if output in observed_outputs:
                continue
            else:
                observed_outputs.add(output)
            # copy state
            new_state = state.copy()

            # lower is better, same as the scores returned by generate_text_sequence
            assert score >= 0, (
                "Score from generation assumed to be +ve. Got: {}! Needs to be "
                "+ve to ensure monotonically increasing scores as expected by the"
                " search.".format(score)
            )
            new_state._score += score

            new_state.data.add_qgen(QuestionGenerationStep(question=output, score=score, participant=state.next))

            if output == self.end_state:
                new_state.next = self.end_state
            else:
                new_state.data.add_task(Task(task_question=None, task_participant=new_state.next))
                new_state.next = self.next_model

            new_states.append(new_state)

        return new_states
