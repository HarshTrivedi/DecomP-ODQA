import os
import argparse

from prompt_generator.common import (
    QuestionDecompositionPromptGenerator,
    RetrieveODQAPromptGenerator,
    SinglehopTitleqaPromptGenerator,
    MultihopTitleqaPromptGenerator,
)


def get_question_decomposition_prompt_generator_args_and_names(dataset_name: str):
    prompt_generator_args = {}
    prompt_name = "question_decomposition.txt"
    return [{"generator_args": prompt_generator_args, "name": prompt_name}]


def get_retrieve_odqa_prompt_generator_args_and_names(dataset_name: str):
    prompt_generator_args = {}
    prompt_name = "retrieve_odqa.txt"
    return [{"generator_args": prompt_generator_args, "name": prompt_name}]


def get_singlehop_titleqa_prompt_generator_args_and_names(dataset_name: str):
    prompt_generator_args_and_names = []
    for distractor_count in (0, 1, 2, 3):
        prompt_generator_args = {"distractor_count": distractor_count, "max_paragraph_tokens": 250}
        prompt_name = f"singlehop_titleqa_with_all_gold_paras_and_{distractor_count}_distractor.txt"
        prompt_generator_args_and_names.append({"generator_args": prompt_generator_args, "name": prompt_name})
    return prompt_generator_args_and_names


def get_multihop_direct_titleqa_prompt_generator_args_and_names(dataset_name: str):
    prompt_generator_args_and_names = []
    for distractor_count in (0, 1, 2, 3):
        prompt_generator_args = {"distractor_count": distractor_count, "max_paragraph_tokens": 250, "qa_type": "direct"}
        prompt_name = f"multihop_direct_titleqa_with_all_gold_paras_and_{distractor_count}_distractor.txt"
        prompt_generator_args_and_names.append({"generator_args": prompt_generator_args, "name": prompt_name})
    return prompt_generator_args_and_names


def get_multihop_cot_titleqa_prompt_generator_args_and_names(dataset_name: str):
    prompt_generator_args_and_names = []
    for distractor_count in (0, 1, 2, 3):
        prompt_generator_args = {"distractor_count": distractor_count, "max_paragraph_tokens": 250, "qa_type": "cot"}
        prompt_name = f"multihop_cot_titleqa_with_all_gold_paras_and_{distractor_count}_distractor.txt"
        prompt_generator_args_and_names.append({"generator_args": prompt_generator_args, "name": prompt_name})
    return prompt_generator_args_and_names


def main():
    parser = argparse.ArgumentParser(description="Generate prompts.")
    parser.add_argument(
        "dataset_name", type=str, help="dataset_name", choices={"hotpotqa", "2wikimultihopqa", "musique"}
    )
    args = parser.parse_args()

    input_file_path = os.path.join("processed_data", args.dataset_name, "annotated_only_train.jsonl")
    output_directory = os.path.join("prompts", args.dataset_name)

    task_names = [
        "question_decomposition",
        "retrieve_odqa",
        "multihop_direct_qa",
        "multihop_cot_qa",
        "singlehop_titleqa",
    ]

    for task_name in task_names:
        if args.task_name == "question_decomposition":
            args_name_generator = get_question_decomposition_prompt_generator_args_and_names
            prompt_generator_cls = QuestionDecompositionPromptGenerator
        elif args.task_name == "retrieve_odqa":
            args_name_generator = get_retrieve_odqa_prompt_generator_args_and_names
            prompt_generator_cls = RetrieveODQAPromptGenerator
        elif args.task_name == "singlehop_titleqa":
            args_name_generator = get_singlehop_titleqa_prompt_generator_args_and_names
            prompt_generator_cls = SinglehopTitleqaPromptGenerator
        elif args.task_name == "multihop_direct_qa":
            args_name_generator = get_multihop_direct_titleqa_prompt_generator_args_and_names
            prompt_generator_cls = MultihopTitleqaPromptGenerator
        elif args.task_name == "multihop_cot_qa":
            args_name_generator = get_multihop_cot_titleqa_prompt_generator_args_and_names
            prompt_generator_cls = MultihopTitleqaPromptGenerator
        else:
            raise Exception(f"Invalid task_name {task_name}")

        for prompt_args_and_name in args_name_generator(args.dataset_name):
            generator_args = prompt_args_and_name["generator_args"]
            generator_args["input_file_path"] = input_file_path
            prompt_generator = prompt_generator_cls(**generator_args)

            output_file_name = prompt_args_and_name["name"]
            output_file_path = os.path.join(output_directory, output_file_name)

            prompt_args_and_name.pop("generator_args")
            prompt_args_and_name.pop("name")
            prompt_args_and_name.pop("max_paragraph_tokens")
            if prompt_args_and_name:
                raise Exception("Looks like prompt_args_and_name has extra unused args.")

            print(f"Writing in {output_file_path}")
            with open(output_file_path, "w") as file:
                file.write(prompt_generator.generate())


if __name__ == "__main__":
    main()
