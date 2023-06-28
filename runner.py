import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Wrapper around run.py to make experimentation easier.")
    parser.add_argument("system", type=str, choices=("decomp_context", "no_decomp_context", "no_context"))
    parser.add_argument("reader", type=str, choices=("direct", "cot"))
    parser.add_argument("model", type=str, choices=("codex", "flan-t5-xxl", "flan-t5-xl", "flan-t5-large"))
    parser.add_argument("dataset", type=str, choices=("hotpotqa", "2wikimultihopqa", "musique"))
    parser.add_argument(
        "command",
        type=str,
        help="command",
        choices={
            "print",
            "write",
            "verify",
            "predict",
            "evaluate",
            "track",
            "summarize",
            "ground_truth_check",
            "backup",
            "print_backup",
            "recover_backup",
            "delete_predictions",
        },
    )
    parser.add_argument(
        "--prompt_set",
        type=str,
        help="prompt_set",
        choices={"1", "2", "3", "aggregate"},
        default="1",
    )
    parser.add_argument("--dry_run", action="store_true", default=False, help="dry_run")
    parser.add_argument("--use_backup", action="store_true", default=False, help="pass --use_backup flag")
    parser.add_argument("--skip_evaluation_path", action="store_true", default=False, help="skip_evaluation_path")
    parser.add_argument("--eval_test", action="store_true", default=False, help="eval_test")
    parser.add_argument("--best", action="store_true", default=False, help="pass --best flag")
    parser.add_argument("--skip_if_exists", action="store_true", default=False, help="skip evaluation of it exists.")
    parser.add_argument(
        "--only_print", action="store_true", default=False, help="print only for eval, ignore otherwise."
    )
    parser.add_argument("--force", action="store_true", default=False, help="force predict if it exists")
    parser.add_argument(
        "--official", action="store_true", default=False, help="use official evaluation for evaluate and summarize."
    )
    args = parser.parse_args()

    experiment_name = "_".join([args.system, args.reader, "qa", args.model.replace("-", "_"), args.dataset])
    instantiation_scheme = args.system + "_qa_" + ("codex" if args.model.startswith("codex") else "flan_t5")

    run_command_array = [
        f"python run.py {args.command} {experiment_name} --instantiation_scheme {instantiation_scheme} --prompt_set {args.prompt_set}",
    ]

    if args.command in ("write", "predict", "evaluate", "print", "summarize") and args.best:
        run_command_array += ["--best"]

    if args.command == "write":
        run_command_array += ["--no_diff"]

    if (
        args.command in ("predict", "evaluate", "track", "summarize", "ground_truth_check")
        and not args.skip_evaluation_path
    ) or args.best:
        set_name = "test" if args.eval_test else "dev"
        evaluation_path = os.path.join("processed_data", args.dataset, f"{set_name}_subsampled.jsonl")
        run_command_array += [f"--evaluation_path {evaluation_path}"]

    if args.command in ("predict"):
        run_command_array.append("--skip_if_exists --silent")

    if args.command in ("predict", "evaluate", "track", "summarize", "ground_truth_check") and args.use_backup:
        run_command_array += ["--use_backup"]

    if args.command == "predict" and args.force:
        run_command_array += ["--force"]

    if args.command == "evaluate" and args.skip_if_exists:
        run_command_array += ["--skip_if_exists"]

    if args.command == "evaluate" and args.only_print:
        run_command_array += ["--only_print"]

    if args.command in ("evaluate", "summarize") and args.official:
        run_command_array += ["--official"]

    assert args.dataset in experiment_name

    print("", flush=True)
    message = f"Experiment Name: {experiment_name}"
    print("*" * len(message), flush=True)
    print(message, flush=True)
    print("*" * len(message), flush=True)

    run_command_str = " ".join(run_command_array)
    print(run_command_str + "\n", flush=True)
    if not args.dry_run:
        subprocess.call(run_command_str, shell=True)


if __name__ == "__main__":
    main()
