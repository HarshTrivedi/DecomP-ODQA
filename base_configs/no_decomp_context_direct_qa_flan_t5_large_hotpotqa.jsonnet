# Set dataset:
local dataset = "hotpotqa";
local retrieval_corpus_name = dataset;
local add_pinned_paras = false;
local valid_qids = {
    "hotpotqa": ["5ab92dba554299131ca422a2","5a7bbc50554299042af8f7d0","5add363c5542990dbb2f7dc8","5a835abe5542996488c2e426","5ae0185b55429942ec259c1b","5a790e7855429970f5fffe3d","5a754ab35542993748c89819","5a89c14f5542993b751ca98a","5abb14bd5542992ccd8e7f07","5a89d58755429946c8d6e9d9","5a88f9d55542995153361218","5a90620755429933b8a20508","5a77acab5542992a6e59df76","5abfb3435542990832d3a1c1","5a8f44ab5542992414482a25","5adfad0c554299603e41835a","5a7fc53555429969796c1b55","5a8ed9f355429917b4a5bddd","5ac2ada5554299657fa2900d","5a758ea55542992db9473680"],
    "2wikimultihopqa": ["5811079c0bdc11eba7f7acde48001122","97954d9408b011ebbd84ac1f6bf848b6","35bf3490096d11ebbdafac1f6bf848b6","c6805b2908a911ebbd80ac1f6bf848b6","5897ec7a086c11ebbd61ac1f6bf848b6","e5150a5a0bda11eba7f7acde48001122","a5995da508ab11ebbd82ac1f6bf848b6","cdbb82ec0baf11ebab90acde48001122","f44939100bda11eba7f7acde48001122","4724c54e08e011ebbda1ac1f6bf848b6","f86b4a28091711ebbdaeac1f6bf848b6","13cda43c09b311ebbdb0ac1f6bf848b6","228546780bdd11eba7f7acde48001122","c6f63bfb089e11ebbd78ac1f6bf848b6","1ceeab380baf11ebab90acde48001122","8727d1280bdc11eba7f7acde48001122","f1ccdfee094011ebbdaeac1f6bf848b6","79a863dc0bdc11eba7f7acde48001122","028eaef60bdb11eba7f7acde48001122","af8c6722088b11ebbd6fac1f6bf848b6"],
    "musique": ["2hop__323282_79175","2hop__292995_8796","2hop__439265_539716","4hop3__703974_789671_24078_24137","2hop__154225_727337","2hop__861128_15822","3hop1__858730_386977_851569","2hop__642271_608104","2hop__387702_20661","2hop__131516_53573","2hop__496817_701819","2hop__804754_52230","3hop1__61746_67065_43617","3hop1__753524_742157_573834","2hop__427213_79175","3hop1__443556_763924_573834","2hop__782642_52667","2hop__102217_58400","2hop__195347_20661","4hop3__463724_100414_35260_54090"],
}[dataset];
local prompt_reader_args = {
    "filter_by_key_values": {
        "qid": valid_qids
    },
    "order_by_key": "qid",
    "estimated_generation_length": 200,
    "shuffle": false,
    "model_length_limit": 2000,
};
local bm25_retrieval_count = 4;
local distractor_count = 1;

{
    "start_state": "generate_titles",
    "end_state": "[EOQ]",
    "models": {
        "generate_titles": {
            "name": "retriever",
            "next_model": "generate_main_question",
            "retrieval_type": "bm25",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": bm25_retrieval_count,
            "query_source": "original_question",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title__paragraph_text",
            "end_state": "[EOQ]",
        },
        "generate_main_question": {
            "name": "copy_question",
            "next_model": "answer_main_question",
            "eoq_after_n_calls": 1,
            "end_state": "[EOQ]",
        },
        "answer_main_question": {
            "name": "llmqa",
            "next_model": null,
            "prompt_file": "prompts/" + dataset + "/multihop_direct_titleqa_with_all_gold_paras_and_" + distractor_count + "_distractor.txt",
            "prompt_reader_args": prompt_reader_args,
            "end_state": "[EOQ]",
            "gen_model": "llm_api",
            "model_name": "google/flan-t5-large",
            "model_tokens_limit": 2000,
            "max_length": 200,
            "add_context": true,
        },
        "extract": {
            "name": "answer_extractor",
            "query_source": "last_answer",
            "regex": ".* answer is:? (.*)\\.?",
            "match_all_on_failure": true,
            "remove_last_fullstop": true,
        }
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": add_pinned_paras,
    },
    "prediction_type": "answer"
}