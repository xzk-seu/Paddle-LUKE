from task.squad.train import load_examples, get_args_parser
from luke.tokenizer import LukeTokenizer


def main():
    args = get_args_parser().parse_args()
    tokenizer = LukeTokenizer.from_pretrained("weight/pd/luke-large-finetuned-open-entity")
    train_dataloader, _, _, _ = load_examples(args, tokenizer, is_evaluate=True)
    for i in train_dataloader:
        print(i)


if __name__ == "__main__":
    main()
