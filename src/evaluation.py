from utils.models import load_best_model_from_exp_dir


def main():
    model = load_best_model_from_exp_dir("../logs/runs/2021-06-09/19-14-46")
    print(model)


if __name__ == "__main__":
    main()
