from pathlib import Path

dataset_path = Path(__file__).resolve().parent.parent / "data" / "SMSSpamCollection"


#load the dataset into a list
def load_dataset():
    
    messages = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            label, message = line.rstrip("\n").split("\t", 1)
            messages.append((label, message))
    return messages

def main():
    messages = load_dataset()

    print(f"Total messages: {len(messages)}")
    print("\nFirst 5 messages:\n")

    for label, message in messages[:5]:
        print(f"Label: {label}")
        print(f"Message: {message}")
        print("-" * 40)


if __name__ == "__main__":
    main()
