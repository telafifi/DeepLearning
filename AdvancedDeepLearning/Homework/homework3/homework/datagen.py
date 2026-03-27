def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    raise NotImplementedError()


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
