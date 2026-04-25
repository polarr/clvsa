from types import ModuleType

def get_dataset_module(data_source: str) -> ModuleType:
    data_source = data_source.lower()

    if data_source in {"equity", "stock"}:
        from dataset import load_dataset
        return load_dataset

    if data_source == "futures":
        from dataset import load_futures
        return load_futures

    raise ValueError(
        f"Unknown data_source={data_source}. "
        "Expected one of: equity/stock, futures."
    )