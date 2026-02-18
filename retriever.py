def get_retriever(vectordb, config):
    retriever_cfg = config["retriever"]

    retriever_type = retriever_cfg.get("type", "similarity")
    top_k = retriever_cfg.get("top_k", 3)

    if retriever_type == "similarity":
        return vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

    elif retriever_type == "mmr":
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": retriever_cfg.get(config["retriever"]["fetch_k"]),
                "lambda_mult": retriever_cfg.get("lambda_mult", 0.7),
            },
        )

    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")
