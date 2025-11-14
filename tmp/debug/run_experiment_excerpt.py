
    dataset_manager = TruthfulQADatasetManager(
        dataset_name=config["truthfulqa"].get("dataset_name", "truthful_qa"),
        dataset_config=config["truthfulqa"].get("dataset_config", "generation"),
        cache_dir=config["truthfulqa"].get("cache_dir"),
        seed=config["run"].get("seed", 42),
    )

    splits = _prepare_pipeline_splits(dataset_manager, config)
    _persist_splits(run_dir, splits)

    model_cfg = config["model"]
    loaded = load_causal_model(
        model_cfg["name"],
        dtype=model_cfg.get("dtype"),
        device_map=model_cfg.get("device_map", "auto"),
        max_memory=model_cfg.get("max_memory"),
        revision=model_cfg.get("revision"),
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    primary_device = loaded.primary_device
    model.eval()

    steering_cfg = config.get("steering", {})
    extractor = ActivationExtractor(
        loaded,
        model_cfg["layer"],
        max_length=steering_cfg.get("max_length", 512),
        batch_size=steering_cfg.get("batch_size", 8),
    )

    vector_bank = _build_vector_bank(
        extractor,
        dataset_manager,
        splits,
        steering_cfg,
        seed=config["run"].get("seed", 42),
        run_dir=run_dir,
    )

    mlp_results = {}

    use_mlp = steering_cfg.get("use_mlp", True)
    mlp_mc = None
    mlp_gen = None

    if use_mlp:
        hidden_dim = vector_bank.base_vector.shape[0]
        arch_cfg = config.get("mlp", {}).get("architecture", {})
        param_dtype = next(model.parameters()).dtype  # Get model's dtype
        mlp_mc = SteeringMLP(
            input_dim=hidden_dim,
            hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
            dropout=arch_cfg.get("dropout", 0.1),
        ).to(primary_device, dtype=param_dtype)  # Match model dtype
        mlp_gen = SteeringMLP(
            input_dim=hidden_dim,
            hidden_multiplier=arch_cfg.get("hidden_multiplier", 2.0),
            dropout=arch_cfg.get("dropout", 0.1),
        ).to(primary_device, dtype=param_dtype)  # Match model dtype

        mlp_cfg = config.get("mlp", {})
        mc_cfg = MCTrainingConfig(**mlp_cfg.get("mc_training", {}))
        gen_cfg = GenTrainingConfig(**mlp_cfg.get("gen_training", {}))

        LOGGER.info("Training MC MLP: %s", mc_cfg)
        mc_history = train_mc_mlp(
            mlp_mc,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset_manager,
            train_indices=splits.train,
            vector_bank=vector_bank,
            layer_index=model_cfg["layer"],
            primary_device=primary_device,
            max_length=steering_cfg.get("max_length", 512),
            config=mc_cfg,
            seed=config["run"].get("seed", 42) + 1,
        )
        mlp_results["mc"] = mc_history
        if mc_history.get("loss"):
            torch.save(
                mlp_mc.state_dict(),
                run_dir / "vectors" / "mlp_mc_state_dict.pt",
            )
        else:
            mlp_mc = None

        LOGGER.info("Training Generation MLP: %s", gen_cfg)
        gen_history = train_gen_mlp(
            mlp_gen,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset_manager,
            train_indices=splits.train,
            vector_bank=vector_bank,
            layer_index=model_cfg["layer"],
            primary_device=primary_device,
            max_length=steering_cfg.get("max_length", 512),
            config=gen_cfg,
            seed=config["run"].get("seed", 42) + 2,
        )
        mlp_results["gen"] = gen_history
        torch.save(
            mlp_gen.state_dict(), run_dir / "vectors" / "mlp_gen_state_dict.pt"
        )

        with (run_dir / "training_history.json").open("w") as f:
            json.dump(mlp_results, f, indent=2)

    eval_cfg = config.get("evaluation", {})
    judge = _maybe_build_judge(eval_cfg)
    semantic_judge = _maybe_build_semantic(eval_cfg)

    evaluation = _run_evaluations(
        model,
        tokenizer,
        dataset_manager,
        splits,
        vector_bank,
        mlp_mc,
        mlp_gen,
        config,
        judge,
        semantic_judge,
        primary_device,
        run_dir,
    )

