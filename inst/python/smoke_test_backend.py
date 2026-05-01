import pandas as pd

from survivehr_backend import train_pretrain_model, train_finetune_model, predict_next_events


def main() -> None:
    events = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 2, 2, 2],
            "event": ["T2D", "STATIN", "BP_CHECK", "T2D", "ACEI", "CVD"],
            "age": [58.2, 58.7, 59.1, 62.0, 62.5, 63.0],
            "value": [None, None, 132.0, None, None, None],
        }
    )

    static_df = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "sex": [1, 0],
            "imd": [3, 5],
        }
    )

    targets = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "target_event": ["CVD", "CVD"],
            "target_age": [60.0, 63.3],
            "target_value": [None, None],
        }
    )

    config = {
        "block_size": 16,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "surv_layer": "competing-risk",
    }

    pre = train_pretrain_model(events, static_df=static_df, config=config)
    ft = train_finetune_model(
        events_df=events,
        targets_df=targets,
        outcomes=["CVD"],
        risk_model="competing-risk",
        static_df=static_df,
        config=config,
        pretrained_bundle=pre,
    )

    pred = predict_next_events(ft, events_df=events, static_df=static_df, max_new_tokens=1)
    print(pred.head())


if __name__ == "__main__":
    main()
