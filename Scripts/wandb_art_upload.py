import wandb

entity = 'evdcush'
project = 'enso'

run = wandb.init(
    entity=entity,
    project=project,
    job_type='chore',
)

artifact_name = f'{wandb.run.name}-{wandb.run.id}-model-pytorch'
art_fpath = 'llava-1.6-swoleaf.pth'

art = wandb.Artifact(
    name=artifact_name,
    type='model',
    metadata=dict(
        file_source='hf-blah-blah',
    ),
)

art.add_file(
    local_path=art_fpath,
    name='model.pth',
)

wandb.log_artifact(
    artifact_or_path=art,
)
wandb.finish()
