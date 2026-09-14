from kfp.dsl import Input, Model, component


@component(
    # Esta imagen que queremos usasr
    base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-6:latest",
    packages_to_install=["google-cloud-aiplatform"],
)
def upload_model(project_id: str, location: str, model: Input[Model]):
    from google.cloud import aiplatform

    aiplatform.init()
    aiplatform.Model.upload_scikit_learn_model_file(
        model_file_path=model.path, display_name="IrisModelV3", project=project_id
    )
