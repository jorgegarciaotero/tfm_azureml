from azure.ai.ml import MLClient, load_component
from azure.identity import DefaultAzureCredential

# 1. Conexión al workspace
ml = MLClient(
    DefaultAzureCredential(),
    subscription_id="0d509562-ae9c-4798-b0c6-42a19eeebe22",
    resource_group_name="intelligent_wallet_jorge",
    workspace_name="es-smartwallet-pro-ml",
)

# 2. Cargar el YAML → objeto Component
comp_def = load_component(source="component.yaml")  

# 3. Registrar (crea o actualiza la versión)
comp = ml.components.create_or_update(comp_def)
print(f"✓ componente registrado: {comp.name}  v{comp.version}")
