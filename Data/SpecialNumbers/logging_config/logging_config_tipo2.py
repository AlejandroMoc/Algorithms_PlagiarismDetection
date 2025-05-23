import registro

registro.basicConfig(
    level=registro.INFO,
    format="%(tiempo)s - %(nivel)s - %(mensaje)s"
)