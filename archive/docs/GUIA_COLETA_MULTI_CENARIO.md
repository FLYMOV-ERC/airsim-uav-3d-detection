# Complete guide: multi-drone, multi-scenario dataset collection with AirSim

> **Archived.** This guide documents the multi-scenario collection workflow of
> `archive/dataset_generators/` (`collect_dataset_multi.py`,
> `run_multi_environments.py`, `scene_variation.py`), all of which are archived.
> It is **not** the workflow that produced the dataset reported in Chapter 7 —
> that is `dataset_generation/collect_airsim.py`. Kept as provenance; the body is
> in the original Portuguese and was not translated, because it describes an
> archived workflow.


## Visão Geral do Sistema Atual

Seu sistema já possui:
- **Suporte para múltiplos drones** (até 5: Ego + 4 Intruders)
- **Padrões de movimento diversos** (formação V, órbita circular, enxame, perseguição)
- **Variação de cenário** (clima, horário do dia)
- **Múltiplos ambientes** (Blocks, Neighborhood, City, Mountains, etc)
- **Coleta multi-sensor** (RGB, Segmentação, LiDAR)

## Como Usar o Sistema Atual

### 1. Coleta Multi-Drone Básica

```bash
# Coleta com 4 drones intrusores + ego
python3 collect-dataset-multi.py --frames 1000 --output_dir dataset_multi
```

### 2. Coleta com Variação de Cenário

```bash
# Com mudanças de clima e horário
python3 collect-dataset-multi.py \
    --frames 1000 \
    --weather_interval 100 \
    --time_interval 50 \
    --output_dir dataset_weather
```

### 3. Coleta em Múltiplos Ambientes

```bash
# Automatiza coleta em diferentes mapas
python3 run_multi_environments.py \
    --environments Blocks Neighborhood City \
    --frames_per_env 500
```

## Melhorias Propostas

### 1. Adicionar Mais Drones Dinamicamente

**Problema**: Atualmente limitado a 4 drones intrusores predefinidos.

**Solução**: Modificar `settings.json` e criar drones dinamicamente:

```json
{
  "SettingsVersion": 2.0,
  "LocalHostIp": "0.0.0.0",
  "ApiServerPort": 41451,
  "SimMode": "Multirotor",

  "Vehicles": {
    "Ego": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "Cameras": { ... }
    },

    // Adicione quantos drones quiser
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "X": -10, "Y": 0, "Z": -2
    },
    "Drone2": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "X": 10, "Y": 10, "Z": -3
    },
    "Drone3": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "X": 0, "Y": -10, "Z": -4
    },
    // ... até Drone10, Drone20, etc
  }
}
```

### 2. Cenários Customizados

**Crie cenários específicos** para seu caso de uso:

```python
# Cenário: Inspeção de infraestrutura
cenarios = {
    "inspecao_ponte": {
        "drones": ["Inspector1", "Inspector2", "Observer"],
        "pattern": "linear_scan",
        "weather": "clear",
        "time": "noon"
    },
    "busca_resgate": {
        "drones": ["Search1", "Search2", "Search3", "Rescue"],
        "pattern": "grid_search",
        "weather": "foggy",
        "time": "dawn"
    },
    "vigilancia_urbana": {
        "drones": ["Patrol1", "Patrol2", "Overwatch"],
        "pattern": "perimeter_patrol",
        "weather": "clear",
        "time": "night"
    }
}
```

### 3. Trocar Cenários em Tempo Real

**Sem reiniciar o simulador**, você pode:

```python
# Mudar clima
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.5)
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.8)

# Mudar horário
client.simSetTimeOfDay(True, "2024-01-01 18:30:00")  # Pôr do sol

# Mudar posição do sol
client.simSetSunAngle(math.radians(45))  # Ângulo do sol
```

### 4. Coletar Dados de Perspectivas Diferentes

**Configure câmeras em cada drone**:

```json
"Drone1": {
  "VehicleType": "SimpleFlight",
  "Cameras": {
    "front": {
      "CaptureSettings": [
        {"ImageType": 0, "Width": 640, "Height": 480}
      ],
      "X": 0.5, "Y": 0, "Z": 0
    }
  }
}
```

## Script Melhorado para Multi-Cenário

Vou criar um script que demonstra todas essas capacidades: