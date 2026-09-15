# LiDAR configuration guide for better detection

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status.** The LiDAR settings discussed here come from a different point in the
> project's history and do not all match `configs/settings.json`, which ships the
> 64-channel configuration the dissertation states. The LiDAR is an auxiliary,
> visualisation-only channel — the detectors consume the depth-camera cloud — so
> no reported result depends on this. See configs/README.md.


## Problema Identificado
O LiDAR não estava detectando drones acima do horizonte na imagem devido ao FOV vertical limitado e orientação inadequada.

## Solution via settings.json

### Changes needed in settings.json

#### 1. Asymmetric vertical FOV (most important)
Ajustar o FOV para capturar mais área acima:

```json
"VerticalFOVUpper": 70,    // Era 45, agora 70 graus para cima
"VerticalFOVLower": -20,    // Era -45, agora só -20 graus para baixo
```

**Razão**: Como o drone voa alto, precisa "olhar mais para cima" do que para baixo.

#### 2. Increase the LiDAR resolution
```json
"NumberOfChannels": 128,        // Era 64, dobrou a resolução vertical
"PointsPerSecond": 2000000,     // Era 1000000, mais pontos por segundo
```

**Razão**: Mais canais = melhor resolução vertical = melhor detecção de objetos pequenos.

#### 3. Add an explicit orientation
```json
"Pitch": 0, "Roll": 0, "Yaw": 0,  // Define orientação explícita
```

#### 4. Make sure IgnoreMarked is False
```json
"IgnoreMarked": false  // CRUCIAL para detectar outros drones
```

## How to apply it

### Option 1: copy the new settings.json
```bash
# Fazer backup do atual
cp ~/airsim/settings.json ~/airsim/settings_backup.json

# Usar o melhorado
cp ~/airsim/settings_improved.json ~/airsim/settings.json

# OU copiar para a pasta do AirSim (se existir)
cp ~/airsim/settings_improved.json ~/.config/Epic/AirSim/settings.json
```

### Option 2: edit it by hand
Abra o `settings.json` e modifique a seção do LidarFront:

De:
```json
"VerticalFOVUpper": 45, "VerticalFOVLower": -45,
```

Para:
```json
"VerticalFOVUpper": 70, "VerticalFOVLower": -20,
```

## Validation

Após aplicar as mudanças:
1. Reinicie o simulador AirSim
2. Execute um script de teste
3. Verifique se drones acima do horizonte são detectados

## FOV comparison

### Original FOV (symmetric)
- Total: 90° (-45° a +45°)
- Problema: Perde objetos altos quando drone está voando

### Improved FOV (asymmetric)
- Total: 90° (-20° a +70°)
- Vantagem: 70° para cima captura drones altos
- Trade-off: Menos cobertura do chão (aceitável para detecção aérea)

## Alternative via script (without modifying settings.json)

Se não puder modificar o settings.json, use a técnica de múltiplas varreduras no script:

```python
# Simula FOV expandido com múltiplas capturas
PITCH_OFFSETS = [-0.4, -0.2, 0, 0.2, 0.4]
all_points = []

for pitch in PITCH_OFFSETS:
    client.rotateByYawPitchRollAsync(0, pitch, 0)
    lidar_data = client.getLidarData("LidarFront")
    # Processa e combina pontos
```

## Resultado Esperado

- Drones detectados em toda a altura da imagem
- Melhor cobertura vertical
- Detecções classificadas como "HIGH" (acima horizonte) e "LOW" (abaixo)

## Importante

ATENÇÃO: após modificar o settings.json, SEMPRE reinicie o simulador AirSim para aplicar as mudanças!